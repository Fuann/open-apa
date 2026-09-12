import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gc
import json
import argparse
import random
import torch
import fairseq
import whisper
import numpy as np
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
from dataclasses import dataclass
from utils_assessment import *
from model_assessment import PronunciationPredictor
from Charsiu import charsiu_forced_aligner

gc.collect()
torch.cuda.empty_cache()

from fairseq.data.dictionary import Dictionary
torch.serialization.add_safe_globals([Dictionary])
torch.serialization.add_safe_globals([argparse.Namespace])


SEED = 1984


def configure_deterministic_inference():
    """Make supported Python, NumPy, and PyTorch inference operations reproducible."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def configure_fairseq_spacy_tokenizer():
    """Use spaCy 3's tokenizer with fairseq 0.12.2 word alignment.

    fairseq 0.12.2 calls ``EnglishDefaults.create_tokenizer``, an API removed
    in spaCy 3. Setting fairseq's cached tokenizer directly keeps the original
    alignment logic while avoiding that obsolete factory method.
    """
    from fairseq.models.roberta import alignment_utils
    from spacy.lang.en import English

    nlp = English()
    alignment_utils.spacy_nlp._nlp = nlp
    alignment_utils.spacy_tokenizer._tokenizer = nlp.tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, default='./fairseq_hubert/hubert_base_ls960.pt', help='Path to pretrained fairseq hubert model.')
    parser.add_argument('--fairseq_roberta', type=str, default='./fairseq_roberta', help='Path to pretrained fairseq roberta.')
    parser.add_argument('--datadir', default='./multipa/wav', type=str, help='Path of your DATA/ directory')
    parser.add_argument('--datalist', default='./multipa/multipa_test.txt', type=str, help='')
    parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')
    parser.add_argument('--output-file', type=str, default=None, help='Path to save prediction results.')
    parser.add_argument('--whisper-model', default='large-v3', choices=whisper.available_models(),
                        help='Whisper model for the main open-response transcript.')
    parser.add_argument('--transcripts', '--reference-scores', dest='transcripts', type=str, default=None,
                        help='Optional scores.json whose reference words replace the main Whisper transcript.')
    parser.add_argument('--open-transcripts', type=str, default=None,
                        help='JSONL with fixed open-response transcript and audio_id fields.')
    parser.add_argument('--verbose', action='store_true', help='Print each audio prediction (default: disabled).')


    args = parser.parse_args()
    if args.transcripts is not None and args.open_transcripts is not None:
        parser.error('--transcripts and --open-transcripts are mutually exclusive')
    configure_deterministic_inference()
    configure_fairseq_spacy_tokenizer()

    transcripts = None
    if args.transcripts is not None:
        with open(args.transcripts, encoding='utf-8') as transcript_file:
            samples = json.load(transcript_file)
        transcripts = {}
        for sample_id, sample in samples.items():
            text = ' '.join(str(word['text']) for word in sample['words'])
            text = remove_pun_except_apostrophe(text).lower()
            transcripts[sample_id] = convert_num_to_word(text)
    elif args.open_transcripts is not None:
        transcripts = {}
        with open(args.open_transcripts, encoding='utf-8') as transcript_file:
            for line_number, line in enumerate(transcript_file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = os.path.splitext(os.path.basename(row['audio_id']))[0]
                if sample_id in transcripts:
                    raise ValueError(f'Duplicate transcript for {sample_id} at line {line_number}')
                text = remove_pun_except_apostrophe(str(row['transcript'])).lower()
                transcripts[sample_id] = convert_num_to_word(text)
    
    ssl_path = args.fairseq_base_model
    roberta_path = args.fairseq_roberta
    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir
    datalist = args.datalist


    word_model = RobertaModel.from_pretrained(roberta_path, checkpoint_file='model.pt')
    word_model.eval()
    whisper_model_s = None
    if transcripts is None:
        # Convert on CPU first so large-v3 never occupies GPU memory in FP32.
        whisper_model_s = whisper.load_model(args.whisper_model, device='cpu')
        if torch.cuda.is_available():
            whisper_model_s = whisper_model_s.half()
            # Whisper LayerNorm computes in FP32 and needs FP32 parameters.
            for module in whisper_model_s.modules():
                if isinstance(module, nn.LayerNorm):
                    module.float()
            whisper_model_s = whisper_model_s.cuda()
    whisper_model_w = whisper.load_model("base.en")

    aligment_model = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

    SSL_OUT_DIM = 768
    TEXT_OUT_DIM = 768
    SAMPLE_RATE = 16000
    
    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    ssl_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
    ssl_model = ssl_model[0]
   
    assessment_model = PronunciationPredictor(ssl_model, SSL_OUT_DIM, TEXT_OUT_DIM).to(device)
    assessment_model.eval()
    ckpt = torch.load(os.path.join(my_checkpoint_dir,'PRO'+os.sep+'best'), weights_only=False)
    assessment_model.load_state_dict(ckpt)

    print('Loading data')
    validset = open(datalist,'r').read().splitlines()
    if args.output_file is None:
        outfile = os.path.basename(os.path.normpath(my_checkpoint_dir)) + '_' + os.path.basename(datalist).replace('.txt','_mb.txt')
        output_file = os.path.join('Results', outfile)
    else:
        output_file = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    prediction = open(output_file, 'w')
    
    print('Starting prediction')
    for filename in tqdm(validset):
        
        with torch.no_grad():
            if datalist is not None:
                filepath = os.path.join(datadir, filename)
            else:
                filepath=filename
            wav, sr = torchaudio.load(filepath)
            #resample audio recordin to 16000Hz
            if sr!=16000:
                transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = transform(wav)
                sr = SAMPLE_RATE

            wav  = torch.reshape(wav, (-1,))
            if transcripts is None:
                sen_asr_s = remove_pun_except_apostrophe(get_transcript(wav, whisper_model_s, language='en')).lower()
                sen_asr_s = convert_num_to_word(sen_asr_s)
            else:
                sample_id = os.path.splitext(os.path.basename(filename))[0]
                if sample_id not in transcripts:
                    raise KeyError(f'Missing ground-truth transcript for {sample_id}')
                sen_asr_s = transcripts[sample_id]

            sen_asr_w = remove_pun_except_apostrophe(get_transcript(wav, whisper_model_w)).lower()
            sen_asr_w = convert_num_to_word(sen_asr_w)

            try:
                pred_words_gt, features_p, features_w, phonevector, gt_word_embed, asr_word_embed, word_phone_map = feature_extraction(wav.numpy(), sen_asr_s, sen_asr_w, alignment_model=aligment_model, word_model=word_model)

                timesplit =  [[(int(float(word[0])*SAMPLE_RATE), int(float(word[1])*SAMPLE_RATE)) for word in pred_words_gt]] 
                word_phone_map = [word_phone_map]

                features_p = torch.from_numpy(features_p).to(device).float().unsqueeze(0)
                features_w = torch.from_numpy(features_w).to(device).float().unsqueeze(0) 
                phonevector = torch.from_numpy(phonevector).to(device).float().unsqueeze(0) 
                gt_word_embed = torch.from_numpy(gt_word_embed).to(device).float().unsqueeze(0) 
                asr_word_embed = torch.from_numpy(asr_word_embed).to(device).float().unsqueeze(0) 
                wav = wav.to(device).unsqueeze(0)
                
                score_A, score_F, score_P, score_T, w_acc, w_stress, w_total = assessment_model(wav, asr_word_embed, gt_word_embed, features_p, features_w, phonevector, word_phone_map, timesplit)
                score_A = score_A.cpu().detach().numpy()[0]
                score_F = score_F.cpu().detach().numpy()[0]
                score_P = score_P.cpu().detach().numpy()[0]
                score_T = score_T.cpu().detach().numpy()[0] 
                w_a = w_acc.cpu().detach().numpy()[0]
                w_s = w_stress.cpu().detach().numpy()[0]
                w_t = w_total.cpu().detach().numpy()[0]
                
                w_a = ','.join([str(num) for num in w_a])
                w_s = ','.join([str(num) for num in w_s])
                w_t = ','.join([str(num) for num in w_t])

                valid = 'T'
                output = "{}; A:{}; F:{}; P:{}; T:{}; Valid:{}; ASR_s:{}; ASR_w:{}; w_a:{}; w_s:{}; w_t:{}; alignment:{}".format(filename, score_A, score_F, score_P, score_T, valid, sen_asr_s, sen_asr_w, w_a, w_s, w_t, pred_words_gt.tolist())
                if args.verbose:
                    print(output)
                prediction.write(output+'\n')

            except Exception as e:
                valid = 'F'
                tqdm.write(
                    f"{filename} failed: {type(e).__name__}: {e}; "
                    "marked as Valid:F"
                )
                output = "{}; A:{}; F:{}; P:{}; T:{}; Valid:{}; ASR_s:{}; ASR_w:{}; w_a:{}; w_s:{}; w_t:{}; alignment:{}".format(filename, '', '', '', '', valid, sen_asr_s, sen_asr_w, '', '', '', '')
                prediction.write(output+'\n')
                continue
               

            torch.cuda.empty_cache()

    prediction.close()
 

if __name__ == '__main__':
    main()
