#!/usr/bin/env python3
"""Rebuild Hippo evaluation inputs from WAVs, never from historical feature caches."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unicodedata
from itertools import groupby
import re
from tqdm.auto import tqdm

from feature_utils import (UTT, SSL_MODELS, BERT_MODEL, normalize_transcript,
                           phone_targets, pad_rows, word_mask)

ROOT = Path(__file__).resolve().parents[2]


def configure(models):
    models.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(models / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(models / 'huggingface' / 'hub')
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ['HF_HUB_CACHE']
    os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HUB_CACHE']
    os.environ['TORCH_HOME'] = str(models / 'torch')
    os.environ['XDG_CACHE_HOME'] = str(models / 'cache')
    os.environ['NLTK_DATA'] = str(models / 'nltk_data')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    import nltk
    validate_ctc(args.models)
    for repo in [*SSL_MODELS.values(), BERT_MODEL]:
        files = HfApi().list_repo_files(repo)
        weights = '*.safetensors' if any(p.endswith('.safetensors') for p in files) else 'pytorch_model*.bin'
        snapshot_download(repo, cache_dir=os.environ['HF_HUB_CACHE'],
                          allow_patterns=['*.json', '*.txt', '*.model', weights])
    nltk.data.path[:] = [os.environ['NLTK_DATA']]
    for resource in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict'):
        if not nltk.download(resource, download_dir=os.environ['NLTK_DATA'], raise_on_error=True):
            raise RuntimeError(f'NLTK download failed: {resource}')
    # g2p_en distributes its neural checkpoint as package data; keep the runtime copy here too.
    import importlib.util
    package = Path(importlib.util.find_spec('g2p_en').origin).parent
    target = args.models / 'g2p-en'
    target.mkdir(exist_ok=True)
    shutil.copy2(package / 'checkpoint20.npz', target / 'checkpoint20.npz')
    print(f'Model downloads complete: {args.models}', flush=True)


def validate_ctc(models):
    ctc = models / 'ctc-gop' / 'checkpoint-8000'
    proc = models / 'ctc-gop' / 'processor_config_gop'
    for path in (ctc / 'config.json', proc / 'preprocessor_config.json', proc / 'vocab.json'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing local CTC-GOP model file: {path}')
    if not any((ctc / name).is_file() for name in ('pytorch_model.bin', 'model.safetensors')):
        raise FileNotFoundError(f'Missing CTC-GOP weights: {ctc}')
    config = json.loads((ctc / 'config.json').read_text())
    vocab = json.loads((proc / 'vocab.json').read_text())
    if config['vocab_size'] != 40 or sorted(vocab.values()) != list(range(40)) or vocab.get('<pad>') != 0:
        raise ValueError('Hippo requires the original 40-token CTC vocabulary with blank=0')
    return ctc, proc, vocab


def read_audio(path):
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(path, dtype='float32')
    if rate != 16000 or audio.ndim != 1 or len(audio) < 400 or not np.isfinite(audio).all():
        raise ValueError(f'Expected finite mono 16 kHz audio, at least 400 samples: {path}')
    return audio


def g2p_processor(models):
    import numpy as np
    import nltk
    nltk.data.path[:] = [os.environ['NLTK_DATA']]
    # Check before importing g2p_en, whose import otherwise downloads to default locations.
    for resource in ('taggers/averaged_perceptron_tagger.zip', 'corpora/cmudict.zip'):
        nltk.data.find(resource)
    from g2p_en import G2p
    from g2p_en.expand import normalize_numbers
    from nltk.tokenize import TweetTokenizer

    class LocalG2p(G2p):
        def load_variables(self):
            with np.load(models / 'g2p-en' / 'checkpoint20.npz') as data:
                self.variables = {key: data[key] for key in data.files}
            for key, value in self.variables.items():
                setattr(self, key, value)

    g2p = LocalG2p()
    tokenize = TweetTokenizer().tokenize

    def convert(text):
        phones = [tuple(g) for k, g in groupby(g2p(text), key=lambda x: x != ' ') if k]
        cleaned = normalize_numbers(text)
        cleaned = ''.join(c for c in unicodedata.normalize('NFD', cleaned) if unicodedata.category(c) != 'Mn')
        cleaned = re.sub("[^ a-z'.,?!\\-]", '', cleaned.lower())
        cleaned = cleaned.replace('i.e.', 'that is').replace('e.g.', 'for example')
        tokens = tokenize(cleaned)
        if len(tokens) != len(phones):
            raise ValueError(f'G2P token/phone group mismatch: {text!r}')
        pairs = [(p, w) for p, w in zip(phones, tokens) if re.search(r'\w+\d?', p[0])]
        return [w for p, w in pairs], [list(p) for p, w in pairs]
    return convert


def release(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_array(folder, name, array):
    import numpy as np
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f'Nonfinite output: {name}')
    np.save(folder / f'te_{name}.npy', array)


def ctc_gop(log_probs, phones, max_phones):
    """Original 41 dimensions: total CTC NLL, then deletion/substitution NLL minus total NLL."""
    import torch
    import numpy as np
    log_probs = log_probs.float().cpu().contiguous()
    phones = torch.as_tensor(phones, dtype=torch.long)
    length = torch.tensor([len(log_probs)], dtype=torch.long)
    def loss(target):
        return torch.nn.functional.ctc_loss(log_probs, target, length,
                 torch.tensor([len(target)], dtype=torch.long), blank=0, reduction='sum')
    total = loss(phones)
    result = np.zeros((len(phones), 41), dtype=np.float32)
    for i in range(len(phones)):
        result[i, 0] = total.item()
        for replacement in range(40):
            target = phones.clone()
            if replacement == 0:
                target = torch.cat((target[:i], target[i+1:]))
            else:
                target[i] = replacement
            result[i, replacement+1] = (loss(target) - total).item()
    if not np.isfinite(result).all():
        raise ValueError('CTC alignment is impossible/nonfinite for this transcript')
    return pad_rows(result, max_phones)


def extract(args):
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, Wav2Vec2ForCTC
    ctc_path, proc_path, vocab = validate_ctc(args.models)
    refs = json.loads(args.scores.read_text())
    ids = sorted(refs)
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError('No input utterances')
    for key in ids:
        if not (args.wav_dir / f'{key}.wav').is_file():
            raise FileNotFoundError(args.wav_dir / f'{key}.wav')
    if args.output.exists():
        raise FileExistsError(f'Refusing to mix features: {args.output}. Select a new --feature-dir or remove it explicitly.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A failed extraction retains a clearly incomplete directory; no completion marker is published.
    work = Path(tempfile.mkdtemp(prefix=args.output.name + '.incomplete-', dir=args.output.parent))
    print(f'Extracting {len(ids)} utterances into {work}', flush=True)
    folder = work / ('seq_data_open' if args.response_mode == 'open' else 'seq_data_close')
    folder.mkdir()
    ssl_folder = work / 'seq_data_close'
    ssl_folder.mkdir(exist_ok=True)
    rows, transcripts = [], []
    failures = {}
    transcript_provenance = None
    resolved = None
    if args.response_mode == 'open':
        from transcripts import resolve_transcripts
        resolved, transcript_provenance = resolve_transcripts(
            ids, args.wav_dir, args.transcript_dir, args.whisper_model, args.models, args.device, args.threads)
    for key in tqdm(ids, desc="Transcript inputs", unit="utt", dynamic_ncols=True):
        ref = refs[key]
        path = args.wav_dir / f'{key}.wav'
        read_audio(path)
        text = normalize_transcript(resolved[key]) if resolved is not None else normalize_transcript(' '.join(w['text'] for w in ref['words']))
        if not text:
            failures[key] = 'empty_asr_transcript'
            tqdm.write(f'ASR fallback {key}: empty transcript; using MultiPA scores')
        transcripts.append(text)
        rows.append({'utt_id': key, 'audio_path': str(path.resolve()),
                     'ref_text': ' '.join(w['text'].lower() for w in ref['words']), 'asr_transcript': text,
                     'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with (work / 'et.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    convert = g2p_processor(args.models) if args.response_mode == 'open' else None
    phone_ids, phone_labels, word_labels, word_groups, word_texts = [], [], [], [], []
    for key, text in zip(ids, transcripts):
        ref = refs[key]
        if convert:
            words, groups = convert(text) if text else ([], [])
            if not groups or not any(groups):
                failures.setdefault(key, 'asr_has_no_phones')
            phone_count = sum(len(group) for group in groups)
            if phone_count > args.max_phones:
                failures[key] = f'asr_too_many_phones: {phone_count} > {args.max_phones}'
                tqdm.write(f'ASR fallback {key}: {failures[key]}; using MultiPA scores')
        else:
            words = [w['text'].lower() for w in ref['words']]
            groups = [w['phones'] for w in ref['words']]
        if key in failures:
            pids = np.empty(0, dtype=np.int64)
            pl, wl = np.empty((0, 2)), np.empty((0, 4))
            words, groups = [], []
        else:
            pids, pl, wl = phone_targets(ref, words, groups, vocab, closed=convert is None)
        phone_ids.append(pids)
        phone_labels.append(pad_rows(pl, args.max_phones, -1))
        phone_labels[-1][len(pl):, 1] = 0
        word_labels.append(pad_rows(wl, args.max_phones, -1))
        word_groups.append(groups)
        word_texts.append(words)
    (work / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    save_array(folder, 'label_phn', phone_labels)
    save_array(folder, 'label_word', word_labels)
    save_array(folder, 'label_utt', [[refs[key][k] for k in UTT] for key in ids])
    save_array(folder, 'word_atten_msk', [word_mask(w[:, -1]) for w in word_labels])
    local = dict(cache_dir=os.environ['HF_HUB_CACHE'], local_files_only=True)
    processor = AutoFeatureExtractor.from_pretrained(str(proc_path), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(ctc_path), local_files_only=True).eval().to(args.device) if len(failures) < len(ids) else None
    gops = []
    with torch.inference_mode():
        for i, (key, pids) in enumerate(tqdm(zip(ids, phone_ids), total=len(ids), desc="CTC-GOP", unit="utt", dynamic_ncols=True)):
            if key in failures:
                gops.append(np.zeros((args.max_phones, 41), dtype=np.float32))
                continue
            values = processor(read_audio(args.wav_dir / f'{key}.wav'), sampling_rate=16000, return_tensors='pt').input_values.to(args.device)
            logits = model(values).logits[0]
            if logits.shape[-1] != 40:
                raise ValueError('CTC output vocabulary differs from checkpoint contract')
            gops.append(ctc_gop(logits.log_softmax(-1), pids, args.max_phones))
    save_array(folder, 'feat_ctc_gop_af', gops)
    del model, gops
    release(None)
    for name, repo in SSL_MODELS.items():
        processor = AutoFeatureExtractor.from_pretrained(repo, **local)
        model = AutoModel.from_pretrained(repo, **local).eval().to(args.device) if len(failures) < len(ids) else None
        features = []
        with torch.inference_mode():
            for i, key in enumerate(tqdm(ids, desc=f"SSL {name}", unit="utt", dynamic_ncols=True)):
                if key in failures:
                    features.append(np.zeros(1024, dtype=np.float32))
                    continue
                values = processor(read_audio(args.wav_dir / f'{key}.wav'), sampling_rate=16000, return_tensors='pt').input_values.to(args.device)
                vector = model(values).last_hidden_state.mean(1)[0].cpu().numpy()
                if vector.shape != (1024,):
                    raise ValueError(f'{repo} must produce 1024-dimensional SSL vectors')
                features.append(vector)
        save_array(ssl_folder, f'utt_ssl_{name}_ali', features)
        del model, features
        release(None)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, **local)
    model = AutoModel.from_pretrained(BERT_MODEL, attn_implementation='eager', **local).eval().to(args.device) if len(failures) < len(ids) else None
    features = []
    with torch.inference_mode():
        for i, (words, groups) in enumerate(tqdm(zip(word_texts, word_groups), total=len(ids), desc="ModernBERT", unit="utt", dynamic_ncols=True)):
            if ids[i] in failures:
                features.append(np.zeros((args.max_phones, 768), dtype=np.float32))
                continue
            inputs = tokenizer(words, padding=True, return_tensors='pt').to(args.device)
            # Preserve original pooling, including special tokens and padded denominator.
            hidden = model(**inputs).last_hidden_state
            vectors = (hidden * inputs['attention_mask'].unsqueeze(-1)).mean(1)
            if vectors.shape[-1] != 768:
                raise ValueError('Hippo requires 768-dimensional ModernBERT word embeddings')
            word_ids = np.repeat(np.arange(len(words)), [len(p) for p in groups])
            features.append(pad_rows(vectors.cpu().numpy()[word_ids], args.max_phones))
    # Inference only consumes the first 768 dims of the historical 1536-dim array.
    save_array(folder, 'langUse_mbert_wordEmbs', features)
    metadata = {'schema_version': 4, 'text_normalization': 'multipa', 'transcript_provenance': transcript_provenance, 'failures': failures, 'source': 'fresh_audio', 'whisper_model': args.whisper_model,
                'response_mode': args.response_mode, 'max_phones': args.max_phones,
                'utterance_ids': ids, 'requested_ids': ids, 'scores_sha256': hashlib.sha256(args.scores.read_bytes()).hexdigest(),
                'models_dir': str(args.models), 'ssl_models': SSL_MODELS, 'word_model': BERT_MODEL,
                'manifest_sha256': hashlib.sha256((work / 'et.csv').read_bytes()).hexdigest()}
    import importlib.metadata
    metadata['versions'] = {name: importlib.metadata.version(name) for name in
                            ('torch', 'transformers', 'g2p-en')}
    metadata['feature_sha256'] = {str(path.relative_to(work)): sha256(path)
                                   for path in sorted(work.rglob('*.npy'))}
    metadata['ctc_sha256'] = {path.name: sha256(path) for path in sorted(ctc_path.iterdir())
                              if path.name in ('config.json', 'pytorch_model.bin', 'model.safetensors')}
    (work / 'features.json').write_text(json.dumps(metadata, indent=2) + '\n')
    work.rename(args.output)
    print(f'Fresh features complete: {args.output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['download', 'extract', 'check'])
    p.add_argument('--models', type=Path, default=ROOT / 'pretrained-models')
    p.add_argument('--whisper-model', default='large-v3')
    p.add_argument('--transcript-dir', type=Path, default=ROOT / 'data/speechocean762/transcript/test')
    p.add_argument('--response-mode', choices=['open', 'closed'], default='open')
    p.add_argument('--wav-dir', type=Path, default=ROOT / 'data/speechocean762/wav')
    p.add_argument('--scores', type=Path, default=ROOT / 'data/speechocean762/scores.json')
    p.add_argument('--output', type=Path, default=ROOT / 'data/features/open_large-v3')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-phones', type=int)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.max_phones is None:
        args.max_phones = 65 if args.response_mode == 'open' else 50
    if args.limit < 0 or args.max_phones < 1 or args.threads < 1:
        p.error('limit must be nonnegative; max-phones and threads must be positive')
    args.models = args.models.resolve()
    configure(args.models)
    if args.action == 'check':
        meta = json.loads((args.output / 'features.json').read_text())
        if meta['source'] != 'fresh_audio' or meta['whisper_model'] != args.whisper_model or meta['response_mode'] != args.response_mode:
            raise ValueError('Feature origin/ASR configuration mismatch; rerun extraction into a new directory')
        if meta['scores_sha256'] != hashlib.sha256(args.scores.read_bytes()).hexdigest():
            raise ValueError('Scores changed since extraction')
        if meta['manifest_sha256'] != hashlib.sha256((args.output / 'et.csv').read_bytes()).hexdigest():
            raise ValueError('Feature manifest changed since extraction')
        for filename, expected in meta['feature_sha256'].items():
            if sha256(args.output / filename) != expected:
                raise ValueError(f'Feature array changed since extraction: {filename}')
        return
    import torch
    torch.set_num_threads(args.threads)
    if args.action == 'download':
        download(args)
    else:
        extract(args)


if __name__ == '__main__':
    main()
