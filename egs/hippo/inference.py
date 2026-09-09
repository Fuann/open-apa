"""Run released HiPPO checkpoints on prepared evaluation features."""
import sys
import json
import hashlib
import os
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".deps"))


from feats_extract.feature_utils import UTT_FALLBACK

import argparse
import torch
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
import random


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument('--seed', type=int, required=True)
parser.add_argument("--p_depth", type=int, default=3, help="depth of HiPPO model")
parser.add_argument("--w_depth", type=int, default=2, help="depth of HiPPO model")
parser.add_argument("--u_depth", type=int, default=1, help="depth of HiPPO model")
parser.add_argument("--goptheads", type=int, default=1, help="heads of HiPPO model")
parser.add_argument("--batch_size", type=int, default=25, help="inference batch size")
parser.add_argument("--embed_dim", type=int, default=24, help="HiPPO embedding dimension")
parser.add_argument("--ssl_drop", type=float, default=0.1, help="SSL dropout probability")

parser.add_argument("--feature-dir", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--response-mode", choices=["open", "closed"], default="open")
parser.add_argument("--asr-variant", choices=["med", "large", "small", "fresh"], default="med")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--device", default="cpu")

# just to generate the header for the result.csv
def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header
    return header

def infer(audio_model, test_open_loader, args):
    device = torch.device(args.device)
    result_open = np.zeros([1, 32])
    exp_dir = args.exp_dir
    print('running on ' + str(device))

    print('start validation')
    te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr = validate(audio_model, test_open_loader, args)
    tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr = te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr
    print('-------------------openend test-------------------')
    print('Phone: Test MSE: {:.3f}, CORR: {:.3f}'.format(te_mse.item(), te_corr))
    print('Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PROC: {:.3f}, Total: {:.3f}'.format(te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4]))
    print('Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(te_word_corr[0], te_word_corr[1], te_word_corr[2]))

    result_open[0, :6] = [0, tr_mse, tr_corr, te_mse, te_corr, 0]
    result_open[0, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])
    result_open[0, 26:32] = np.concatenate([tr_word_corr, te_word_corr])

    header = ','.join(gen_result_header())
    np.savetxt(exp_dir + '/result.csv', result_open, delimiter=',', header=header, comments='')

def validate(audio_model, val_loader, args):
    device = torch.device(args.device)
    # This recipe selects one device; DataParallel can select GPUs even for CPU runs.
    if isinstance(audio_model, nn.DataParallel):
        audio_model = audio_model.module
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []
    with torch.no_grad():
        for i, (audio_input, audio_input_utt_ssl1, audio_input_utt_ssl2,
         audio_input_utt_ssl3, phn_label, phns, utt_label, word_label, word_pos,
          word_mbert_feat, word_atten_msk) in enumerate(tqdm(val_loader, desc="Hippo inference", unit="batch", dynamic_ncols=True)):
            phns = phns.to(device)
            audio_input = audio_input.to(device)
            audio_input_utt_ssl1 = audio_input_utt_ssl1.to(device)
            audio_input_utt_ssl2 = audio_input_utt_ssl2.to(device)
            audio_input_utt_ssl3 = audio_input_utt_ssl3.to(device)
            word_atten_msk = word_atten_msk.to(device)
            word_pos = word_pos.to(device)
            word_mbert_feat = word_mbert_feat.to(device)
            audio_input_utt_ssl = torch.cat([audio_input_utt_ssl2, audio_input_utt_ssl1, audio_input_utt_ssl3], dim=-1)
            # compute output
            valid_rows = (phn_label >= 0).any(dim=1).to(device)
            batch, length = audio_input.shape[:2]
            outputs = [torch.zeros(batch, 1, device=device) for _ in range(5)]
            outputs += [torch.zeros(batch, length, 1, device=device) for _ in range(4)]
            # Failed ASR rows never enter Hippo. Completeness has no MultiPA default;
            # its internal zero placeholder is excluded from evaluation for these rows.
            outputs[0][:] = UTT_FALLBACK['accuracy'] / 5
            if valid_rows.any():
                predicted = audio_model(audio_input[valid_rows], audio_input_utt_ssl[valid_rows],
                    phns[valid_rows], word_pos[valid_rows], word_mbert_feat[valid_rows], word_atten_msk[valid_rows])
                for output, prediction in zip(outputs, predicted):
                    output[valid_rows] = prediction
            u1, u2, u3, u4, u5, p, w1, w2, w3 = outputs
            p = p.to('cpu').detach()
            u1, u2, u3, u4, u5 = u1.to('cpu').detach(), u2.to('cpu').detach(), u3.to('cpu').detach(), u4.to('cpu').detach(), u5.to('cpu').detach()
            w1, w2, w3 = w1.to('cpu').detach(), w2.to('cpu').detach(), w3.to('cpu').detach()

            A_phn.append(p)
            A_phn_target.append(phn_label)

            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label)

            A_w1.append(w1)
            A_w2.append(w2)
            A_w3.append(w3)
            A_word_target.append(word_label)

        # phone level
        A_phn, A_phn_target  = torch.cat(A_phn), torch.cat(A_phn_target)

        # utterance level
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)

        # word level
        A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

        # get the scores
        phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)

        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_mse, utt_corr = valid_utt(A_utt, A_utt_target, (A_phn_target >= 0).any(dim=1))


        A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
        word_mse, word_corr, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

        print('Saving predictions, including explicit ASR fallback rows.')

        # create the directory
        if os.path.exists(args.exp_dir + '/preds') == False:
            os.mkdir(args.exp_dir + '/preds')

        # Keep targets synchronized when an experiment directory is rerun.
        np.save(args.exp_dir + '/preds/phn_target.npy', A_phn_target)
        np.save(args.exp_dir + '/preds/word_target.npy', valid_word_target)
        np.save(args.exp_dir + '/preds/utt_target.npy', A_utt_target)

        np.save(args.exp_dir + '/preds/phn_pred.npy', A_phn)
        np.save(args.exp_dir + '/preds/word_pred.npy', valid_word_pred)
        np.save(args.exp_dir + '/preds/utt_pred.npy', A_utt)
        np.save(args.exp_dir + '/preds/word_phone_pred.npy', A_word)
        np.save(args.exp_dir + '/preds/word_phone_target.npy', A_word_target)

    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr

def safe_corr(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if len(prediction) < 2 or np.std(prediction) == 0 or np.std(target) == 0:
        return float('nan')
    return np.corrcoef(prediction, target)[0, 1]


def valid_phn(audio_output, target):
    valid_token_pred = []
    valid_token_target = []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            # only count valid tokens, not padded tokens (represented by negative values)
            if target[i, j] >= 0:
                valid_token_pred.append(audio_output[i, j])
                valid_token_target.append(target[i, j])
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)

    if not len(valid_token_pred):
        return np.float32('nan'), float('nan')
    valid_token_mse = np.mean((valid_token_target - valid_token_pred) ** 2)
    corr = safe_corr(valid_token_pred, valid_token_target)
    return valid_token_mse, corr

def valid_utt(audio_output, target, valid_rows=None):
    mse = []
    corr = []
    for i in range(5):
        prediction, reference = audio_output[:, i], target[:, i]
        if i == 1 and valid_rows is not None:
            prediction, reference = prediction[valid_rows], reference[valid_rows]
        cur_mse = np.mean(((prediction - reference) ** 2).numpy()) if len(prediction) else float('nan')
        cur_corr = safe_corr(prediction, reference)
        mse.append(cur_mse)
        corr.append(cur_corr)
    return mse, corr

def valid_word(audio_output, target):
    word_id = target[:, :, -1]
    target = target[:, :, 0:3]

    valid_token_pred = []
    valid_token_target = []



    for i in range(target.shape[0]):
        # Include the last word even when the sequence fills the entire tensor.
        ids = word_id[i].numpy()
        valid_length = int(np.sum(ids >= 0))
        boundaries = [0] + (np.flatnonzero(np.diff(ids[:valid_length])) + 1).tolist() + [valid_length]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end > start:
                valid_token_pred.append(audio_output[i, start:end].numpy().mean(axis=0))
                valid_token_target.append(target[i, start:end].numpy().mean(axis=0))

    valid_token_pred = np.array(valid_token_pred).reshape(-1, 3)
    if not len(valid_token_pred):
        return [float("nan")]*3, [float("nan")]*3, valid_token_pred, np.empty((0, 3))
    # this rounding is to solve the precision issue in the label
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list, corr_list = [], []
    # for each (accuracy, stress, total) word score
    for i in range(3):
        valid_token_mse = np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)
        corr = safe_corr(valid_token_pred[:, i], valid_token_target[:, i])
        mse_list.append(valid_token_mse)
        corr_list.append(corr)
    return mse_list, corr_list, valid_token_pred, valid_token_target


class GoPDataset(Dataset):
    def __init__(self, feature_dir, response_mode="open", asr_variant="med"):
        folder = feature_dir / ("seq_data_open" if response_mode == "open" else "seq_data_close")
        suffix = "_" + asr_variant if response_mode == "open" and asr_variant not in ("large", "fresh") else ""
        def read(name, ssl=False):
            path = feature_dir / "seq_data_close" / f"te_{name}.npy" if ssl else folder / f"te_{name}{suffix}.npy"
            return torch.from_numpy(np.array(np.load(path, mmap_mode="r"), dtype=np.float32))
        self.feat = read("feat_ctc_gop_af")
        self.utt_ssl_feat1 = read("utt_ssl_hu_ali", True)
        self.utt_ssl_feat2 = read("utt_ssl_w2v_ali", True)
        self.utt_ssl_feat3 = read("utt_ssl_wlm_ali", True)
        self.word_mbert_feat = read("langUse_mbert_wordEmbs")
        self.phn_label = read("label_phn")
        self.utt_label = read("label_utt")
        self.word_label = read("label_word")
        self.word_atten_msk = read("word_atten_msk")
        for name, value in vars(self).items():
            if len(value) != len(self.feat) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid feature array: {name}")

        self.feat = self.norm_valid(self.feat, 9.0098, 6.9739)
        # normalize the utt_label to 0-2 (same with phn score range)
        self.utt_label = self.utt_label / 5
        # the last dim is word_id, so not normalizing
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5

    # only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        # feat, phn_label, phn_id, utt_label, word_label
        return self.feat[idx, :], self.utt_ssl_feat1[idx, :], self.utt_ssl_feat2[idx, :], self.utt_ssl_feat3[idx, :],\
         self.phn_label[idx, :, 0], self.phn_label[idx, :, 1], self.utt_label[idx, :], self.word_label[idx, :], self.word_label[idx, :, -1], self.word_mbert_feat[idx, :, :768], self.word_atten_msk[idx, :]


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    # NOTE: set seed
    print("setting seed %d" %(args.seed))
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from models.hippo import GOPT
    audio_mdl = GOPT(embed_dim=args.embed_dim, num_heads=args.goptheads,
                     p_depth=args.p_depth, w_depth=args.w_depth, u_depth=args.u_depth,
                     ssl_drop=args.ssl_drop, input_dim=41)
    print(f'Loading checkpoint: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    audio_mdl.load_state_dict(state_dict, strict=True)
    audio_mdl.eval()

    te_dataset_open = GoPDataset(args.feature_dir, args.response_mode, args.asr_variant)
    te_dataloader_open = DataLoader(te_dataset_open, batch_size=args.batch_size, shuffle=False)

    infer(audio_mdl, te_dataloader_open, args)
    if args.asr_variant == 'fresh':
        provenance = args.feature_dir / 'features.json'
        (Path(args.exp_dir) / 'feature_provenance.json').write_text(json.dumps({
            'features_sha256': hashlib.sha256(provenance.read_bytes()).hexdigest(),
            'phone_id_offset': 1,
            'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'model_source_sha256': hashlib.sha256((ROOT / 'src/models/hippo.py').read_bytes()).hexdigest()
        }) + '\n')
