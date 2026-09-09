#!/usr/bin/env python3
"""Prepare MultiPA timestamp-overlap targets for Hippo's predicted words."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MULTIPA = ROOT.parent / 'multipa'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = args.feature_dir / 'et.csv'
    fingerprint = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                   [('manifest', manifest), ('scores', args.scores)]}
    if args.output.exists() and json.loads(args.output.read_text()).get('fingerprint') == fingerprint:
        return
    cache = Path.home() / '.cache/huggingface'
    for name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE'):
        os.environ[name] = str(cache / 'hub')
    os.environ['HF_HOME'] = str(cache)
    os.environ['HF_HUB_OFFLINE'] = '1'
    # Isolate the legacy Charsiu modules from Hippo's modules of the same names.
    sys.path.insert(0, str(MULTIPA / 'src'))
    import numpy as np
    import torch
    from tqdm import tqdm
    torch.set_num_threads(4)
    from evaluate_speechocean762 import overlap_indices
    sys.path.append(str(ROOT / 'src/feats_extract'))
    from extract_features import g2p_processor
    convert = g2p_processor(ROOT / 'pretrained-models')
    refs = json.loads(args.scores.read_text())
    failures = json.loads((args.feature_dir / 'features.json').read_text()).get('failures', {})
    rows = list(csv.DictReader(manifest.open()))
    cached = {}
    for path in (MULTIPA / 'exp').glob('*/decode_speechocean762*/test_mb.txt'):
        for line in path.read_text().splitlines():
            parts = line.split(';')
            if len(parts) >= 12 and parts[5].split(':', 1)[1].strip() == 'T':
                alignment = ast.literal_eval(parts[11].split(':', 1)[1].strip())
                cached[(Path(parts[0].strip()).stem, tuple(str(w[2]).lower() for w in alignment))] = alignment
    aligner = None
    output = {}
    evaluation_failures = {}
    for row in tqdm(rows, desc='MultiPA word evaluation'):
        key = row['utt_id']
        if key in failures:
            continue
        words, _ = convert(row['asr_transcript'])
        reference = refs[key]['words']
        ref_words = [w['text'].lower() for w in reference]
        alignment = None
        score_indices = list(range(len(words)))
        if words == ref_words:
            mapping = [[i] for i in range(len(words))]
        else:
            alignment = cached.get((key, tuple(words)))
            if alignment is None:
                if aligner is None:
                    from Charsiu import charsiu_forced_aligner
                    aligner = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', device='cpu')
                _, aligned, aligned_words, _, _, _ = aligner.align(audio=row['audio_path'], text=row['asr_transcript'])
                alignment = []
                start = 0
                score_indices = []
                for word_index, word in enumerate(words):
                    for i in range(start, len(aligned)):
                        if aligned[i][2] == word:
                            alignment.append(aligned[i]); score_indices.append(word_index); start = i + 1; break
                if not alignment:
                    raise ValueError(f'Charsiu produced no aligned words: {key}')
            gt_path = MULTIPA / 'data/speechocean762/gt-alignments' / f'{key}.pt'
            if not gt_path.exists():
                if aligner is None:
                    from Charsiu import charsiu_forced_aligner
                    aligner = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', device='cpu')
                from librosa.util.exceptions import ParameterError
                try:
                    _, aligned_gt, gt_words, _, _, _ = aligner.align(audio=row['audio_path'], text=' '.join(ref_words))
                except ParameterError as error:
                    evaluation_failures[key] = 'ground_truth_alignment_failed: ' + str(error)
                    continue
                gt = []
                start = 0
                for word in gt_words:
                    for i in range(start, len(aligned_gt)):
                        if aligned_gt[i][2] == word:
                            gt.append(aligned_gt[i]); start = i + 1; break
                torch.save(np.asarray(gt), gt_path)
            gt = torch.load(gt_path, weights_only=False)
            mapping = overlap_indices(alignment, gt)
        defaults = {'accuracy': 0., 'stress': 5., 'total': 1.}
        targets = [[float(np.mean([reference[i][metric] for i in indices])) if indices else defaults[metric]
                    for metric in defaults] for indices in mapping]
        output[key] = {'words': words, 'targets': targets, 'score_indices': score_indices, 'alignment': alignment or [[0., 0., w] for w in words]}
    args.output.write_text(json.dumps({'fingerprint': fingerprint, 'protocol': 'multipa_timestamp_overlap', 'items': output, 'failures': evaluation_failures}, indent=2)+'\n')


if __name__ == '__main__':
    main()
