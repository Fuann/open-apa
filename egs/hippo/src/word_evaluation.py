"""Shared open-response word evaluation for inference and final reports."""
import csv
import hashlib
import json

import numpy as np


def load_word_evaluation(path, manifest, scores):
    evaluation = json.loads(path.read_text())
    expected = {name: hashlib.sha256(source.read_bytes()).hexdigest()
                for name, source in [('manifest', manifest), ('scores', scores)]}
    if evaluation.get('protocol') != 'levenshtein_match_substitution':
        raise ValueError('Expected Levenshtein match+substitution word evaluation')
    if evaluation.get('fingerprint') != expected:
        raise ValueError('Word evaluation manifest/scores mismatch')
    with manifest.open() as handle:
        ids = [row['utt_id'] for row in csv.DictReader(handle)]
    return evaluation, ids


def open_word_arrays(raw, raw_target, ids, evaluation, failures):
    """Pool phone predictions on the 0–10 scale; retain M+S and cap at 10."""
    if len(raw) != len(ids) or len(raw_target) != len(ids):
        raise ValueError('Word evaluation/prediction row count mismatch')
    pred_words, target_words = [], []
    for i, key in enumerate(ids):
        if key in failures or key in evaluation.get('failures', {}):
            continue
        item = evaluation['items'][key]
        wids = raw_target[i, :, -1].astype(int)
        unique = np.unique(wids[wids >= 0])
        if unique.tolist() != list(range(len(item['words']))):
            raise ValueError(f'Word evaluation/prediction token mismatch: {key}')
        indices = item['score_indices']
        if (len(indices) != len(item['targets']) or len(set(indices)) != len(indices)
                or any(w not in unique for w in indices)):
            raise ValueError(f'Invalid word evaluation indices/targets: {key}')
        pred_words.extend(np.minimum(10., raw[i, wids == w].mean(0)) for w in indices)
        target_words.extend(item['targets'])
    return (np.asarray(pred_words).reshape(-1, 3),
            np.asarray(target_words).reshape(-1, 3))
