"""Shared open-response word evaluation for inference and final reports."""
import csv
import hashlib
import json

import numpy as np


def load_word_evaluation(path, manifest, scores):
    evaluation = json.loads(path.read_text())
    expected = {name: hashlib.sha256(source.read_bytes()).hexdigest()
                for name, source in [('manifest', manifest), ('scores', scores)]}
    if evaluation.get('protocol') != 'levenshtein_match_substitution_scopes':
        raise ValueError('Expected scoped Levenshtein word evaluation')
    if evaluation.get('fingerprint') != expected:
        raise ValueError('Word evaluation manifest/scores mismatch')
    with manifest.open() as handle:
        ids = [row['utt_id'] for row in csv.DictReader(handle)]
    return evaluation, ids


def open_word_arrays(raw, raw_target, ids, evaluation, failures):
    """Return capped pooled word arrays for match, substitution, and M+S."""
    if len(raw) != len(ids) or len(raw_target) != len(ids):
        raise ValueError('Word evaluation/prediction row count mismatch')
    arrays = {scope: ([], []) for scope in ('M', 'S', 'M+S')}
    for i, key in enumerate(ids):
        if key in failures or key in evaluation.get('failures', {}):
            continue
        item = evaluation['items'][key]
        wids = raw_target[i, :, -1].astype(int)
        unique = np.unique(wids[wids >= 0])
        if unique.tolist() != list(range(len(item['words']))):
            raise ValueError(f'Word evaluation/prediction token mismatch: {key}')
        indices = item['score_indices']
        operations = item.get('operations', [])
        if (len(indices) != len(item['targets']) or len(indices) != len(operations)
                or any(operation not in ('match', 'substitution') for operation in operations)
                or len(set(indices)) != len(indices)
                or any(w not in unique for w in indices)):
            raise ValueError(f'Invalid word evaluation indices/targets: {key}')
        for word_index, target, operation in zip(indices, item['targets'], operations):
            prediction = np.minimum(10., raw[i, wids == word_index].mean(0))
            scope = 'M' if operation == 'match' else 'S'
            for selected_scope in (scope, 'M+S'):
                arrays[selected_scope][0].append(prediction)
                arrays[selected_scope][1].append(target)
    return {scope: (np.asarray(prediction).reshape(-1, 3),
                    np.asarray(target).reshape(-1, 3))
            for scope, (prediction, target) in arrays.items()}
