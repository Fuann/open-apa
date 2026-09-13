#!/usr/bin/env python3
"""Prepare deterministic Levenshtein M+S targets for Hippo open evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def align_words(reference, hypothesis):
    """Return one reference index per hypothesis word plus M/S/I/D counts."""
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    operation = [[None] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        cost[row][0], operation[row][0] = row, 'deletion'
    for column in range(1, columns + 1):
        cost[0][column], operation[0][column] = column, 'insertion'
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = 'match' if reference[row - 1] == hypothesis[column - 1] else 'substitution'
            cost[row][column], operation[row][column] = min([
                (cost[row - 1][column - 1] + (diagonal != 'match'), diagonal),
                (cost[row - 1][column] + 1, 'deletion'),
                (cost[row][column - 1] + 1, 'insertion'),
            ], key=lambda item: item[0])
    mapping = [None] * columns
    aligned_operations = [None] * columns
    counts = {name: 0 for name in ('match', 'substitution', 'insertion', 'deletion')}
    row, column = rows, columns
    while row or column:
        current = operation[row][column]
        counts[current] += 1
        if current in ('match', 'substitution'):
            mapping[column - 1] = row - 1
            aligned_operations[column - 1] = current
            row, column = row - 1, column - 1
        elif current == 'deletion':
            row -= 1
        else:
            column -= 1
    return mapping, aligned_operations, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--statistics-output', type=Path, required=True)
    p.add_argument('--transcript', type=Path, required=True)
    args = p.parse_args()
    manifest = args.feature_dir / 'et.csv'
    fingerprint = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                   [('manifest', manifest), ('scores', args.scores)]}
    if args.output.exists() and args.statistics_output.exists():
        cached = json.loads(args.output.read_text())
        if (cached.get('fingerprint') == fingerprint
                and cached.get('protocol') == 'levenshtein_match_substitution_scopes'):
            return
    sys.path.append(str(ROOT / 'src/feats_extract'))
    from extract_features import g2p_processor
    convert = g2p_processor(ROOT / 'pretrained-models')
    refs = json.loads(args.scores.read_text())
    failures = json.loads((args.feature_dir / 'features.json').read_text()).get('failures', {})
    rows = list(csv.DictReader(manifest.open()))
    output = {}
    totals = {name: 0 for name in ('match', 'substitution', 'insertion', 'deletion')}
    reference_count = hypothesis_count = 0
    for row in rows:
        key = row['utt_id']
        reference = refs[key]['words']
        ref_words = [w['text'].lower() for w in reference]
        words, _ = convert(row['asr_transcript']) if row['asr_transcript'] else ([], [])
        mapping, aligned_operations, counts = align_words(ref_words, words)
        for name in totals:
            totals[name] += counts[name]
        reference_count += len(ref_words)
        hypothesis_count += len(words)
        if key in failures:
            continue
        score_indices = [i for i, reference_index in enumerate(mapping) if reference_index is not None]
        targets = [[float(reference[mapping[i]][metric]) for metric in ('accuracy', 'stress', 'total')]
                   for i in score_indices]
        operations = [aligned_operations[i] for i in score_indices]
        output[key] = {'words': words, 'targets': targets,
                       'score_indices': score_indices, 'operations': operations}
    evaluated = totals['match'] + totals['substitution']
    statistics = {
        'num_matches': totals['match'], 'num_substitutions': totals['substitution'],
        'num_insertions': totals['insertion'], 'num_deletions': totals['deletion'],
        'num_reference_words': reference_count, 'num_asr_words': hypothesis_count,
        'num_evaluated_words': evaluated,
        'evaluated_coverage': evaluated / reference_count if reference_count else 0.,
        'wer': (totals['substitution'] + totals['insertion'] + totals['deletion']) / reference_count if reference_count else 0.,
    }
    args.output.write_text(json.dumps({'fingerprint': fingerprint,
        'protocol': 'levenshtein_match_substitution_scopes',
        'items': output, 'failures': failures}, indent=2)+'\n')
    args.statistics_output.parent.mkdir(parents=True, exist_ok=True)
    args.statistics_output.write_text(json.dumps({
        'dataset': 'mispeech/speechocean762', 'split': 'test',
        'num_utterances': len(rows), 'transcript': str(args.transcript),
        'word_alignment': 'levenshtein', 'metrics': statistics,
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
