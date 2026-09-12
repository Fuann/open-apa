#!/usr/bin/env python3
"""Audit Hippo test rows and evaluate newly generated predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

UTT = ('accuracy', 'completeness', 'fluency', 'prosodic', 'total')
WORD = ('accuracy', 'stress', 'total')
from feats_extract.feature_utils import UTT_FALLBACK, WORD_FALLBACK
from word_evaluation import load_word_evaluation, open_word_arrays


def metrics(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Prediction/target shape or finite-value check failed')
    if not len(pred):
        return dict(n=0, pcc=None, srcc=None, mae=None, rmse=None)
    variable = len(pred) > 1 and np.std(pred) > 0 and np.std(target) > 0
    return dict(n=len(pred), pcc=float(pearsonr(pred, target).statistic) if variable else None,
                srcc=float(spearmanr(pred, target).statistic) if variable else None,
                mae=float(np.abs(pred-target).mean()), rmse=float(np.sqrt(((pred-target)**2).mean())))


def audit(args):
    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    refs = json.loads(args.scores.read_text())
    ids = [Path(row['audio_path']).stem for row in rows]
    if len(ids) != len(set(ids)) or set(ids)-set(refs):
        raise ValueError('Duplicate or non-test IDs in feature manifest')
    suffix = '_' + args.asr_variant if args.response_mode == 'open' and args.asr_variant not in ('large', 'fresh') else ''
    folder = args.feature_dir / ('seq_data_open' if args.response_mode == 'open' else 'seq_data_close')
    targets = np.load(folder / f'te_label_utt{suffix}.npy')
    expected = np.array([[refs[key][m] for m in UTT] for key in ids])
    np.testing.assert_allclose(targets, expected, atol=1e-6)
    for row, key in zip(rows, ids):
        if row['ref_text'].lower().split() != [word['text'].lower() for word in refs[key]['words']]:
            raise ValueError(f'Reference text mismatch: {key}')
    files = [folder / f'te_{name}{suffix}.npy' for name in
             ('feat_ctc_gop_af', 'label_phn', 'label_word', 'label_utt', 'langUse_mbert_wordEmbs', 'word_atten_msk')]
    files += [args.feature_dir / 'seq_data_close' / f'te_utt_ssl_{name}_ali.npy' for name in ('hu','w2v','wlm')]
    inventory = []
    for path in files:
        array = np.load(path, mmap_mode='r')
        if len(array) != len(ids) or not np.isfinite(array).all():
            raise ValueError(f'Invalid cached features: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(block)
        inventory.append(dict(path=str(path.resolve()), shape=list(array.shape), sha256=digest.hexdigest()))
    missing = sorted(set(refs)-set(ids))
    report = dict(dataset='SpeechOcean762 official test', official_count=len(refs), predicted_count=len(ids),
                  missing_ids=missing, response_mode=args.response_mode, asr_variant=args.asr_variant,
                  feature_manifest=str(args.manifest.resolve()), feature_inventory=inventory,
                  protocol='Hippo native sequence-aligned word labels, not MultiPA timestamp overlap')
    if args.asr_variant == 'fresh':
        report['feature_provenance'] = json.loads((args.feature_dir / 'features.json').read_text())
        if report['feature_provenance']['utterance_ids'] != ids:
            raise ValueError('Fresh feature ID order mismatch')
    failures = report.get('feature_provenance', {}).get('failures', {})
    if set(failures) - set(ids):
        raise ValueError('Fallback IDs are absent from manifest')
    if args.asr_variant == 'fresh':
        phone_labels = np.load(folder / 'te_label_phn.npy')
        invalid_rows = ~(phone_labels[:, :, 0] >= 0).any(axis=1)
        if {key for key, invalid in zip(ids, invalid_rows) if invalid} != set(failures):
            raise ValueError('Fallback records do not match rows without phone features')
    report.update(failures=failures, fallback_count=len(failures), evaluated_count=len(ids),
                  predicted_count=len(ids)-len(failures))
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    (args.exp_dir/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.exp_dir/'utterance_ids.json').write_text(json.dumps(ids)+'\n')
    print(f'Validated {len(ids)}/{len(refs)} official test rows; missing count: {len(missing)} (IDs in audit.json)', flush=True)
    return ids, refs, report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--feature-dir',type=Path,required=True)
    p.add_argument('--exp-dir',type=Path,required=True)
    p.add_argument('--response-mode',choices=['open','closed'],default='open')
    p.add_argument('--asr-variant',choices=['med','large','small','fresh'],default='med')
    p.add_argument('--word-evaluation', type=Path)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--seeds',default='0 1 2 3 4')
    args=p.parse_args()
    ids,refs,report=audit(args)
    if args.audit_only:
        return
    word_evaluation = None
    if args.word_evaluation:
        word_evaluation, word_ids = load_word_evaluation(
            args.word_evaluation, args.manifest, args.scores)
        if word_ids != ids:
            raise ValueError('Word evaluation/manifest ID order mismatch')
    evaluation_failures = word_evaluation.get('failures', {}) if word_evaluation else {}
    report['evaluation_failures'] = evaluation_failures
    all_results={}
    for seed in args.seeds.split():
        seed_dir = args.exp_dir / seed
        if not (seed_dir/'preds/utt_pred.npy').exists():
            raise FileNotFoundError(f'Missing predictions for requested seed: {seed}')
        if args.asr_variant == 'fresh':
            provenance = json.loads((seed_dir/'feature_provenance.json').read_text())
            if provenance['features_sha256'] != hashlib.sha256((args.feature_dir/'features.json').read_bytes()).hexdigest():
                raise ValueError(f'Predictions use different features: seed {seed}')
        pred_dir=seed_dir/'preds'
        up=np.load(pred_dir/'utt_pred.npy')*5
        ut=np.load(pred_dir/'utt_target.npy')*5
        np.testing.assert_allclose(ut, [[refs[key][m] for m in UTT] for key in ids],atol=1e-5)
        valid_rows = np.array([key not in report['failures'] for key in ids])
        for i, key in enumerate(ids):
            if key in report['failures']:
                for metric, value in UTT_FALLBACK.items():
                    up[i, UTT.index(metric)] = value
        result={'utterance':{m:metrics(up[valid_rows,i],ut[valid_rows,i]) for i,m in enumerate(UTT)}}
        wp=np.load(pred_dir/'word_pred.npy')*5
        wt=np.load(pred_dir/'word_target.npy')*5
        result['word_native']={m:metrics(wp[:,i],wt[:,i]) for i,m in enumerate(WORD)}
        pp=np.load(pred_dir/'phn_pred.npy').squeeze(-1)
        pt=np.load(pred_dir/'phn_target.npy'); valid=pt>=0
        result['phone_native']=metrics(pp[valid],pt[valid])
        # Separately report the full-test utterance protocol with MultiPA's failure defaults.
        fallback=UTT_FALLBACK
        requested = report.get('feature_provenance', {}).get('requested_ids', list(refs))
        missing_requested = sorted(set(requested) - set(ids))
        fallback_group = 'utterance_full_test_fallback' if len(requested) == len(refs) else 'utterance_requested_fallback'
        main_up = up.copy()
        for i, key in enumerate(ids):
            if key in evaluation_failures:
                for metric, value in UTT_FALLBACK.items():
                    main_up[i, UTT.index(metric)] = value
        result[fallback_group]={m:metrics(
            list(main_up[:,UTT.index(m)])+[fallback[m]]*len(missing_requested),
            [refs[key][m] for key in ids+missing_requested]) for m in fallback}
        failed_words = [word for key in ids if key in report['failures'] for word in refs[key]['words']]
        result['word_with_asr_fallback'] = {m: metrics(
            list(wp[:, i]) + [WORD_FALLBACK[m]] * len(failed_words),
            list(wt[:, i]) + [word[m] for word in failed_words]) for i, m in enumerate(WORD)}
        # Use the same score cap and ASR defaults for both word protocols.
        # Native targets exclude deletions and give insertions/substitutions zero.
        result['word_hippo_sequence'] = {m: metrics(
            list(np.minimum(10., wp[:, i])) + [WORD_FALLBACK[m]] * len(failed_words),
            list(wt[:, i]) + [word[m] for word in failed_words]) for i, m in enumerate(WORD)}
        raw=np.load(pred_dir/'word_phone_pred.npy')*5
        raw_target=np.load(pred_dir/'word_phone_target.npy')
        with (seed_dir/'predictions.jsonl').open('w') as handle:
            for i,key in enumerate(ids):
                if key in report['failures'] or key in evaluation_failures:
                    words = [dict(word_index=j, text=word['text'], scores=dict(WORD_FALLBACK))
                             for j, word in enumerate(refs[key]['words'])]
                    handle.write(json.dumps(dict(id=key, valid=False, fallback=True,
                        reason=report['failures'].get(key, evaluation_failures.get(key)), utterance=dict(UTT_FALLBACK), words=words))+'\n')
                    continue
                wids=raw_target[i,:,-1].astype(int)
                words=[dict(word_index=int(w),scores=dict(zip(WORD,raw[i,wids==w].mean(0).tolist()))) for w in np.unique(wids[wids>=0])]
                handle.write(json.dumps(dict(id=key,valid=True,fallback=False,utterance=dict(zip(UTT,up[i].tolist())),words=words))+'\n')
        if word_evaluation:
            pred_words, target_words = open_word_arrays(
                raw, raw_target, ids, word_evaluation, report['failures'])
            result['word_open_ms'] = {m: metrics(pred_words[:, i], target_words[:, i]) for i, m in enumerate(WORD)}
        (seed_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        all_results[seed_dir.name]=result
    if not all_results:
        raise ValueError('No fresh predictions found')
    summary={}
    lines=[f"Coverage: {len(ids)}/{len(refs)}; missing count: {len(report['missing_ids'])} (IDs in audit.json)",
           'Primary open word protocol: deterministic Levenshtein; matches and substitutions included, insertions and deletions excluded.' if word_evaluation else 'Legacy native word alignment (not the open M+S protocol).']
    lines.append('Feature text normalization: ' + report.get('feature_provenance', {}).get('text_normalization', 'legacy Hippo (not recomputed)'))
    lines.append(f'Evaluation alignment fallback: {len(evaluation_failures)} (details in metrics.json)')
    lines.append(f"ASR fallback: {report['fallback_count']}; model predictions: {report['predicted_count']}")
    groups = (fallback_group, 'word_open_ms') if word_evaluation else (fallback_group, 'word_hippo_sequence')
    for group in groups:
        summary[group]={}
        count = next(iter(next(iter(all_results.values()))[group].values()))['n']
        lines.append('\n'+group+f' PCC (N={count}; mean ± population SD across checkpoints)')
        for metric in next(iter(all_results.values()))[group]:
            values=[r[group][metric]['pcc'] for r in all_results.values()]
            values=[x for x in values if x is not None]
            if not values:
                continue
            summary[group][metric]=dict(pcc_mean=float(np.mean(values)),pcc_std=float(np.std(values)))
            lines.append(f'{metric:14s} {np.mean(values):.4f} ± {np.std(values):.4f}')
    lines.append('\nWord PCC scope: match+substitution. Insertions and deletions are excluded; failed utterances are excluded from word PCC.')
    report.update(seeds=all_results, summary=summary,
                  word_pcc_scope='match+substitution',
                  main_word_protocols=['levenshtein_match_substitution'] if word_evaluation else ['hippo_sequence'])
    (args.exp_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.exp_dir/'result.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
