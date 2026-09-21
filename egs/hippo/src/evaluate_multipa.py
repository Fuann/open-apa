#!/usr/bin/env python3
"""Evaluate Hippo predictions on the MultiPA pilot set."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


UTT = ("accuracy", "completeness", "fluency", "prosodic", "total")


def pcc(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if len(prediction) < 2 or np.std(prediction) == 0 or np.std(target) == 0:
        return None
    return float(pearsonr(prediction, target).statistic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    args = parser.parse_args()

    refs = json.loads(args.scores.read_text())
    with args.manifest.open() as handle:
        ids = [Path(row["audio_path"]).stem for row in csv.DictReader(handle)]
    failures = json.loads((args.feature_dir / "features.json").read_text()).get("failures", {})
    valid_rows = np.array([key not in failures for key in ids])
    pred_dir = args.exp_dir / "preds"
    utterance = np.load(pred_dir / "utt_pred.npy") * 5
    raw = np.load(pred_dir / "word_phone_pred.npy") * 5
    raw_target = np.load(pred_dir / "word_phone_target.npy")

    results = {"utterance": {}}
    lines = [f"MultiPA valid utterances: {int(valid_rows.sum())}/{len(ids)}"]
    for display, model_name in (("accuracy", "accuracy"), ("fluency", "fluency"), ("prosody", "prosodic")):
        prediction = utterance[valid_rows, UTT.index(model_name)]
        target = [refs[key][model_name] for key, valid in zip(ids, valid_rows) if valid]
        value = pcc(prediction, target)
        results["utterance"][display] = {"n": len(target), "pcc": value}
        lines.append(f"utterance {display:<8} PCC={'N/A' if value is None else f'{value:.4f}'}")

    prediction, target = [], []
    for row_index, (row, key, valid) in enumerate(zip(raw, ids, valid_rows)):
        if not valid:
            continue
        word_ids = raw_target[row_index, :, -1].astype(int)
        for word_index in sorted(set(word_ids[word_ids >= 0])):
            if word_index >= len(refs[key]["words"]):
                raise ValueError(f"Word index exceeds MultiPA reference: {key}")
            if not refs[key]["words"][word_index].get("multipa-valid", False):
                continue
            prediction.append(min(10.0, float(row[word_ids == word_index, 2].mean())))
            target.append(float(refs[key]["words"][word_index]["total"]))
    value = pcc(prediction, target)
    results["word_total"] = {"n": len(target), "pcc": value}
    lines.extend(("", "Word-level score: Hippo word total vs. MultiPA word score",
                  f"word total     N={len(target)} PCC={'N/A' if value is None else f'{value:.4f}'}"))
    args.exp_dir.joinpath("multipa_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    args.exp_dir.joinpath("result.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
