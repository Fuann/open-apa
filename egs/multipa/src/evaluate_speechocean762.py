#!/usr/bin/env python3
"""Evaluate MultiPA-model predictions on the SpeechOcean762 test set."""

import argparse
import ast
import json
import string
from pathlib import Path

import numpy as np
import torch
from num2words import num2words
from scipy.stats import pearsonr, spearmanr


FALLBACK = {
    "accuracy": 1.0,
    "fluency": 0.0,
    "prosodic": 0.0,
    "total": 0.0,
    "word_accuracy": 0.0,
    "word_stress": 5.0,
    "word_total": 1.0,
}
PCC_RESULTS = {}


def align_words(reference, hypothesis):
    """Return one reference index per hypothesis plus deterministic M/S/I/D counts."""
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    operation = [[None] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        cost[row][0], operation[row][0] = row, "deletion"
    for column in range(1, columns + 1):
        cost[0][column], operation[0][column] = column, "insertion"
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = "match" if reference[row - 1] == hypothesis[column - 1] else "substitution"
            cost[row][column], operation[row][column] = min([
                (cost[row - 1][column - 1] + (diagonal != "match"), diagonal),
                (cost[row - 1][column] + 1, "deletion"),
                (cost[row][column - 1] + 1, "insertion"),
            ], key=lambda item: item[0])
    mapping = [None] * columns
    counts = {name: 0 for name in ("match", "substitution", "insertion", "deletion")}
    row, column = rows, columns
    while row or column:
        current = operation[row][column]
        counts[current] += 1
        if current in ("match", "substitution"):
            mapping[column - 1] = row - 1
            row, column = row - 1, column - 1
        elif current == "deletion":
            row -= 1
        else:
            column -= 1
    return mapping, counts


def normalize_word_units(words):
    punctuation = string.punctuation.replace("'", "")
    cleaned = [str(word).translate(str.maketrans("", "", punctuation)).lower() for word in words]
    compact = "".join(cleaned)
    if compact.isdigit():
        return [" ".join(num2words(int(character)) for character in word) for word in cleaned]
    return [num2words(int(word)) if word.isdigit() else word for word in cleaned]


def transcript_statistics(path, references):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[Path(row["audio_id"]).stem] = row
    totals = {name: 0 for name in ("match", "substitution", "insertion", "deletion")}
    reference_count = hypothesis_count = 0
    for key, item in references.items():
        if key not in records:
            raise ValueError(f"Open transcript missing SpeechOcean762 ID: {key}")
        row = records[key]
        raw_words = row.get("words") or [
            {"word": word} for word in row.get("transcript", "").split()
        ]
        reference = normalize_word_units([word["text"] for word in item["words"]])
        hypothesis = normalize_word_units([
            word.get("word", word.get("text", "")) for word in raw_words
        ])
        _, counts = align_words(reference, hypothesis)
        for name in totals:
            totals[name] += counts[name]
        reference_count += len(reference)
        hypothesis_count += len(hypothesis)
    evaluated = totals["match"] + totals["substitution"]
    return {
        "num_matches": totals["match"], "num_substitutions": totals["substitution"],
        "num_insertions": totals["insertion"], "num_deletions": totals["deletion"],
        "num_reference_words": reference_count, "num_asr_words": hypothesis_count,
        "num_evaluated_words": evaluated,
        "evaluated_coverage": evaluated / reference_count if reference_count else 0.0,
        "wer": (totals["substitution"] + totals["insertion"]
                + totals["deletion"]) / reference_count if reference_count else 0.0,
    }


def parse_list(field):
    value = field.split(":", 1)[1].strip()
    return [min(10.0, float(item)) for item in value.split(",") if item.strip()]


def load_predictions(path):
    predictions = {}
    invalid = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip().split(";")
            if len(parts) < 12:
                raise ValueError(f"Malformed prediction at {path}:{line_number}")
            key = Path(parts[0].strip()).stem
            valid = parts[5].split(":", 1)[1].strip() == "T"
            if not valid:
                invalid += 1
                predictions[key] = {**FALLBACK, "valid": False, "alignment": None}
                continue

            alignment = ast.literal_eval(parts[11].split(":", 1)[1].strip())
            prediction = {
                "valid": True,
                "accuracy": float(parts[1].split(":", 1)[1]),
                "fluency": float(parts[2].split(":", 1)[1]),
                "prosodic": float(parts[3].split(":", 1)[1]),
                "total": float(parts[4].split(":", 1)[1]),
                "word_accuracy": parse_list(parts[8]),
                "word_stress": parse_list(parts[9]),
                "word_total": parse_list(parts[10]),
                "alignment": alignment,
            }
            if any(
                len(prediction[field]) != len(alignment)
                for field in ("word_accuracy", "word_stress", "word_total")
            ):
                raise ValueError(f"Word score/alignment mismatch for {key}")
            predictions[key] = prediction
    return predictions, invalid


def overlap_indices(predicted_alignment, ground_truth_alignment):
    result = []
    for predicted in predicted_alignment:
        predicted_start, predicted_end = float(predicted[0]), float(predicted[1])
        indices = []
        for index, ground_truth in enumerate(ground_truth_alignment):
            gt_start, gt_end = float(ground_truth[0]), float(ground_truth[1])
            if predicted_end <= gt_start:
                break
            if predicted_start >= gt_end:
                continue
            if max(predicted_start, gt_start) <= min(predicted_end, gt_end):
                indices.append(index)
        result.append(indices)
    return result


def pad_merged_words(reference_text, predicted_text, predicted_scores):
    padded = {metric: [] for metric in predicted_scores}
    predicted_index = 0
    for word in reference_text:
        if predicted_index >= len(predicted_text):
            for metric in padded:
                padded[metric].append(predicted_scores[metric][predicted_index - 1])
            break
        score_index = predicted_index if word == predicted_text[predicted_index] else predicted_index - 1
        for metric in padded:
            padded[metric].append(predicted_scores[metric][score_index])
        if word == predicted_text[predicted_index]:
            predicted_index += 1
    return padded


def correlation(name, prediction, reference, display=True):
    variable = (len(prediction) > 1 and np.std(prediction) > 0
                and np.std(reference) > 0)
    pcc = float(pearsonr(prediction, reference).statistic) if variable else None
    srcc = float(spearmanr(prediction, reference).statistic) if variable else None
    PCC_RESULTS[name] = pcc
    if display:
        pcc_text = "N/A" if pcc is None else f"{pcc:.4f}"
        srcc_text = "N/A" if srcc is None else f"{srcc:.4f}"
        print(f"{name:<14} N={len(prediction):>5}  PCC={pcc_text:>8s}  SRCC={srcc_text:>8s}")
    return pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--gt-alignments")
    parser.add_argument("--pcc-json")
    parser.add_argument("--evaluation-mode", choices=("open", "close"), default="open")
    parser.add_argument("--open-transcripts")
    parser.add_argument("--word-evaluation-json")
    args = parser.parse_args()

    with open(args.scores, encoding="utf-8") as handle:
        references = json.load(handle)
    predictions, invalid = load_predictions(args.predictions)
    matched = sorted(set(predictions) & set(references))
    print(f"Predictions: {len(predictions)}")
    print(f"Invalid predictions using fallback scores: {invalid}")
    print(f"Matched files: {len(matched)}")

    print("\nUtterance-level correlation")
    for metric in ("accuracy", "fluency", "prosodic", "total"):
        correlation(
            metric,
            [predictions[key][metric] for key in matched],
            [references[key][metric] for key in matched],
        )

    predicted_words = {metric: [] for metric in ("accuracy", "stress", "total")}
    reference_words = {metric: [] for metric in predicted_words}
    open_words = {
        scope: ({metric: [] for metric in predicted_words},
                {metric: [] for metric in predicted_words})
        for scope in ("M", "S", "M+S")
    }
    edit_counts = {name: 0 for name in ("match", "substitution", "insertion", "deletion")}
    reference_count = hypothesis_count = 0
    for key in matched:
        prediction = predictions[key]
        words = references[key]["words"]
        if not prediction["valid"]:
            if args.evaluation_mode == "open":
                continue
            for metric in predicted_words:
                predicted_words[metric].extend([prediction[f"word_{metric}"]] * len(words))
                reference_words[metric].extend(float(word[metric]) for word in words)
            continue

        predicted_text = normalize_word_units([item[2] for item in prediction["alignment"]])
        reference_text = normalize_word_units([word["text"] for word in words])
        if args.evaluation_mode == "open":
            mapping, counts = align_words(reference_text, predicted_text)
            for name in edit_counts:
                edit_counts[name] += counts[name]
            reference_count += len(reference_text)
            hypothesis_count += len(predicted_text)
            for predicted_index, reference_index in enumerate(mapping):
                if reference_index is None:
                    continue
                scope = ("M" if predicted_text[predicted_index]
                         == reference_text[reference_index] else "S")
                for metric in predicted_words:
                    predicted_score = prediction[f"word_{metric}"][predicted_index]
                    reference_score = float(words[reference_index][metric])
                    predicted_words[metric].append(predicted_score)
                    reference_words[metric].append(reference_score)
                    for selected_scope in (scope, "M+S"):
                        open_words[selected_scope][0][metric].append(predicted_score)
                        open_words[selected_scope][1][metric].append(reference_score)
            continue
        if " ".join(predicted_text) == " ".join(reference_text):
            scores = {
                metric: prediction[f"word_{metric}"] for metric in predicted_words
            }
            if len(predicted_text) != len(reference_text):
                scores = pad_merged_words(reference_text, predicted_text, scores)
            for metric in predicted_words:
                predicted_words[metric].extend(scores[metric])
                reference_words[metric].extend(float(word[metric]) for word in words)
            continue
        else:
            if not args.gt_alignments:
                raise ValueError("--gt-alignments is required for close evaluation")
            alignment_path = Path(args.gt_alignments) / f"{key}.pt"
            if not alignment_path.is_file():
                raise FileNotFoundError(f"Missing ground-truth alignment: {alignment_path}")
            ground_truth_alignment = torch.load(alignment_path, weights_only=False)
            mapping = overlap_indices(prediction["alignment"], ground_truth_alignment)

        defaults = {"accuracy": 0.0, "stress": 5.0, "total": 1.0}
        for predicted_index, reference_indices in enumerate(mapping):
            for metric in predicted_words:
                predicted_words[metric].append(prediction[f"word_{metric}"][predicted_index])
                if reference_indices:
                    reference_words[metric].append(
                        np.mean([float(words[index][metric]) for index in reference_indices])
                    )
                else:
                    reference_words[metric].append(defaults[metric])

    protocol = ("Levenshtein match+substitution" if args.evaluation_mode == "open"
                else "ground-truth timestamp overlap")
    print(f"\nWord-level correlation ({protocol})")
    if args.evaluation_mode == "open":
        print(f"{'metric':14s} {'M':>9s} {'S':>9s} {'M+S':>9s}")
        for metric in predicted_words:
            cells = []
            for scope in ("M", "S", "M+S"):
                prediction, reference = open_words[scope]
                value = correlation(f"word_{metric}_{scope}", prediction[metric],
                                    reference[metric], display=False)
                cells.append("N/A" if value is None else f"{value:.4f}")
            print(f"{metric:14s} {cells[0]:>9s} {cells[1]:>9s} {cells[2]:>9s}")
    else:
        for metric in predicted_words:
            correlation(f"word {metric}", predicted_words[metric], reference_words[metric])

    if args.pcc_json:
        output = Path(args.pcc_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(PCC_RESULTS, indent=2) + "\n", encoding="utf-8")
    if args.evaluation_mode == "open" and args.word_evaluation_json:
        if not args.open_transcripts:
            raise ValueError("--open-transcripts is required with --word-evaluation-json")
        statistics = transcript_statistics(args.open_transcripts, references)
        output = Path(args.word_evaluation_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"dataset": "mispeech/speechocean762",
            "split": "test", "num_utterances": len(references),
            "transcript": args.open_transcripts, "word_alignment": "levenshtein",
            "metrics": statistics}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
