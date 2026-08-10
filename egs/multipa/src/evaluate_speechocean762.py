#!/usr/bin/env python3
"""Evaluate MultiPA open-response predictions on SpeechOcean762."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
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


def correlation(name, prediction, reference):
    pcc = pearsonr(prediction, reference).statistic
    srcc = spearmanr(prediction, reference).statistic
    print(f"{name:<14} N={len(prediction):>5}  PCC={pcc:>8.4f}  SRCC={srcc:>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--gt-alignments", required=True)
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
    for key in matched:
        prediction = predictions[key]
        words = references[key]["words"]
        if not prediction["valid"]:
            for metric in predicted_words:
                predicted_words[metric].extend([prediction[f"word_{metric}"]] * len(words))
                reference_words[metric].extend(float(word[metric]) for word in words)
            continue

        predicted_text = [str(item[2]).lower() for item in prediction["alignment"]]
        reference_text = [word["text"].lower() for word in words]
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

    print("\nWord-level correlation (ground-truth timestamp overlap)")
    for metric in predicted_words:
        correlation(f"word {metric}", predicted_words[metric], reference_words[metric])


if __name__ == "__main__":
    main()
