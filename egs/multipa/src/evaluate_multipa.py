import argparse
import ast
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr


SCORE_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def parse_score(value):
    match = SCORE_PATTERN.match(value or "")
    if match is None:
        raise ValueError(f"Cannot parse score from {value!r}")
    return float(match.group(1))


def parse_prediction_file(path):
    predictions = {}
    invalid = 0

    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            parts = line.rstrip("\n").split(";")
            if len(parts) < 12:
                raise ValueError(f"Malformed prediction at {path}:{line_number}")

            audio = os.path.basename(parts[0].strip())
            valid = parts[5].split(":", 1)[1].strip() == "T"
            if not valid:
                invalid += 1
                continue

            word_accuracy = [
                float(value)
                for value in parts[8].split(":", 1)[1].strip().split(",")
                if value.strip()
            ]
            alignment = ast.literal_eval(parts[11].split(":", 1)[1].strip())
            if len(word_accuracy) != len(alignment):
                raise ValueError(
                    f"Word score/alignment length mismatch for {audio}: "
                    f"{len(word_accuracy)} scores vs. {len(alignment)} spans"
                )

            predictions[audio] = {
                "accuracy": float(parts[1].split(":", 1)[1]),
                "fluency": float(parts[2].split(":", 1)[1]),
                "prosody": float(parts[3].split(":", 1)[1]),
                "word_accuracy": word_accuracy,
                "alignment": alignment,
            }

    return predictions, invalid


def parse_annotation_file(path):
    annotations = defaultdict(list)
    invalid_word_rows = 0

    with open(path, encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            audio = os.path.basename(row["audio"].strip())
            word_spans = []
            try:
                for span in json.loads(row["word-label"]):
                    labels = span.get("labels", [])
                    if not labels:
                        continue
                    word_spans.append(
                        {
                            "start": float(span["start"]),
                            "end": float(span["end"]),
                            "score": parse_score(labels[0]),
                        }
                    )
            except (TypeError, ValueError, json.JSONDecodeError, KeyError):
                invalid_word_rows += 1
                word_spans = []

            annotations[audio].append(
                {
                    "annotator": row["annotator"],
                    "accuracy": parse_score(row["accuracy"]),
                    "fluency": parse_score(row["fluency"]),
                    "prosody": parse_score(row["prosody"]),
                    "word_spans": word_spans,
                    "line_number": line_number,
                }
            )

    return annotations, invalid_word_rows


def correlation(name, prediction, reference):
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(prediction) < 2:
        print(f"{name:<14} N={len(prediction):>4}  PCC=N/A      SRCC=N/A")
        return

    pcc = pearsonr(prediction, reference).statistic
    srcc = spearmanr(prediction, reference).statistic
    print(f"{name:<14} N={len(prediction):>4}  PCC={pcc:>8.4f}  SRCC={srcc:>8.4f}")


def weighted_overlap_score(start, end, spans):
    weighted_sum = 0.0
    total_overlap = 0.0
    for span in spans:
        overlap = max(0.0, min(end, span["end"]) - max(start, span["start"]))
        if overlap > 0:
            weighted_sum += overlap * span["score"]
            total_overlap += overlap
    if total_overlap == 0:
        return None
    return weighted_sum / total_overlap


def evaluate(predictions, annotations):
    prediction_ids = set(predictions)
    annotation_ids = set(annotations)
    common_ids = sorted(prediction_ids & annotation_ids)

    print(f"Valid predictions: {len(prediction_ids)}")
    print(f"Annotated files:   {len(annotation_ids)}")
    print(f"Matched files:     {len(common_ids)}")
    if prediction_ids - annotation_ids:
        print(f"Predictions without annotations: {len(prediction_ids - annotation_ids)}")
    if annotation_ids - prediction_ids:
        print(f"Annotations without valid predictions: {len(annotation_ids - prediction_ids)}")

    print("\nUtterance-level correlation (human scores averaged across annotators)")
    for metric in ("accuracy", "fluency", "prosody"):
        predicted = []
        reference = []
        for audio in common_ids:
            predicted.append(predictions[audio][metric])
            reference.append(np.mean([row[metric] for row in annotations[audio]]))
        correlation(metric, predicted, reference)

    predicted_words = []
    reference_words = []
    for audio in common_ids:
        prediction = predictions[audio]
        for predicted_score, aligned_word in zip(
            prediction["word_accuracy"], prediction["alignment"]
        ):
            start, end = float(aligned_word[0]), float(aligned_word[1])
            annotator_scores = []
            for annotation in annotations[audio]:
                score = weighted_overlap_score(start, end, annotation["word_spans"])
                if score is not None:
                    annotator_scores.append(score)
            if annotator_scores:
                predicted_words.append(predicted_score)
                reference_words.append(np.mean(annotator_scores))

    print("\nWord-level correlation (timestamp-overlap accuracy labels)")
    correlation("word accuracy", predicted_words, reference_words)
    print(f"Matched predicted words: {len(predicted_words)}")
    print("\nNo metrics are reported for total, word stress, or word total because")
    print("The MultiPA pilot-set annotations do not contain those ground-truth labels.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MultiPA-model open-response predictions against "
            "the MultiPA pilot-set annotations."
        )
    )
    parser.add_argument(
        "--predictions",
        default="Results/model_assessment_multipa_test_mb.txt",
    )
    parser.add_argument("--annotations", default="multipa/annotation.csv")
    args = parser.parse_args()

    predictions, invalid_predictions = parse_prediction_file(args.predictions)
    annotations, invalid_word_rows = parse_annotation_file(args.annotations)
    print(f"Invalid prediction rows skipped: {invalid_predictions}")
    print(f"Annotation rows with unusable word labels: {invalid_word_rows}")
    evaluate(predictions, annotations)


if __name__ == "__main__":
    main()


