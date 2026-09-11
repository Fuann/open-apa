#!/usr/bin/env python3
"""Summarize PCC metrics from five MultiPA evaluation result files."""

import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path


def load_pcc(path):
    metrics = OrderedDict(json.loads(Path(path).read_text(encoding="utf-8")))
    if not metrics:
        raise ValueError(f"No PCC metrics found in {path}")
    metrics = OrderedDict((name, float(value)) for name, value in metrics.items())
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs=5, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = [load_pcc(path) for path in args.results]
    metric_names = list(runs[0])
    for path, metrics in zip(args.results[1:], runs[1:]):
        if list(metrics) != metric_names:
            raise ValueError(f"Metric mismatch in {path}")

    lines = ["PCC across 5 seed models; sample std (ddof=1)", "", "Metric                 Mean       Std"]
    lines.append("--------------------------------------")
    for metric in metric_names:
        values = [run[metric] for run in runs]
        lines.append(
            f"{metric:<20} {statistics.mean(values):>8.4f}  "
            f"{statistics.stdev(values):>8.4f}"
        )
    report = "\n".join(lines) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
