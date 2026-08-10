#!/usr/bin/env python3
"""Generate SpeechOcean762 ground-truth word alignments with Charsiu."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from Charsiu import charsiu_forced_aligner
from utils_assessment import get_charsiu_alignment, get_match_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.scores, encoding="utf-8") as handle:
        scores = json.load(handle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligner = charsiu_forced_aligner(aligner="charsiu/en_w2v2_fc_10ms")

    failures = 0
    for key, sample in tqdm(scores.items(), desc="Ground-truth alignment"):
        output = output_dir / f"{key}.pt"
        if output.is_file():
            continue
        wav = Path(args.wav_dir) / f"{key}.wav"
        text = " ".join(word["text"].lower() for word in sample["words"])
        try:
            _, aligned, words, _, _, _ = get_charsiu_alignment(str(wav), text, aligner)
            aligned = np.asarray(aligned)[get_match_index(aligned, words)]
            torch.save(aligned, output)
        except Exception as error:
            failures += 1
            print(f"Alignment failed for {key}: {error}")
    print(f"Alignment failures: {failures}")


if __name__ == "__main__":
    main()
