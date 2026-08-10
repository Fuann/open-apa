#!/usr/bin/env python3
"""Extract SpeechOcean762 test WAVs and labels from Hugging Face Parquet."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--data-list", required=True)
    parser.add_argument("--scores", required=True)
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    Path(args.data_list).parent.mkdir(parents=True, exist_ok=True)

    filenames = []
    scores = {}
    parquet = pq.ParquetFile(args.parquet)
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            audio = row.pop("audio")
            filename = Path(audio["path"]).with_suffix(".wav").name
            audio_bytes = audio.get("bytes")
            if audio_bytes is None:
                raise ValueError(f"No embedded audio bytes for {filename}")
            (wav_dir / filename).write_bytes(audio_bytes)
            filenames.append(filename)
            scores[Path(filename).stem] = row

    filenames.sort()
    Path(args.data_list).write_text("\n".join(filenames) + "\n", encoding="utf-8")
    Path(args.scores).write_text(
        json.dumps(scores, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared {len(filenames)} SpeechOcean762 test samples")


if __name__ == "__main__":
    main()

