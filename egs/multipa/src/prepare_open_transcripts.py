#!/usr/bin/env python3
"""Validate or resumably generate fixed SpeechOcean762 open transcripts."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def load_existing(path):
    rows = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audio_id = Path(row["audio_id"]).name
                transcript = row["transcript"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Malformed transcript at {path}:{line_number}") from error
            if audio_id in rows:
                raise ValueError(f"Duplicate transcript for {audio_id} in {path}")
            if not isinstance(transcript, str):
                raise ValueError(f"Transcript for {audio_id} is not a string")
            rows[audio_id] = row
    return rows


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcribe(model, wav_path, model_name):
    segments, _ = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=True,
    )
    segments = list(segments)
    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append(
                {
                    "word": word.word.strip(),
                    "start": round(float(word.start), 6),
                    "end": round(float(word.end), 6),
                    "probability": round(float(word.probability), 6),
                }
            )
    return {
        "transcript": "".join(segment.text for segment in segments).strip(),
        "words": words,
        "fingerprint": fingerprint(wav_path),
        "audio_id": wav_path.name,
        "asr": {
            "backend": "faster-whisper",
            "model": model_name,
            "device": "cuda",
            "compute_type": "float16",
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "vad_filter": False,
            "word_timestamps": True,
            "random_seed": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("medium.en", "large-v3"))
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(0)
    expected = [Path(line.strip()).name for line in args.datalist.read_text().splitlines() if line.strip()]
    if len(expected) != len(set(expected)):
        raise ValueError(f"Duplicate audio IDs in {args.datalist}")
    rows = load_existing(args.output)
    unexpected = sorted(set(rows) - set(expected))
    if unexpected:
        raise ValueError(f"Unexpected audio ID in {args.output}: {unexpected[0]}")
    missing = [audio_id for audio_id in expected if audio_id not in rows]
    if not missing:
        print(f"Using complete fixed transcript: {args.output} ({len(rows)} utterances)")
        return

    from faster_whisper import WhisperModel

    print(f"Generating {len(missing)} missing transcripts with faster-whisper {args.model}")
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for index, audio_id in enumerate(missing, 1):
            wav_path = args.wav_dir / audio_id
            if not wav_path.is_file():
                raise FileNotFoundError(wav_path)
            row = transcribe(model, wav_path, args.model)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            rows[audio_id] = row
            print(f"[{index}/{len(missing)}] {audio_id}", flush=True)

    if set(rows) != set(expected):
        raise RuntimeError(f"Transcript preparation incomplete: {args.output}")


if __name__ == "__main__":
    main()
