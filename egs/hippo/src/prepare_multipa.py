#!/usr/bin/env python3
"""Prepare Hippo-compatible MultiPA targets from ordered word annotations."""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "feats_extract"))
from extract_features import configure, g2p_processor
from feature_utils import normalize_transcript


SCORE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def parse_score(value):
    match = SCORE.match(value or "")
    if not match:
        raise ValueError(f"Invalid MultiPA score: {value!r}")
    return float(match.group(1))


def load_transcripts(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = Path(row["audio_id"]).stem
            if key in rows or not isinstance(row.get("transcript"), str):
                raise ValueError(f"Invalid transcript row: {path}:{line_number}")
            words = row.get("words", [])
            if not words or any(
                word.get("start") is None or word.get("end") is None for word in words
            ):
                raise ValueError(
                    f"WhisperX word timestamps are required: {path}:{line_number}"
                )
            rows[key] = row
    return rows


def forced_word_intervals(log_probs, token_ids, group_lengths, duration):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    states = np.zeros(2 * len(token_ids) + 1, dtype=np.int64)
    states[1::2] = token_ids
    score = np.full(len(states), -np.inf)
    score[:2] = log_probs[0, states[:2]]
    backtrack = np.zeros((len(log_probs), len(states)), dtype=np.int8)
    for frame in range(1, len(log_probs)):
        choices = np.stack((score, np.r_[-np.inf, score[:-1]],
                            np.r_[[-np.inf, -np.inf], score[:-2]]))
        allowed = np.zeros(len(states), dtype=bool)
        allowed[3::2] = states[3::2] != states[1:-2:2]
        choices[2, ~allowed] = -np.inf
        selected = choices.argmax(0)
        score = choices[selected, np.arange(len(states))] + log_probs[frame, states]
        backtrack[frame] = selected
    state = len(states) - 1 if score[-1] >= score[-2] else len(states) - 2
    token_frames = [[] for _ in token_ids]
    for frame in range(len(log_probs) - 1, -1, -1):
        if state % 2:
            token_frames[state // 2].append(frame)
        if frame:
            state -= int(backtrack[frame, state])
    if state not in (0, 1) or any(not frames for frames in token_frames):
        raise ValueError("CTC forced alignment did not cover every phone")
    seconds_per_frame = duration / len(log_probs)
    intervals, offset = [], 0
    for length in group_lengths:
        frames = [frame for item in token_frames[offset:offset + length] for frame in item]
        intervals.append((min(frames) * seconds_per_frame,
                          (max(frames) + 1) * seconds_per_frame))
        offset += length
    return intervals


def overlap_word_scores(annotation_rows, intervals):
    values = [[] for _ in intervals]
    for row in annotation_rows:
        try:
            spans = sorted(
                json.loads(row["word-label"]), key=lambda span: float(span["start"])
            )
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            continue
        usable = []
        for span in spans:
            try:
                score = parse_score((span.get("labels") or [None])[0])
            except ValueError:
                continue
            usable.append((span, score))
        for index, (start, end) in enumerate(intervals):
            weighted = []
            for span, score in usable:
                overlap = max(0.0, min(end, float(span["end"]))
                              - max(start, float(span["start"])))
                if overlap:
                    weighted.append((overlap, score))
            if weighted:
                values[index].append(sum(w * score for w, score in weighted)
                                     / sum(w for w, _ in weighted))
    return [float(np.mean(scores)) if scores else None for scores in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--word-alignment", choices=("whisperx", "ctc"), default="whisperx")
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    annotations = defaultdict(list)
    with args.annotations.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            annotations[Path(row["audio"]).stem].append(row)
    transcripts = load_transcripts(args.transcript)
    if set(annotations) != set(transcripts):
        raise ValueError(
            f"MultiPA annotation/transcript IDs differ: "
            f"annotations={len(annotations)}, transcripts={len(transcripts)}"
        )

    configure(args.models)
    convert = g2p_processor(args.models)
    processor = model = vocab = None
    if args.word_alignment == "ctc":
        if args.wav_dir is None:
            parser.error("--wav-dir is required with --word-alignment ctc")
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
        from extract_features import read_audio, validate_ctc
        ctc_path, proc_path, vocab = validate_ctc(args.models)
        processor = AutoFeatureExtractor.from_pretrained(str(proc_path), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(
            str(ctc_path), local_files_only=True
        ).eval().to(args.device)
    scores = {}
    matched_words = total_words = 0
    for key in sorted(transcripts):
        transcript = transcripts[key]
        words, phone_groups = convert(normalize_transcript(transcript["transcript"]))
        if not words:
            raise ValueError(f"Transcript has no pronounceable words: {key}")
        timed_words = transcript["words"]
        if len(timed_words) != len(words):
            raise ValueError(
                f"WhisperX timestamp/G2P word count mismatch for {key}: "
                f"{len(timed_words)} != {len(words)}"
            )
        rows = annotations[key]
        if args.word_alignment == "whisperx":
            intervals = [(float(word["start"]), float(word["end"])) for word in timed_words]
        else:
            audio = read_audio(args.wav_dir / f"{key}.wav")
            phone_ids = [vocab[re.sub(r"\d", "", phone)]
                         for group in phone_groups for phone in group]
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_values.to(args.device)
            with torch.inference_mode():
                log_probs = model(inputs).logits[0].log_softmax(-1).cpu().numpy()
            intervals = forced_word_intervals(
                log_probs, phone_ids, [len(group) for group in phone_groups],
                len(audio) / 16000,
            )
        reference = overlap_word_scores(rows, intervals)
        total_words += len(words)
        matched_words += sum(value is not None for value in reference)
        utterance = {
            name: float(np.mean([parse_score(row[name]) for row in rows]))
            for name in ("accuracy", "fluency", "prosody")
        }
        scores[key] = {
            "accuracy": utterance["accuracy"],
            "completeness": utterance["accuracy"],
            "fluency": utterance["fluency"],
            "prosodic": utterance["prosody"],
            "total": float(np.mean(list(utterance.values()))),
            "words": [
                {
                    "text": word,
                    "accuracy": target,
                    "stress": 5.0,
                    "total": target,
                    "phones": phones,
                    "phones-accuracy": [target] * len(phones),
                }
                for word, phones, target in zip(words, phone_groups, reference)
            ],
        }
        for word in scores[key]["words"]:
            word["multipa-valid"] = word["total"] is not None
            if word["total"] is None:
                word["accuracy"] = word["total"] = utterance["accuracy"]
                word["phones-accuracy"] = [word["total"]] * len(word["phones"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores) + "\n")
    metadata = {
        "dataset": "MultiPA pilot",
        "utterances": len(scores),
        "word_target": "MultiPA word score mapped to Hippo word total",
        "alignment": f"{args.word_alignment} word timestamps; duration-overlap annotation pooling",
        "matched_words": matched_words,
        "total_words": total_words,
        "transcript": str(args.transcript),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(scores)} MultiPA utterances: {args.output}")


if __name__ == "__main__":
    main()
