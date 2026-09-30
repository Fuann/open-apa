#!/usr/bin/env python3
"""Generate a reusable ASR transcript JSONL for SpeechOcean762."""

import argparse
from pathlib import Path

import ctranslate2
from datasets import Audio, load_dataset
from tqdm.auto import tqdm

from asr_transcripts import (
    load_faster_whisper_model,
    load_qwen3_asr_model,
    load_whisperx_model,
    load_transcript_records,
    transcribe_faster_whisper,
    transcribe_qwen3_asr,
    transcribe_whisperx,
    write_transcript_jsonl,
)
from data import decode_audio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mispeech/speechocean762")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend", choices=("faster-whisper", "whisperx", "qwen3-asr"),
        default="faster-whisper",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--redo-audio-ids",
        default="",
        help="Comma-separated wav IDs to transcribe again even when cached",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    redo_audio_ids = {
        value.strip() for value in args.redo_audio_ids.split(",") if value.strip()
    }
    records = (
        load_transcript_records(output, require_timestamps=False)
        if output.exists()
        else {}
    )
    dataset = load_dataset(args.dataset, split=args.split).cast_column(
        "audio", Audio(decode=False)
    )
    missing = []
    for index, example in enumerate(dataset):
        path = example["audio"].get("path")
        audio_id = Path(path).name if path else str(index)
        if audio_id not in records or audio_id in redo_audio_ids:
            missing.append((index, audio_id))
    if not missing:
        print(f"Reusing all {len(records)} transcripts from {output}")
        return

    if args.backend == "whisperx":
        model, asr_config = load_whisperx_model(
            args.model, args.device, args.compute_type, args.beam_size,
            args.batch_size, args.local_files_only,
        )
        transcribe = transcribe_whisperx
    elif args.backend == "faster-whisper":
        model, asr_config = load_faster_whisper_model(
            args.model, args.device, args.compute_type, args.beam_size,
            args.local_files_only,
        )
        asr_config.update({
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_new_tokens": args.max_new_tokens,
        })
        transcribe = transcribe_faster_whisper
    else:
        model, asr_config = load_qwen3_asr_model(
            args.model, args.device, args.compute_type, args.local_files_only,
            max_new_tokens=args.max_new_tokens or 256,
        )
        transcribe = transcribe_qwen3_asr
    for index, audio_id in tqdm(
        missing, desc=f"SpeechOcean762 {args.split} ASR", unit="audio", dynamic_ncols=True
    ):
        waveform, _ = decode_audio(dataset[index]["audio"])
        ctranslate2.set_random_seed(asr_config["random_seed"])
        record = transcribe(model, waveform, asr_config)
        record.update({"audio_id": audio_id, "asr": asr_config})
        records[audio_id] = record
        write_transcript_jsonl(records, output)
    print(f"Saved {len(records)} transcripts to {output}")


if __name__ == "__main__":
    main()
