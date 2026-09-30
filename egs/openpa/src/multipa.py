"""Open-response evaluation on the public yuwchen/multipa repository."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, Dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from asr_transcripts import (
    load_faster_whisper_model,
    load_whisperx_model,
    load_transcript_records,
    transcript_result_key,
    transcribe_faster_whisper,
    transcribe_whisperx,
    write_transcript_jsonl,
)
from apa_utils import safe_pearson
from data import (
    DataCollatorWhisperAPA,
    FLUENCY_FEATURE_NAMES,
    PROSODIC_FEATURE_NAMES,
    build_apa_token_ids,
    decode_audio,
    extract_fluency_features,
    extract_prosodic_features,
)
from multipa_normalization import normalize_multipa_word_units


_SCORE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def _score(label):
    label = str(label or "").strip()
    match = _SCORE_RE.match(label)
    if not match:
        raise ValueError(f"Cannot parse MultiPA word score from {label!r}")
    return float(match.group(1))


def load_annotations(csv_path):
    """Return word spans and averaged utterance labels grouped by audio."""
    grouped = defaultdict(list)
    utterance_ratings = defaultdict(lambda: defaultdict(list))
    invalid_word_rows = 0
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            spans = []
            try:
                for item in json.loads(row.get("word-label") or ""):
                    labels = item.get("labels", [])
                    if labels:
                        spans.append({
                            "start": float(item["start"]),
                            "end": float(item["end"]),
                            "score": _score(labels[0]),
                        })
            except (TypeError, ValueError, json.JSONDecodeError, KeyError):
                # Match open-apa: one unusable label invalidates this
                # annotator's complete set of word spans for the utterance.
                invalid_word_rows += 1
                spans = []
            audio = Path(row["audio"].strip()).name
            grouped[audio].append(spans)
            utterance_ratings[audio]["accuracy"].append(_score(row["accuracy"]))
            utterance_ratings[audio]["fluency"].append(_score(row["fluency"]))
            utterance_ratings[audio]["prosodic"].append(_score(row["prosody"]))
    print(f"MultiPA annotation rows with unusable word labels: {invalid_word_rows}")
    utterance_labels = {
        audio: {
            name: float(np.mean(scores)) for name, scores in ratings.items()
        }
        for audio, ratings in utterance_ratings.items()
    }
    return dict(grouped), utterance_labels


def _weighted_overlap_score(start, end, spans):
    weighted_sum = 0.0
    total_overlap = 0.0
    for span in spans:
        overlap = max(0.0, min(end, span["end"]) - max(start, span["start"]))
        if overlap > 0:
            weighted_sum += overlap * span["score"]
            total_overlap += overlap
    return weighted_sum / total_overlap if total_overlap else None


def align_word_scores(word_chunks, annotators):
    """Match open-apa's overlap-weighted MultiPA word-accuracy targets."""
    scores = []
    for chunk in word_chunks:
        start, end = chunk["timestamp"]
        if start is None or end is None or end <= start:
            scores.append(-100.0)
            continue
        per_annotator = []
        for spans in annotators:
            score = _weighted_overlap_score(start, end, spans)
            if score is not None:
                per_annotator.append(score)
        scores.append(float(np.mean(per_annotator)) if per_annotator else -100.0)
    return scores


def _update_multipa_results(path, transcript, metrics):
    """Atomically save metrics keyed by the transcript used for inference."""
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
    else:
        existing = {}
    if "runs" in existing:
        existing = {
            (
                run.get("evaluation_config", {}).get("transcript", {}).get("source")
                or run.get("evaluation_config", {}).get("transcript", {}).get("id")
                or run_id
            ): run.get("metrics", {})
            for run_id, run in existing["runs"].items()
        }
    elif existing and all(key.startswith("multipa_") for key in existing):
        existing = {"legacy": existing}
    existing = {
        transcript_result_key(key): value for key, value in existing.items()
    }
    existing[transcript_result_key(transcript)] = metrics
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def generate_asr_transcripts(
    repo_dir,
    audio_names,
    output_path,
    asr_model,
    asr_device,
    asr_compute_type,
    asr_beam_size,
    asr_backend,
    asr_batch_size,
    local_files_only,
    existing_records=None,
):
    """Generate and incrementally persist HF-friendly JSONL rows."""
    import ctranslate2
    from tqdm.auto import tqdm

    if asr_backend == "qwen3-asr":
        raise ValueError(
            "MultiPA word-level evaluation requires timestamped ASR words; "
            "Qwen3-ASR text-only decoding is not supported. Use WhisperX, "
            "faster-whisper, or provide a timestamped transcript source."
        )

    if asr_backend == "whisperx":
        model, asr_config = load_whisperx_model(
            asr_model, asr_device, asr_compute_type, asr_beam_size,
            asr_batch_size, local_files_only, word_timestamps=True,
        )
        transcribe = transcribe_whisperx
    else:
        model, asr_config = load_faster_whisper_model(
            asr_model, asr_device, asr_compute_type, asr_beam_size,
            local_files_only,
        )
        transcribe = transcribe_faster_whisper
    records = dict(existing_records or {})
    progress = tqdm(
        audio_names,
        desc="MultiPA ASR transcripts",
    )
    for audio_name in progress:
        audio_path = repo_dir / "wav" / audio_name
        ctranslate2.set_random_seed(asr_config["random_seed"])
        record = transcribe(model, str(audio_path), asr_config)
        record.update({"audio_id": audio_name, "asr": asr_config})
        records[audio_name] = record
        write_transcript_jsonl(records, output_path)
    del model
    return records


def prepare_multipa_features(
    processor,
    score_token_id,
    word_label_names,
    cache_dir,
    word_output="accuracy",
    text_normalization="none",
    transcript_source=None,
    transcript_split="train",
    transcript_output=None,
    asr_model="medium.en",
    asr_device="cuda:0",
    asr_compute_type="float16",
    asr_beam_size=5,
    asr_backend="faster-whisper",
    asr_batch_size=16,
    max_samples=None,
    local_files_only=False,
    normalize_word_labels=False,
    normalize_utterance_labels=False,
    fluency_vad="energy",
    fluency_vad_threshold=0.5,
    handcrafted_features=False,
):
    if word_output not in {"accuracy", "total"}:
        raise ValueError(
            "MultiPA word output must be 'accuracy' or 'total'; "
            f"got {word_output!r}"
        )
    if word_output not in word_label_names:
        raise ValueError(
            f"MultiPA word-{word_output} evaluation requires a checkpoint "
            f"trained with {word_output} in --word-labels"
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(snapshot_download(
        repo_id="yuwchen/multipa",
        repo_type="model",
        allow_patterns=["annotation.csv", "wav/*.wav"],
        local_files_only=local_files_only,
    ))
    annotations, utterance_targets = load_annotations(repo_dir / "annotation.csv")
    audio_names = sorted(annotations)
    if max_samples is not None:
        audio_names = audio_names[:max_samples]

    if transcript_source:
        transcript_records = load_transcript_records(
            transcript_source, split=transcript_split
        )
        print(f"Loaded fixed transcripts from {transcript_source}", flush=True)
        transcript_reference = f"{transcript_source}[split={transcript_split}]"
    else:
        asr_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", asr_model)
        transcript_output = Path(
            transcript_output
            or cache_dir / "transcripts"
            / (
                f"{asr_backend}-{asr_slug}-{asr_compute_type}"
                f"-beam{asr_beam_size}.jsonl"
            )
        )
        if transcript_output.exists():
            transcript_records = load_transcript_records(transcript_output)
            print(f"Loaded cached transcripts from {transcript_output}", flush=True)
        else:
            transcript_records = {}
        missing_generated = [
            name for name in audio_names if name not in transcript_records
        ]
        if missing_generated:
            print(
                f"Generating {len(missing_generated)} MultiPA transcripts with "
                f"{asr_backend} {asr_model}",
                flush=True,
            )
            transcript_records = generate_asr_transcripts(
                repo_dir,
                missing_generated,
                transcript_output,
                asr_model,
                asr_device,
                asr_compute_type,
                asr_beam_size,
                asr_backend,
                asr_batch_size,
                local_files_only,
                existing_records=transcript_records,
            )
        transcript_reference = str(transcript_output)
    missing_transcripts = [
        name for name in audio_names if name not in transcript_records
    ]
    if missing_transcripts:
        raise ValueError(
            f"Transcript source is missing {len(missing_transcripts)} audio files; "
            f"first missing: {missing_transcripts[0]}"
        )

    paths = [str(repo_dir / "wav" / name) for name in audio_names]
    audio_dataset = Dataset.from_dict({"audio": paths}).cast_column(
        "audio", Audio(decode=False)
    )

    word_output_index = word_label_names.index(word_output)
    features = []
    matched_words = 0
    for name, row in zip(audio_names, audio_dataset):
        waveform, sampling_rate = decode_audio(row["audio"])
        record = transcript_records[name]
        chunks = [
            {
                "text": word["word"],
                "timestamp": [word["start"], word["end"]],
            }
            for word in record["words"]
        ]
        chunks = [
            chunk for chunk in chunks
            if chunk.get("text", "").strip() and chunk.get("timestamp")
        ]
        words = [chunk["text"].strip() for chunk in chunks]
        if text_normalization == "multipa":
            words = normalize_multipa_word_units(words)
        elif text_normalization != "none":
            raise ValueError(f"Unknown text normalization: {text_normalization!r}")
        normalized = [
            (chunk, word) for chunk, word in zip(chunks, words) if word
        ]
        chunks = [item[0] for item in normalized]
        words = [item[1] for item in normalized]
        multipa_fallback = not words
        if not words:
            # Match MultiPA's failed-alignment policy.  A whole-utterance span
            # keeps a human word target available for the fallback prediction.
            words = ["nospeech"]
            chunks = [{
                "text": "nospeech",
                "timestamp": [0.0, len(waveform) / sampling_rate],
            }]
            print(f"Using MultiPA fallback scores for empty transcript: {name}", flush=True)
        targets = align_word_scores(chunks, annotations[name])
        labels = [[-100.0] * len(word_label_names) for _ in words]
        for label, target in zip(labels, targets):
            label[word_output_index] = target
            matched_words += target != -100.0
        acoustic_inputs = processor.feature_extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_attention_mask=True,
        )
        if handcrafted_features:
            fluency_features, acoustic_analysis = extract_fluency_features(
                waveform,
                sampling_rate,
                vad_method=fluency_vad,
                vad_threshold=fluency_vad_threshold,
                return_analysis=True,
            )
            prosodic_features = extract_prosodic_features(
                waveform,
                sampling_rate,
                vad_method=fluency_vad,
                vad_threshold=fluency_vad_threshold,
                analysis=acoustic_analysis,
            )
        else:
            fluency_features = np.zeros(
                len(FLUENCY_FEATURE_NAMES), dtype=np.float32
            )
            prosodic_features = np.zeros(
                len(PROSODIC_FEATURE_NAMES), dtype=np.float32
            )
        features.append({
            "input_features": acoustic_inputs.input_features[0],
            "attention_mask": acoustic_inputs.attention_mask[0],
            "fluency_features": fluency_features,
            "prosodic_features": prosodic_features,
            "labels": build_apa_token_ids(processor.tokenizer, words, score_token_id),
            "word_labels": labels,
            "word_syllable_counts": [0] * len(words),
            "word_stress_all_labels": [-100.0] * len(words),
            "utterance_labels": [
                utterance_targets[name]["accuracy"],
                utterance_targets[name]["fluency"],
                utterance_targets[name]["prosodic"],
                -100.0,
            ],
            "multipa_fallback": multipa_fallback,
        })
    if not features or matched_words < 2:
        raise ValueError("MultiPA evaluation produced fewer than two matched words")
    return features, word_output_index, transcript_reference


def evaluate_multipa(
    model,
    processor,
    score_token_id,
    word_label_names,
    output_dir,
    cache_dir=None,
    word_output="accuracy",
    text_normalization="none",
    batch_size=8,
    num_workers=0,
    transcript_source=None,
    transcript_split="train",
    transcript_output=None,
    asr_model="medium.en",
    asr_device="cuda:0",
    asr_compute_type="float16",
    asr_beam_size=5,
    asr_backend="faster-whisper",
    asr_batch_size=16,
    max_samples=None,
    local_files_only=False,
    normalize_word_labels=False,
    normalize_utterance_labels=False,
    fluency_vad="energy",
    fluency_vad_threshold=0.5,
    handcrafted_features=False,
):
    """Evaluate the selected word output and available utterance PCCs on MultiPA."""
    from tqdm.auto import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, word_output_index, transcript_reference = prepare_multipa_features(
        processor, score_token_id, tuple(word_label_names), cache_dir or output_dir,
        word_output=word_output,
        text_normalization=text_normalization,
        transcript_source=transcript_source,
        transcript_split=transcript_split,
        transcript_output=transcript_output,
        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_beam_size=asr_beam_size,
        asr_backend=asr_backend,
        asr_batch_size=asr_batch_size,
        max_samples=max_samples,
        local_files_only=local_files_only,
        normalize_word_labels=normalize_word_labels,
        normalize_utterance_labels=normalize_utterance_labels,
        fluency_vad=fluency_vad,
        fluency_vad_threshold=fluency_vad_threshold,
        handcrafted_features=handcrafted_features,
    )
    loader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DataCollatorWhisperAPA(
            processor,
            score_token_id,
            word_representation=getattr(
                model.config, "word_representation", "score-token"
            ),
        ),
    )
    device = next(model.parameters()).device
    predictions = []
    targets = []
    utterance_predictions = defaultdict(list)
    utterance_targets = defaultdict(list)
    model.eval()
    for batch in tqdm(
        loader,
        desc="MultiPA model inference",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        fallback = batch["multipa_fallback"]
        if fallback.any():
            word_fallback = {"accuracy": 0.0, "stress": 5.0, "total": 1.0}
            value = word_fallback[word_output]
            if normalize_word_labels:
                value /= 10.0
            outputs.word_logits[fallback, :, word_output_index] = value
            utterance_fallback = outputs.utterance_logits.new_tensor(
                [1.0, 0.0, 0.0, 0.0]
            )
            if normalize_utterance_labels:
                utterance_fallback /= 10.0
            outputs.utterance_logits[fallback] = utterance_fallback
        valid = (
            batch["word_label_mask"]
            & batch["word_labels"][:, :, word_output_index].ne(-100)
        )
        predictions.extend(
            outputs.word_logits[:, :, word_output_index][valid].float().cpu().tolist()
        )
        targets.extend(
            batch["word_labels"][:, :, word_output_index][valid].float().cpu().tolist()
        )
        for index, name in enumerate(("accuracy", "fluency", "prosodic")):
            valid_utterance = batch["utterance_labels"][:, index].ne(-100)
            utterance_predictions[name].extend(
                outputs.utterance_logits[:, index][valid_utterance].float().cpu().tolist()
            )
            utterance_targets[name].extend(
                batch["utterance_labels"][:, index][valid_utterance].float().cpu().tolist()
            )
    word_metric_key = f"multipa_word_{word_output}_pcc"
    result = {
        word_metric_key: safe_pearson(
            np.asarray(predictions), np.asarray(targets)
        ),
        "multipa_matched_words": len(targets),
        "multipa_utterances": len(features),
        "multipa_fallback_utterances": sum(
            feature.get("multipa_fallback", False) for feature in features
        ),
    }
    pcc_keys = [word_metric_key]
    for name in ("accuracy", "fluency", "prosodic"):
        key = f"multipa_utterance_{name}_pcc"
        result[key] = safe_pearson(
            np.asarray(utterance_predictions[name]), np.asarray(utterance_targets[name])
        )
        pcc_keys.append(key)
    result["multipa_apa_macro_pcc"] = float(np.mean([result[key] for key in pcc_keys]))
    result_file = output_dir / "results.json"
    _update_multipa_results(result_file, transcript_reference, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved MultiPA metrics for {transcript_reference!r} in {result_file}")
    return result
