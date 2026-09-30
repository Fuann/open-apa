"""PathBench-compatible evaluation on the oral-cancer YouTube set."""

import csv
import gc
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from apa_utils import safe_pearson
from asr_transcripts import (
    load_faster_whisper_model,
    load_qwen3_asr_model,
    load_whisperx_model,
    transcribe_faster_whisper,
    transcribe_qwen3_asr,
    transcribe_whisperx,
    write_transcript_jsonl,
)
from data import (
    DataCollatorWhisperAPA,
    FLUENCY_FEATURE_NAMES,
    PROSODIC_FEATURE_NAMES,
    UTTERANCE_LABELS,
    build_apa_token_ids,
    decode_audio,
    extract_fluency_features,
    extract_prosodic_features,
)
from multipa_normalization import normalize_multipa_text


PATHBENCH_COMMIT = "ab85a8046851d608495dd61b9f90718a03e59184"
PATHBENCH_RAW = (
    "https://raw.githubusercontent.com/karkirowle/pathbench/"
    f"{PATHBENCH_COMMIT}/datasets/youtube"
)
MANIFEST_NAMES = ("text", "utt2spk", "utt2score", "spk2score", "wav.scp")
TRANSCRIPT_CONDITIONS = ("reference_text", "reference_free")


def _load_cached_asr_transcripts(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            row["audio_id"]: row
            for line in handle
            if line.strip()
            for row in (json.loads(line),)
        }


def _prepare_feature(
    audio_path,
    text,
    processor,
    score_token_id,
    num_word_scores,
    handcrafted_features,
    fluency_vad,
    fluency_vad_threshold,
):
    words = text.split()
    if not words:
        raise ValueError(f"Empty YouTube transcript for {audio_path}")
    waveform, sampling_rate = decode_audio({"path": str(audio_path)})
    acoustic_inputs = processor.feature_extractor(
        waveform, sampling_rate=sampling_rate, return_attention_mask=True
    )
    if handcrafted_features:
        fluency_features, analysis = extract_fluency_features(
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
            analysis=analysis,
        )
    else:
        fluency_features = np.zeros(len(FLUENCY_FEATURE_NAMES), dtype=np.float32)
        prosodic_features = np.zeros(len(PROSODIC_FEATURE_NAMES), dtype=np.float32)
    return {
        "input_features": acoustic_inputs.input_features[0],
        "attention_mask": acoustic_inputs.attention_mask[0],
        "fluency_features": fluency_features,
        "prosodic_features": prosodic_features,
        "labels": build_apa_token_ids(processor.tokenizer, words, score_token_id),
        "word_labels": [[-100.0] * num_word_scores for _ in words],
        "word_syllable_counts": [0] * len(words),
        "word_stress_all_labels": [-100.0] * len(words),
        "utterance_labels": [-100.0] * len(UTTERANCE_LABELS),
        "disable_asr_loss": True,
    }


def _read_kaldi_map(path, transform=lambda value: value):
    values = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed manifest line {path}:{line_number}")
            key, value = parts
            if key in values:
                raise ValueError(f"Duplicate utterance ID {key!r} in {path}")
            values[key] = transform(value)
    return values


def ensure_pathbench_youtube_manifests(cache_root):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    for name in MANIFEST_NAMES:
        destination = cache_root / name
        if destination.exists():
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        url = f"{PATHBENCH_RAW}/{name}"
        print(f"Downloading pinned PathBench manifest: {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return cache_root


def load_pathbench_youtube(manifest_root):
    root = Path(manifest_root)
    texts = _read_kaldi_map(root / "text")
    speakers = _read_kaldi_map(root / "utt2spk")
    utterance_scores = _read_kaldi_map(root / "utt2score", float)
    speaker_scores = _read_kaldi_map(root / "spk2score", float)
    if not (texts.keys() == speakers.keys() == utterance_scores.keys()):
        raise ValueError("PathBench YouTube text/utt2spk/utt2score IDs differ")
    unknown = set(speakers.values()) - set(speaker_scores)
    if unknown:
        raise ValueError(f"YouTube utterances have unknown speakers: {sorted(unknown)}")
    return texts, speakers, utterance_scores, speaker_scores


def resolve_youtube_audio(dataset_root, utterance_id):
    root = Path(dataset_root)
    direct_candidates = (
        root / f"{utterance_id}.wav",
        root / "youtube_utterances" / f"{utterance_id}.wav",
        root / "transcription_subset" / f"{utterance_id}.wav",
    )
    for path in direct_candidates:
        if path.exists():
            return path
    matches = list(root.rglob(f"{utterance_id}.wav"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple audio files found for PathBench YouTube ID {utterance_id}: "
            f"{[str(path) for path in matches]}"
        )
    raise FileNotFoundError(
        f"PathBench YouTube audio is missing: {utterance_id}.wav under {root}. "
        "Run scripts/transfer/download_youtube.sh again after updating it to download "
        "youtube_utterances.zip."
    )


def youtube_asr_run_id(
    asr_model, asr_compute_type, asr_beam_size, asr_backend="faster-whisper"
):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", asr_model)
    beam = "" if asr_backend == "qwen3-asr" else f"-beam{asr_beam_size}"
    return f"{asr_backend}-{slug}-{asr_compute_type}{beam}"


def prepare_youtube_asr_texts(
    model, utterance_ids, dataset_root, cache_dir, asr_model, asr_backend,
    asr_batch_size, asr_device, asr_compute_type, asr_beam_size,
    local_files_only,
):
    """Create deterministic reference-free transcripts and cache each result."""
    import ctranslate2
    from tqdm.auto import tqdm

    run_id = youtube_asr_run_id(
        asr_model, asr_compute_type, asr_beam_size, asr_backend
    )
    transcript_path = Path(cache_dir) / "transcripts" / f"{run_id}.jsonl"
    cached = _load_cached_asr_transcripts(transcript_path)
    missing = [uid for uid in utterance_ids if uid not in cached]
    if missing:
        original_device = next(model.parameters()).device
        if original_device.type == "cuda":
            model.to("cpu")
            torch.cuda.empty_cache()
        if asr_backend == "whisperx":
            asr, asr_config = load_whisperx_model(
                asr_model, asr_device, asr_compute_type, asr_beam_size,
                asr_batch_size, local_files_only, word_timestamps=False,
            )
            transcribe = transcribe_whisperx
        elif asr_backend == "faster-whisper":
            asr, asr_config = load_faster_whisper_model(
                asr_model, asr_device, asr_compute_type, asr_beam_size,
                local_files_only,
            )
            transcribe = transcribe_faster_whisper
        else:
            asr, asr_config = load_qwen3_asr_model(
                asr_model, asr_device, asr_compute_type, local_files_only,
            )
            transcribe = transcribe_qwen3_asr
        asr_config["word_timestamps"] = False
        try:
            for index, utterance_id in enumerate(tqdm(
                missing, desc=f"YouTube {asr_model} transcripts", unit="audio"
            ), 1):
                ctranslate2.set_random_seed(asr_config["random_seed"])
                record = transcribe(
                    asr, str(resolve_youtube_audio(dataset_root, utterance_id)),
                    asr_config,
                )
                record.update({"audio_id": utterance_id, "asr": asr_config})
                cached[utterance_id] = record
                if index % 10 == 0:
                    write_transcript_jsonl(cached, transcript_path)
            write_transcript_jsonl(cached, transcript_path)
        finally:
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if original_device.type == "cuda":
                model.to(original_device)
    texts = {}
    for utterance_id in utterance_ids:
        transcript = normalize_multipa_text(
            cached[utterance_id].get("transcript", "")
        )
        texts[utterance_id] = transcript or "nospeech"
    return texts, transcript_path


class YoutubeFeatureDataset(Dataset):
    def __init__(
        self, utterance_ids, texts, dataset_root, processor, score_token_id,
        num_word_scores, handcrafted_features, fluency_vad, fluency_vad_threshold,
    ):
        self.utterance_ids = utterance_ids
        self.texts = texts
        self.dataset_root = dataset_root
        self.processor = processor
        self.score_token_id = score_token_id
        self.num_word_scores = num_word_scores
        self.handcrafted_features = handcrafted_features
        self.fluency_vad = fluency_vad
        self.fluency_vad_threshold = fluency_vad_threshold

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, index):
        utterance_id = self.utterance_ids[index]
        return _prepare_feature(
            resolve_youtube_audio(self.dataset_root, utterance_id),
            self.texts[utterance_id], self.processor, self.score_token_id,
            self.num_word_scores, self.handcrafted_features, self.fluency_vad,
            self.fluency_vad_threshold,
        )


def _speaker_means(predictions, speakers, expected_speakers):
    grouped = defaultdict(list)
    for utterance_id, score in predictions.items():
        grouped[speakers[utterance_id]].append(score)
    missing = set(expected_speakers) - set(grouped)
    if missing:
        raise ValueError(f"No predictions for YouTube speakers: {sorted(missing)}")
    return {
        speaker: float(np.mean(grouped[speaker])) for speaker in expected_speakers
    }


def _write_recording_csv(path, utterance_ids, texts, speakers, targets, predictions):
    score_names = list(predictions)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "utterance_id", "speaker_id", "text", "ground_truth", *score_names,
        ))
        for utterance_id in utterance_ids:
            writer.writerow((
                utterance_id, speakers[utterance_id], texts[utterance_id],
                targets[utterance_id],
                *(predictions[name][utterance_id] for name in score_names),
            ))


def _write_speaker_csv(path, targets, counts, predictions):
    score_names = list(predictions)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("speaker_id", "ground_truth", "utterances", *score_names))
        for speaker in targets:
            writer.writerow((
                speaker, targets[speaker], counts[speaker],
                *(predictions[name][speaker] for name in score_names),
            ))


def _evaluate_youtube_condition(
    model, processor, score_token_id, word_label_names,
    binary_word_label_indices, output_dir, dataset_root="data/youtube/raw",
    manifest_root="data/youtube/pathbench", batch_size=32, num_workers=0,
    max_samples=None, fluency_vad="energy", fluency_vad_threshold=0.5,
    handcrafted_features=False, inference_dtype=None,
    transcript_condition="reference_text", asr_cache_dir="data/youtube",
    asr_model="medium.en", asr_backend="faster-whisper", asr_batch_size=16,
    asr_device="cuda:0", asr_compute_type="float16", asr_beam_size=5,
    local_files_only=False,
):
    """Evaluate all utterance heads using PathBench speaker-level PCC."""
    from tqdm.auto import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_root = ensure_pathbench_youtube_manifests(manifest_root)
    texts, speakers, utterance_targets, speaker_targets = load_pathbench_youtube(
        manifest_root
    )
    utterance_ids = list(texts)
    if max_samples is not None:
        utterance_ids = utterance_ids[:max_samples]
        selected_speakers = {speakers[utterance_id] for utterance_id in utterance_ids}
        speaker_targets = {
            speaker: score for speaker, score in speaker_targets.items()
            if speaker in selected_speakers
        }
    # Resolve before model inference so a wrong corpus fails immediately.
    for utterance_id in utterance_ids:
        resolve_youtube_audio(dataset_root, utterance_id)
    transcript_source = "PathBench YouTube transcript"
    if transcript_condition == "reference_free":
        texts, transcript_path = prepare_youtube_asr_texts(
            model, utterance_ids, dataset_root, asr_cache_dir, asr_model,
            asr_backend, asr_batch_size, asr_device, asr_compute_type,
            asr_beam_size, local_files_only,
        )
        transcript_source = str(transcript_path)
    elif transcript_condition != "reference_text":
        raise ValueError(f"Unknown YouTube transcript condition: {transcript_condition}")

    dataset = YoutubeFeatureDataset(
        utterance_ids, texts, dataset_root, processor, score_token_id,
        len(word_label_names), handcrafted_features, fluency_vad,
        fluency_vad_threshold,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=DataCollatorWhisperAPA(
            processor,
            score_token_id,
            word_representation=getattr(
                model.config, "word_representation", "score-token"
            ),
        ),
    )
    score_names = tuple(f"utterance_{name}" for name in UTTERANCE_LABELS)
    predictions = {name: {} for name in score_names}
    device = next(model.parameters()).device
    model.eval()
    offset = 0
    for batch in tqdm(loader, desc="YouTube inference", unit="batch"):
        current_ids = utterance_ids[offset:offset + len(batch["labels"])]
        offset += len(current_ids)
        batch = {key: value.to(device) for key, value in batch.items()}
        autocast_dtype = {
            "float16": torch.float16, "bfloat16": torch.bfloat16,
        }.get(inference_dtype)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=autocast_dtype or torch.float32,
            enabled=device.type == "cuda" and autocast_dtype is not None,
        ):
            outputs = model(**batch)
        for score_index, name in enumerate(UTTERANCE_LABELS):
            scores = outputs.utterance_logits[:, score_index].float().cpu().tolist()
            predictions[f"utterance_{name}"].update(zip(current_ids, scores))

    speaker_predictions = {
        name: _speaker_means(values, speakers, speaker_targets)
        for name, values in predictions.items()
    }
    speaker_ids = list(speaker_targets)
    speaker_target_array = np.asarray([speaker_targets[s] for s in speaker_ids])
    metrics = {
        f"speaker_{name}_pcc": safe_pearson(
            np.asarray([speaker_predictions[name][s] for s in speaker_ids]),
            speaker_target_array,
        )
        for name in score_names
    }

    counts = defaultdict(int)
    for utterance_id in utterance_ids:
        counts[speakers[utterance_id]] += 1
    _write_recording_csv(
        output_dir / "recording_scores.csv", utterance_ids, texts, speakers,
        utterance_targets, predictions,
    )
    _write_speaker_csv(
        output_dir / "speaker_scores.csv", speaker_targets, counts,
        speaker_predictions,
    )
    result = {
        "condition": transcript_condition.replace("_", "-"),
        "protocol": "PathBench",
        "pathbench_commit": PATHBENCH_COMMIT,
        "transcript": transcript_source,
        "speakers": len(speaker_targets),
        "utterances": len(utterance_ids),
        **metrics,
    }
    if transcript_condition == "reference_free":
        result.update({
            "asr_backend": asr_backend,
            "asr_model": asr_model,
            "asr_compute_type": asr_compute_type,
            "asr_beam_size": asr_beam_size,
        })
    return result


def evaluate_youtube(
    model, processor, score_token_id, word_label_names,
    binary_word_label_indices, output_dir, conditions=("reference_text",), **kwargs,
):
    """Evaluate separate reference-text and reference-free conditions."""
    invalid = set(conditions) - set(TRANSCRIPT_CONDITIONS)
    if invalid:
        raise ValueError(f"Unknown YouTube conditions: {sorted(invalid)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for condition in conditions:
        if condition == "reference_free":
            run_id = youtube_asr_run_id(
                kwargs.get("asr_model", "medium.en"),
                kwargs.get("asr_compute_type", "float16"),
                kwargs.get("asr_beam_size", 5),
                kwargs.get("asr_backend", "faster-whisper"),
            )
            condition_root = output_dir / "reference-free"
            condition_dir = condition_root / run_id
            result_file = condition_root / "results.json"
        else:
            run_id = None
            condition_root = output_dir / "reference-text"
            condition_dir = condition_root
            result_file = condition_root / "results.json"
        condition_dir.mkdir(parents=True, exist_ok=True)
        result = _evaluate_youtube_condition(
            model, processor, score_token_id, word_label_names,
            binary_word_label_indices, condition_dir,
            transcript_condition=condition, **kwargs,
        )
        if run_id is None:
            result_data = result
        else:
            if result_file.exists():
                with result_file.open(encoding="utf-8") as handle:
                    result_data = json.load(handle)
            else:
                result_data = {}
            result_data[run_id] = result
        temporary = result_file.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result_data, handle, indent=2, sort_keys=True)
        temporary.replace(result_file)
        written[condition] = result_data
        print(json.dumps(result_data, indent=2, sort_keys=True))
        print(f"Saved YouTube PathBench evaluation to {result_file}")
    return written
