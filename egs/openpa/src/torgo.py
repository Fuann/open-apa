"""PathBench-compatible speaker-level evaluation on TORGO."""

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
    load_whisperx_model,
    transcribe_faster_whisper,
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
    f"{PATHBENCH_COMMIT}/datasets/torgo/pathological"
)
TASKS = ("word", "utterances")
BALANCINGS = ("balanced", "unbalanced")
PATHBENCH_PROTOCOLS = {"balanced": "mc", "unbalanced": "ex"}
TRANSCRIPT_CONDITIONS = ("reference_text", "reference_free")
FDA_SCORES = {
    "word": {
        "F01": 2.0, "F03": 5.0, "F04": 5.0, "M01": 2.5,
        "M02": 2.5, "M03": 5.0, "M04": 1.5, "M05": 5.0,
    },
    "utterances": {
        "F01": 1.5, "F03": 5.0, "F04": 5.0, "M01": 1.5,
        "M02": 1.5, "M03": 5.0, "M04": 1.5, "M05": 3.0,
    },
}


def _manifest_path(root, task, balancing, name):
    return Path(root) / task / balancing / name


def ensure_pathbench_manifests(cache_root):
    """Download the pinned PathBench TORGO text/utt2spk files if absent."""
    cache_root = Path(cache_root)
    for task in TASKS:
        for balancing in BALANCINGS:
            for name in ("text", "utt2spk"):
                destination = _manifest_path(cache_root, task, balancing, name)
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                url = f"{PATHBENCH_RAW}/{task}/{balancing}/{name}"
                temporary = destination.with_suffix(".tmp")
                print(f"Downloading pinned PathBench manifest: {url}", flush=True)
                try:
                    urllib.request.urlretrieve(url, temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
    return cache_root


def _read_kaldi_map(path):
    values = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed manifest line {path}:{line_number}")
            key, value = parts
            if key in values:
                raise ValueError(f"Duplicate utterance ID {key!r} in {path}")
            values[key] = value
    return values


def load_pathbench_subset(manifest_root, task, balancing):
    speakers = _read_kaldi_map(
        _manifest_path(manifest_root, task, balancing, "utt2spk")
    )
    texts = _read_kaldi_map(_manifest_path(manifest_root, task, balancing, "text"))
    if speakers.keys() != texts.keys():
        raise ValueError(f"PathBench {task}/{balancing} text and utt2spk IDs differ")
    return {utt_id: (speakers[utt_id], texts[utt_id]) for utt_id in speakers}


def select_pathological_speakers(records, task):
    """Keep only speakers with PathBench FDA targets for this task."""
    targets = FDA_SCORES[task]
    return {
        utterance_id: value
        for utterance_id, value in records.items()
        if value[0] in targets
    }


def resolve_torgo_audio(dataset_root, utterance_id):
    """Resolve ``F01_Session1_0001`` to TORGO's array-microphone WAV."""
    parts = utterance_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected TORGO utterance ID: {utterance_id!r}")
    speaker, recording = parts[0], parts[-1]
    session = "_".join(parts[1:-1])
    path = Path(dataset_root) / speaker / session / "wav_arrayMic" / f"{recording}.wav"
    if not path.exists():
        raise FileNotFoundError(f"TORGO audio required by PathBench is missing: {path}")
    return path


def _load_cached_asr_transcripts(path):
    path = Path(path)
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["audio_id"]] = record
    return records


def torgo_asr_run_id(
    asr_model, asr_compute_type, asr_beam_size, asr_backend="faster-whisper"
):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", asr_model)
    return f"{asr_backend}-{slug}-{asr_compute_type}-beam{asr_beam_size}"


def prepare_torgo_asr_texts(
    model,
    records,
    dataset_root,
    cache_dir,
    asr_model,
    asr_backend,
    asr_batch_size,
    asr_device,
    asr_compute_type,
    asr_beam_size,
    local_files_only,
):
    """Generate deterministic ASR text while avoiding two GPU models at once."""
    import ctranslate2
    from tqdm.auto import tqdm

    run_id = torgo_asr_run_id(
        asr_model, asr_compute_type, asr_beam_size, asr_backend
    )
    transcript_path = Path(cache_dir) / "transcripts" / f"{run_id}.jsonl"
    cached = _load_cached_asr_transcripts(transcript_path)
    missing = [utterance_id for utterance_id in records if utterance_id not in cached]
    if missing:
        original_device = next(model.parameters()).device
        if original_device.type == "cuda":
            model.to("cpu")
            torch.cuda.empty_cache()
        if asr_backend == "whisperx":
            asr, asr_config = load_whisperx_model(
                asr_model,
                asr_device,
                asr_compute_type,
                asr_beam_size,
                asr_batch_size,
                local_files_only,
                word_timestamps=False,
            )
            transcribe = transcribe_whisperx
        else:
            asr, asr_config = load_faster_whisper_model(
                asr_model,
                asr_device,
                asr_compute_type,
                asr_beam_size,
                local_files_only,
            )
            transcribe = transcribe_faster_whisper
        # TORGO only consumes the utterance transcript. Disabling word-level
        # alignment avoids an unnecessary and crash-prone find_alignment pass.
        asr_config["word_timestamps"] = False
        try:
            for index, utterance_id in enumerate(tqdm(
                missing, desc=f"TORGO {asr_model} transcripts", unit="audio"
            ), 1):
                ctranslate2.set_random_seed(asr_config["random_seed"])
                record = transcribe(
                    asr,
                    str(resolve_torgo_audio(dataset_root, utterance_id)),
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
    for utterance_id in records:
        transcript = normalize_multipa_text(
            cached[utterance_id].get("transcript", "")
        )
        texts[utterance_id] = transcript or "nospeech"
    return texts, transcript_path


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
        raise ValueError(f"Empty TORGO transcript for {audio_path}")
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


class TorgoFeatureDataset(Dataset):
    """Lazily decode TORGO audio so the full corpus is not held in RAM."""

    def __init__(
        self,
        utterance_ids,
        records,
        dataset_root,
        processor,
        score_token_id,
        num_word_scores,
        handcrafted_features,
        fluency_vad,
        fluency_vad_threshold,
    ):
        self.utterance_ids = utterance_ids
        self.records = records
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
            resolve_torgo_audio(self.dataset_root, utterance_id),
            self.records[utterance_id][1],
            self.processor,
            self.score_token_id,
            self.num_word_scores,
            self.handcrafted_features,
            self.fluency_vad,
            self.fluency_vad_threshold,
        )


def _aggregate_predictions(predictions, subset, speakers):
    by_speaker = defaultdict(list)
    for utterance_id in subset:
        if utterance_id not in predictions:
            raise ValueError(f"Missing TORGO prediction for {utterance_id}")
        speaker = subset[utterance_id][0]
        by_speaker[speaker].append(predictions[utterance_id])
    speakers = sorted(speakers)
    missing = [speaker for speaker in speakers if not by_speaker[speaker]]
    if missing:
        raise ValueError(f"TORGO subset has no predictions for speakers: {missing}")
    speaker_predictions = {
        speaker: float(np.mean(by_speaker[speaker])) for speaker in speakers
    }
    return speaker_predictions


def _aggregate_and_score(predictions, subset, ground_truth):
    speakers = sorted(ground_truth)
    speaker_predictions = _aggregate_predictions(
        predictions, subset, speakers
    )
    predicted = np.asarray([speaker_predictions[speaker] for speaker in speakers])
    targets = np.asarray([ground_truth[speaker] for speaker in speakers])
    return speaker_predictions, safe_pearson(predicted, targets)


def _limit_per_speaker(records, maximum):
    counts = defaultdict(int)
    selected = {}
    for utterance_id, value in records.items():
        speaker = value[0]
        if counts[speaker] < maximum:
            selected[utterance_id] = value
            counts[speaker] += 1
    return selected


def _write_speaker_csv(path, predictions_by_score, ground_truth, counts):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        score_names = list(predictions_by_score)
        writer.writerow((
            "speaker_id", "ground_truth", "recordings", *score_names
        ))
        for speaker in sorted(ground_truth):
            writer.writerow((
                speaker,
                ground_truth[speaker],
                counts[speaker],
                *(predictions_by_score[name][speaker] for name in score_names),
            ))


def _write_recording_csv(path, records, predictions_by_score, memberships):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        score_names = list(predictions_by_score)
        writer.writerow((
            "utterance_id", "speaker_id", "text", "subsets", *score_names
        ))
        for utterance_id, (speaker, text) in records.items():
            writer.writerow((
                utterance_id,
                speaker,
                text,
                ",".join(memberships[utterance_id]),
                *(predictions_by_score[name][utterance_id] for name in score_names),
            ))


def _evaluate_torgo_condition(
    model,
    processor,
    score_token_id,
    word_label_names,
    binary_word_label_indices,
    output_dir,
    transcript_condition,
    dataset_root="data/torgo/raw",
    manifest_root="data/torgo/pathbench",
    balancings=BALANCINGS,
    word_output="accuracy",
    utterance_output="accuracy",
    batch_size=32,
    num_workers=0,
    max_samples=None,
    fluency_vad="energy",
    fluency_vad_threshold=0.5,
    handcrafted_features=False,
    inference_dtype=None,
    asr_cache_dir="data/torgo",
    asr_model="medium.en",
    asr_backend="faster-whisper",
    asr_batch_size=16,
    asr_device="cuda:0",
    asr_compute_type="float16",
    asr_beam_size=5,
    local_files_only=False,
):
    """Run OpenPA once per task and reproduce PathBench speaker-level PCC."""
    from tqdm.auto import tqdm

    if word_output not in word_label_names:
        raise ValueError(
            f"TORGO word evaluation requires {word_output!r} in word labels; "
            f"checkpoint has {list(word_label_names)!r}"
        )
    if utterance_output not in UTTERANCE_LABELS:
        raise ValueError(f"Unknown utterance output: {utterance_output!r}")
    invalid = set(balancings) - set(BALANCINGS)
    if invalid:
        raise ValueError(f"Unknown TORGO balancing modes: {sorted(invalid)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_root = ensure_pathbench_manifests(manifest_root)
    selected = {
        (task, balancing): select_pathological_speakers(
            load_pathbench_subset(manifest_root, task, balancing), task
        )
        for task in TASKS for balancing in balancings
    }
    if max_samples is not None:
        if set(balancings) != {"unbalanced"}:
            raise ValueError("--torgo-max-samples requires unbalanced-only evaluation")
        selected = {
            key: _limit_per_speaker(subset, max_samples)
            for key, subset in selected.items()
        }
    transcript_source = "TORGO prompt"
    if transcript_condition == "reference_free":
        all_records = {}
        for subset in selected.values():
            all_records.update(subset)
        asr_texts, transcript_path = prepare_torgo_asr_texts(
            model,
            all_records,
            dataset_root,
            asr_cache_dir,
            asr_model,
            asr_backend,
            asr_batch_size,
            asr_device,
            asr_compute_type,
            asr_beam_size,
            local_files_only,
        )
        selected = {
            key: {
                utterance_id: (speaker, asr_texts[utterance_id])
                for utterance_id, (speaker, _) in subset.items()
            }
            for key, subset in selected.items()
        }
        transcript_source = str(transcript_path)
    elif transcript_condition != "reference_text":
        raise ValueError(f"Unknown TORGO transcript condition: {transcript_condition}")
    binary_word_label_indices = set(binary_word_label_indices)
    metrics = {"word": {}, "sentence": {}}
    statistics = {
        "num_speakers": len(FDA_SCORES["word"]),
        "transcript_source": transcript_source,
        "word": {},
        "sentence": {},
    }
    if transcript_condition == "reference_free":
        statistics["asr"] = {
            "backend": asr_backend,
            "model": asr_model,
            "compute_type": asr_compute_type,
            "beam_size": asr_beam_size,
        }

    device = next(model.parameters()).device
    model.eval()
    for task in TASKS:
        union = {}
        for balancing in balancings:
            union.update(selected[(task, balancing)])
        utterance_ids = list(union)
        features = TorgoFeatureDataset(
            utterance_ids,
            union,
            dataset_root,
            processor,
            score_token_id,
            len(word_label_names),
            handcrafted_features,
            fluency_vad,
            fluency_vad_threshold,
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
        word_score_names = tuple(f"word_{name}" for name in word_label_names)
        utterance_score_names = tuple(f"utterance_{name}" for name in UTTERANCE_LABELS)
        score_names = word_score_names + utterance_score_names
        predictions = {name: {} for name in score_names}
        offset = 0
        for batch in tqdm(loader, desc=f"TORGO {task} inference", unit="batch"):
            current_ids = utterance_ids[offset : offset + len(batch["labels"])]
            offset += len(current_ids)
            batch = {key: value.to(device) for key, value in batch.items()}
            autocast_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(inference_dtype)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype or torch.float32,
                enabled=device.type == "cuda" and autocast_dtype is not None,
            ):
                outputs = model(**batch)
            for score_index, name in enumerate(word_label_names):
                score_name = f"word_{name}"
                values = outputs.word_logits[:, :, score_index]
                if score_index in binary_word_label_indices:
                    values = torch.sigmoid(values)
                scores = [
                    float(row[mask].float().mean().cpu())
                    for row, mask in zip(values, batch["word_label_mask"])
                ]
                predictions[score_name].update(zip(current_ids, scores))
            for score_index, name in enumerate(UTTERANCE_LABELS):
                score_name = f"utterance_{name}"
                scores = outputs.utterance_logits[:, score_index].float().cpu().tolist()
                predictions[score_name].update(zip(current_ids, scores))

        memberships = defaultdict(list)
        for balancing in balancings:
            for utterance_id in selected[(task, balancing)]:
                if utterance_id in union:
                    memberships[utterance_id].append(balancing)
        recording_label = "word" if task == "word" else "utterance"
        _write_recording_csv(
            output_dir / f"{recording_label}_recording_scores.csv",
            union,
            predictions,
            memberships,
        )

        for balancing in balancings:
            subset = selected[(task, balancing)]
            counts = defaultdict(int)
            for speaker, _ in subset.values():
                counts[speaker] += 1
            speaker_predictions = {}
            label = "word" if task == "word" else "sentence"
            for score_name in score_names:
                aggregated = _aggregate_predictions(
                    predictions[score_name], subset, FDA_SCORES[task]
                )
                speaker_predictions[score_name] = aggregated
            primary_output = (
                f"word_{word_output}"
                if task == "word"
                else f"utterance_{utterance_output}"
            )
            _, target_pcc = _aggregate_and_score(
                predictions[primary_output], subset, FDA_SCORES[task]
            )
            paper_protocol = PATHBENCH_PROTOCOLS[balancing]
            metrics[label][paper_protocol] = {
                "pcc": target_pcc,
                "num_utterances": len(subset),
            }
            _write_speaker_csv(
                output_dir / (
                    f"{'word' if task == 'word' else 'utterance'}_"
                    f"{balancing}_speaker_scores.csv"
                ),
                speaker_predictions,
                FDA_SCORES[task],
                counts,
            )

    result = {
        "transcript_source": statistics["transcript_source"],
        "num_speakers": statistics["num_speakers"],
        "metrics": metrics,
    }
    if "asr" in statistics:
        result["asr"] = statistics["asr"]
    return result


def _can_reuse_reference_text(result, kwargs):
    """Return whether a flat Reference-Text result matches this evaluation."""
    expected = {
        "condition": "reference-text",
        "protocol": "PathBench",
        "pathbench_commit": PATHBENCH_COMMIT,
        "audio": "wav_arrayMic",
        "word_target": f"word_{kwargs.get('word_output', 'accuracy')}",
        "sentence_target": (
            f"utterance_{kwargs.get('utterance_output', 'accuracy')}"
        ),
        "transcript": "TORGO prompt",
    }
    if any(result.get(name) != value for name, value in expected.items()):
        return False

    protocols = tuple(
        PATHBENCH_PROTOCOLS[name]
        for name in kwargs.get("balancings", BALANCINGS)
    )
    present_protocols = {
        protocol
        for protocol in PATHBENCH_PROTOCOLS.values()
        if f"word_{protocol}_pcc" in result
        or f"sentence_{protocol}_pcc" in result
    }
    if present_protocols != set(protocols):
        return False
    for task in ("word", "sentence"):
        for protocol in protocols:
            if not all(
                f"{task}_{protocol}_{name}" in result
                for name in ("pcc", "num_utterances")
            ):
                return False

    # Older flat files did not record this field and represent a full run.
    requested_limit = kwargs.get("max_samples")
    saved_limit = result.get("max_samples_per_speaker")
    return saved_limit == requested_limit


def evaluate_torgo(
    model,
    processor,
    score_token_id,
    word_label_names,
    binary_word_label_indices,
    output_dir,
    conditions=("reference_text",),
    **kwargs,
):
    """Evaluate one or both PathBench transcript-availability conditions."""
    invalid = set(conditions) - set(TRANSCRIPT_CONDITIONS)
    if invalid:
        raise ValueError(f"Unknown TORGO conditions: {sorted(invalid)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_results = {}
    for condition in conditions:
        if condition == "reference_free":
            run_id = torgo_asr_run_id(
                kwargs.get("asr_model", "medium.en"),
                kwargs.get("asr_compute_type", "float16"),
                kwargs.get("asr_beam_size", 5),
                kwargs.get("asr_backend", "faster-whisper"),
            )
            condition_root = output_dir / "reference-free"
            condition_dir = condition_root / run_id
        else:
            run_id = None
            condition_root = output_dir / "reference-text"
            condition_dir = condition_root
        condition_dir.mkdir(parents=True, exist_ok=True)
        result_file = condition_root / "results.json"
        if condition == "reference_text" and result_file.exists():
            try:
                with result_file.open(encoding="utf-8") as handle:
                    cached_result = json.load(handle)
            except (json.JSONDecodeError, OSError):
                cached_result = None
            if cached_result and _can_reuse_reference_text(cached_result, kwargs):
                written_results[condition] = cached_result
                print(
                    "Skipping TORGO Reference-Text: compatible results already "
                    f"exist at {result_file}"
                )
                continue
        condition_result = _evaluate_torgo_condition(
            model,
            processor,
            score_token_id,
            word_label_names,
            binary_word_label_indices,
            condition_dir,
            condition,
            **kwargs,
        )
        flat_result = {
            "condition": condition.replace("_", "-"),
            "protocol": "PathBench",
            "pathbench_commit": PATHBENCH_COMMIT,
            "audio": "wav_arrayMic",
            "word_target": f"word_{kwargs.get('word_output', 'accuracy')}",
            "sentence_target": (
                f"utterance_{kwargs.get('utterance_output', 'accuracy')}"
            ),
            "transcript": condition_result["transcript_source"],
            "speakers": condition_result["num_speakers"],
            "balancing": ",".join(
                PATHBENCH_PROTOCOLS[name]
                for name in kwargs.get("balancings", BALANCINGS)
            ),
            "max_samples_per_speaker": kwargs.get("max_samples"),
        }
        for task, protocols in condition_result["metrics"].items():
            for protocol, values in protocols.items():
                for name, value in values.items():
                    flat_result[f"{task}_{protocol}_{name}"] = value
        for name, value in condition_result.get("asr", {}).items():
            flat_result[f"asr_{name}"] = value

        if run_id is None:
            result_data = flat_result
        else:
            if result_file.exists():
                with result_file.open(encoding="utf-8") as handle:
                    result_data = json.load(handle)
            else:
                result_data = {}
            result_data[run_id] = flat_result
        temporary = result_file.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result_data, handle, indent=2, sort_keys=True)
        temporary.replace(result_file)
        written_results[condition] = result_data
        print(json.dumps(result_data, indent=2, sort_keys=True))
        print(f"Saved TORGO PathBench evaluation to {result_file}")
    return written_results
