"""Close- and open-response SpeechOcean762 evaluation adapters."""

import json
import re
from pathlib import Path

import ctranslate2
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import Seq2SeqTrainingArguments

from apa_utils import APA_LABEL_NAMES, APATrainer, build_compute_metrics, prepare_apa_dataset
from asr_transcripts import (
    load_faster_whisper_model,
    load_qwen3_asr_model,
    load_whisperx_model,
    load_transcript_records,
    transcript_result_key,
    transcribe_faster_whisper,
    transcribe_qwen3_asr,
    transcribe_whisperx,
    write_transcript_jsonl,
)
from charsiu_alignment import CharsiuForcedAligner
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
from multipa_normalization import normalize_multipa_text, normalize_multipa_word_units


def parse_speechocean_conditions(value):
    requested = tuple(name for name in value.replace(" ", "").split(",") if name)
    conditions = tuple(
        condition
        for name in requested
        for condition in (("close", "open") if name == "both" else (name,))
    )
    invalid = set(conditions).difference({"close", "open"})
    if invalid or not conditions:
        raise ValueError(
            "--speechocean-conditions must contain close, open, or both; "
            f"got {value!r}"
        )
    return tuple(dict.fromkeys(conditions))


def _audio_id(example, index):
    path = example["audio"].get("path")
    return Path(path).name if path else str(index)


def _align_asr_words(reference, hypothesis, return_operations=False):
    """Map each hypothesis word to a reference index using edit alignment."""
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    operation = [[None] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        cost[row][0], operation[row][0] = row, "delete"
    for column in range(1, columns + 1):
        cost[0][column], operation[0][column] = column, "insert"
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = "match" if reference[row - 1] == hypothesis[column - 1] else "substitute"
            candidates = [
                (cost[row - 1][column - 1] + (diagonal != "match"), diagonal),
                (cost[row - 1][column] + 1, "delete"),
                (cost[row][column - 1] + 1, "insert"),
            ]
            cost[row][column], operation[row][column] = min(
                candidates, key=lambda item: item[0]
            )

    mapping = [None] * columns
    hypothesis_operations = ["insert"] * columns
    row, column = rows, columns
    while row or column:
        current = operation[row][column]
        if current in {"match", "substitute"}:
            mapping[column - 1] = row - 1
            hypothesis_operations[column - 1] = current
            row -= 1
            column -= 1
        elif current == "delete":
            row -= 1
        else:
            column -= 1
    if return_operations:
        return mapping, hypothesis_operations
    return mapping


def _word_error_counts(reference, hypothesis):
    """Return corpus-additive Levenshtein substitution/deletion/insertion/hit counts."""
    reference = reference.split()
    hypothesis = hypothesis.split()
    # State: (edit distance, substitutions, deletions, insertions, hits).
    previous = [(index, 0, 0, index, 0) for index in range(len(hypothesis) + 1)]
    for reference_word in reference:
        first = previous[0]
        current = [(first[0] + 1, first[1], first[2] + 1, first[3], first[4])]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, 1):
            diagonal = previous[hypothesis_index - 1]
            if reference_word == hypothesis_word:
                candidates = [
                    (diagonal[0], diagonal[1], diagonal[2], diagonal[3], diagonal[4] + 1)
                ]
            else:
                candidates = [
                    (diagonal[0] + 1, diagonal[1] + 1, diagonal[2],
                     diagonal[3], diagonal[4])
                ]
            deletion = previous[hypothesis_index]
            insertion = current[hypothesis_index - 1]
            candidates.extend([
                (deletion[0] + 1, deletion[1], deletion[2] + 1,
                 deletion[3], deletion[4]),
                (insertion[0] + 1, insertion[1], insertion[2],
                 insertion[3] + 1, insertion[4]),
            ])
            current.append(min(candidates, key=lambda state: state[0]))
        previous = current
    _, substitutions, deletions, insertions, hits = previous[-1]
    return substitutions, deletions, insertions, hits


def _open_transcript_metrics(dataset, transcript_records):
    """Compute corpus WER and non-deletion reference-word coverage."""
    totals = np.zeros(4, dtype=np.int64)
    hypothesis_words = 0
    for index, example in enumerate(dataset):
        audio_id = _audio_id(example, index)
        record = transcript_records[audio_id]
        reference = normalize_multipa_text(
            " ".join(str(word["text"]).strip() for word in example["words"])
        )
        hypothesis = normalize_multipa_text(
            record.get("transcript")
            or " ".join(str(word["word"]).strip() for word in record["words"])
        )
        totals += _word_error_counts(reference, hypothesis)
        hypothesis_words += len(hypothesis.split())

    substitutions, deletions, insertions, hits = map(int, totals)
    reference_words = hits + substitutions + deletions
    if reference_words == 0:
        raise ValueError("Open evaluation WER requires at least one reference word")
    aligned_reference_words = hits + substitutions
    wer = (substitutions + deletions + insertions) / reference_words
    coverage = aligned_reference_words / reference_words
    return {
        "wer": float(wer),
        "wer_percent": float(100.0 * wer),
        "word_coverage": float(coverage),
        "word_coverage_percent": float(100.0 * coverage),
        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,
        "alignment_hits": hits,
        "alignment_substitutions": substitutions,
        "alignment_deletions": deletions,
        "alignment_insertions": insertions,
        "coverage_definition": "(alignment_hits + alignment_substitutions) / reference_words",
        "wer_normalizer": "multipa_paper",
    }


def _syllable_count(word):
    return sum(str(phone)[-1:] in {"0", "1", "2"} for phone in word.get("phones", []))


def _open_word_targets(word, word_label_names, apa_config):
    labels = []
    for name in word_label_names:
        value = float(word[name])
        if name == "stress" and apa_config.get("binary_stress", False):
            syllables = _syllable_count(word)
            if apa_config.get("stress_multisyllabic_only", False) and syllables < 2:
                value = -100.0
            elif value == 10:
                value = 1.0
            elif value == 5:
                value = 0.0
            else:
                raise ValueError(f"Unexpected stress label {value}; expected 5 or 10")
        elif apa_config.get("normalize_word_labels", False):
            value /= 10.0
        labels.append(value)
    return labels


def _load_alignment_jsonl(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["audio_id"]: row for row in rows}


def _write_alignment_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for audio_id in sorted(records):
            handle.write(json.dumps(records[audio_id], ensure_ascii=False) + "\n")
    temporary.replace(path)


def _prepare_charsiu_alignments(args, dataset, transcript_records, transcript_reference, output_dir):
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.speechocean_charsiu_model)
    shared_cache_dir = (
        Path(args.speechocean_cache_dir)
        / re.sub(r"[^A-Za-z0-9_.-]+", "_", args.split)
    )
    cache_dir = Path(
        args.speechocean_charsiu_cache_dir
        or shared_cache_dir / "alignments" / model_slug
    )
    gt_path = cache_dir / "ground-truth.jsonl"
    asr_path = cache_dir / f"{transcript_result_key(transcript_reference)}.jsonl"
    gt_records = _load_alignment_jsonl(gt_path)
    asr_records = _load_alignment_jsonl(asr_path)
    missing = []
    for index, example in enumerate(dataset):
        audio_id = _audio_id(example, index)
        if audio_id not in gt_records or audio_id not in asr_records:
            missing.append(index)
    if missing:
        from tqdm.auto import tqdm

        aligner = CharsiuForcedAligner(
            model_name=args.speechocean_charsiu_model,
            device=args.speechocean_charsiu_device,
            local_files_only=args.local_files_only,
            cache_dir=args.speechocean_charsiu_model_cache_dir,
        )
        for index in tqdm(
            missing,
            desc="SpeechOcean762 Charsiu alignment",
            unit="audio",
            dynamic_ncols=True,
        ):
            example = dataset[index]
            audio_id = _audio_id(example, index)
            waveform, _ = decode_audio(example["audio"])
            if audio_id not in gt_records:
                words = normalize_multipa_word_units(
                    [str(word["text"]).strip() for word in example["words"]]
                )
                text = " ".join(word for word in words if word)
                try:
                    aligned_words = aligner.align(waveform, text)
                    error = None
                except ValueError as exception:
                    aligned_words = []
                    error = str(exception)
                gt_records[audio_id] = {
                    "audio_id": audio_id,
                    "words": aligned_words,
                    "error": error,
                }
                _write_alignment_jsonl(gt_records, gt_path)
            if audio_id not in asr_records:
                words = transcript_records[audio_id]["words"]
                words = normalize_multipa_word_units([word["word"] for word in words])
                text = " ".join(word for word in words if word)
                if not text:
                    text = "nospeech"
                try:
                    aligned_words = aligner.align(waveform, text)
                    error = None
                except ValueError as exception:
                    aligned_words = []
                    error = str(exception)
                asr_records[audio_id] = {
                    "audio_id": audio_id,
                    "words": aligned_words,
                    "error": error,
                }
                _write_alignment_jsonl(asr_records, asr_path)
        del aligner
        if str(args.speechocean_charsiu_device).startswith("cuda"):
            torch.cuda.empty_cache()
    return gt_records, asr_records, {
        "method": "charsiu_timestamp_overlap",
        "model": args.speechocean_charsiu_model,
        "ground_truth_cache": str(gt_path),
        "asr_cache": str(asr_path),
        "ground_truth_failures": sum(not row["words"] for row in gt_records.values()),
        "asr_failures": sum(not row["words"] for row in asr_records.values()),
    }


def _charsiu_word_targets(
    reference_words, gt_spans, asr_spans, word_label_names, apa_config
):
    normalized_reference = normalize_multipa_word_units(
        [str(word["text"]).strip() for word in reference_words]
    )
    normalized_aligned = normalize_multipa_word_units(
        [span["word"] for span in gt_spans]
    )
    gt_mapping = _align_asr_words(normalized_reference, normalized_aligned)
    targets = []
    syllable_counts = []
    stress_all_labels = []
    for asr_span in asr_spans:
        overlaps = []
        for gt_span, reference_index in zip(gt_spans, gt_mapping):
            if reference_index is None:
                continue
            overlap = max(
                0.0,
                min(asr_span["end"], gt_span["end"])
                - max(asr_span["start"], gt_span["start"]),
            )
            if overlap > 0:
                overlaps.append((overlap, reference_index))
        if not overlaps:
            targets.append([-100.0] * len(word_label_names))
            syllable_counts.append(-100)
            stress_all_labels.append(-100.0)
            continue

        labels = []
        for label_index in range(len(word_label_names)):
            values = [
                (overlap, _open_word_targets(
                    reference_words[reference_index], word_label_names, apa_config
                )[label_index])
                for overlap, reference_index in overlaps
            ]
            values = [(weight, value) for weight, value in values if value != -100.0]
            labels.append(
                sum(weight * value for weight, value in values)
                / sum(weight for weight, _ in values)
                if values else -100.0
            )
        targets.append(labels)
        _, best_index = max(overlaps, key=lambda item: item[0])
        reference_word = reference_words[best_index]
        syllable_counts.append(_syllable_count(reference_word))
        stress = float(reference_word.get("stress", -100.0))
        stress_all_labels.append(
            1.0 if apa_config.get("binary_stress", False) and stress == 10.0
            else 0.0 if apa_config.get("binary_stress", False) and stress == 5.0
            else -100.0
        )
    return targets, syllable_counts, stress_all_labels


def _prepare_open_features(
    dataset, transcript_records, processor, score_token_id, word_label_names, apa_config,
    gt_alignments=None, asr_alignments=None,
):
    features = []
    word_edit_operations = []
    matched_words = 0
    text_normalization = apa_config.get("text_normalization", "none")
    for index, example in enumerate(dataset):
        audio_id = _audio_id(example, index)
        if audio_id not in transcript_records:
            raise ValueError(f"SpeechOcean transcript source is missing {audio_id}")
        use_charsiu = (
            gt_alignments is not None
            and asr_alignments is not None
            and bool(gt_alignments[audio_id]["words"])
            and bool(asr_alignments[audio_id]["words"])
        )
        if use_charsiu:
            raw_asr_words = [word["word"] for word in asr_alignments[audio_id]["words"]]
        else:
            raw_asr_words = [word["word"] for word in transcript_records[audio_id]["words"]]
        if not raw_asr_words:
            raw_asr_words = ["nospeech"]

        reference_words = [str(word["text"]).strip() for word in example["words"]]
        alignment_hypothesis = normalize_multipa_word_units(raw_asr_words)
        alignment_reference = normalize_multipa_word_units(reference_words)
        edit_mapping, edit_operations = _align_asr_words(
            alignment_reference, alignment_hypothesis, return_operations=True
        )
        mapping = None if use_charsiu else edit_mapping

        if text_normalization == "multipa":
            decoder_words = alignment_hypothesis
        elif text_normalization == "none":
            decoder_words = raw_asr_words
        else:
            raise ValueError(f"Unknown text normalization: {text_normalization!r}")
        if mapping is not None:
            selected = [
                (word, ref, operation)
                for word, ref, operation in zip(
                    decoder_words, mapping, edit_operations
                )
                if word
            ]
            decoder_words = [word for word, _, _ in selected]
            mapping = [ref for _, ref, _ in selected]
            edit_operations = [operation for _, _, operation in selected]
        else:
            selected = [
                (word, operation)
                for word, operation in zip(decoder_words, edit_operations)
                if word
            ]
            decoder_words = [word for word, _ in selected]
            edit_operations = [operation for _, operation in selected]
        if not decoder_words:
            decoder_words, mapping = ["nospeech"], [None]
            edit_operations = ["insert"]

        if use_charsiu:
            word_labels, syllable_counts, stress_all_labels = _charsiu_word_targets(
                example["words"],
                gt_alignments[audio_id]["words"],
                asr_alignments[audio_id]["words"],
                word_label_names,
                apa_config,
            )
            matched_words += sum(
                any(value != -100.0 for value in labels) for labels in word_labels
            )
        else:
            word_labels = []
            syllable_counts = []
            stress_all_labels = []
        for reference_index in mapping or []:
            if reference_index is None:
                word_labels.append([-100.0] * len(word_label_names))
                syllable_counts.append(-100)
                stress_all_labels.append(-100.0)
                continue
            reference_word = example["words"][reference_index]
            word_labels.append(_open_word_targets(reference_word, word_label_names, apa_config))
            syllables = _syllable_count(reference_word)
            syllable_counts.append(syllables)
            stress = float(reference_word.get("stress", -100.0))
            stress_all_labels.append(
                1.0 if apa_config.get("binary_stress", False) and stress == 10.0
                else 0.0 if apa_config.get("binary_stress", False) and stress == 5.0
                else -100.0
            )
            matched_words += 1

        waveform, sampling_rate = decode_audio(example["audio"])
        utterance_labels = [float(example[name]) for name in UTTERANCE_LABELS]
        if apa_config.get("normalize_utterance_labels", False):
            utterance_labels = [value / 10.0 for value in utterance_labels]
        acoustic_inputs = processor.feature_extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_attention_mask=True,
        )
        use_handcrafted_features = apa_config.get(
            "handcrafted_features",
            apa_config.get("shared_handcrafted_fusion", False),
        )
        if use_handcrafted_features:
            fluency_features, acoustic_analysis = extract_fluency_features(
                waveform,
                sampling_rate,
                vad_method=apa_config.get("fluency_vad", "energy"),
                vad_threshold=apa_config.get("fluency_vad_threshold", 0.5),
                return_analysis=True,
            )
            prosodic_features = extract_prosodic_features(
                waveform,
                sampling_rate,
                vad_method=apa_config.get("fluency_vad", "energy"),
                vad_threshold=apa_config.get("fluency_vad_threshold", 0.5),
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
            "labels": build_apa_token_ids(processor.tokenizer, decoder_words, score_token_id),
            "word_labels": word_labels,
            "word_syllable_counts": syllable_counts,
            "word_stress_all_labels": stress_all_labels,
            "utterance_labels": utterance_labels,
        })
        word_edit_operations.append(edit_operations)
    return features, matched_words, word_edit_operations


def _prepare_open_transcripts(args, dataset, output_dir):
    if args.speechocean_transcript_source:
        source = args.speechocean_transcript_source
        records = load_transcript_records(
            source, split=args.speechocean_transcript_split, require_timestamps=False
        )
        return records, f"{source}[split={args.speechocean_transcript_split}]"

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.speechocean_asr_model)
    shared_cache_dir = (
        Path(args.speechocean_cache_dir)
        / re.sub(r"[^A-Za-z0-9_.-]+", "_", args.split)
    )
    transcript_path = Path(
        args.speechocean_transcript_output
        or shared_cache_dir / "transcripts"
        / (
            f"{args.speechocean_asr_backend}-{slug}-{args.speechocean_asr_compute_type}"
            + (
                ""
                if args.speechocean_asr_backend == "qwen3-asr"
                else f"-beam{args.speechocean_asr_beam_size}"
            )
            + ".jsonl"
        )
    )
    records = (
        load_transcript_records(transcript_path, require_timestamps=False)
        if transcript_path.exists()
        else {}
    )
    missing = [
        index for index, example in enumerate(dataset)
        if _audio_id(example, index) not in records
    ]
    if missing:
        from tqdm.auto import tqdm

        if args.speechocean_asr_backend == "whisperx":
            model, asr_config = load_whisperx_model(
                args.speechocean_asr_model, args.speechocean_asr_device,
                args.speechocean_asr_compute_type,
                args.speechocean_asr_beam_size,
                args.speechocean_asr_batch_size, args.local_files_only,
            )
            transcribe = transcribe_whisperx
        elif args.speechocean_asr_backend == "faster-whisper":
            model, asr_config = load_faster_whisper_model(
                args.speechocean_asr_model, args.speechocean_asr_device,
                args.speechocean_asr_compute_type,
                args.speechocean_asr_beam_size, args.local_files_only,
            )
            transcribe = transcribe_faster_whisper
        else:
            model, asr_config = load_qwen3_asr_model(
                args.speechocean_asr_model,
                args.speechocean_asr_device,
                args.speechocean_asr_compute_type,
                args.local_files_only,
            )
            transcribe = transcribe_qwen3_asr
        for index in tqdm(
            missing, desc="SpeechOcean762 open ASR", unit="audio", dynamic_ncols=True
        ):
            example = dataset[index]
            audio_id = _audio_id(example, index)
            waveform, _ = decode_audio(example["audio"])
            ctranslate2.set_random_seed(asr_config["random_seed"])
            record = transcribe(model, waveform, asr_config)
            record.update({"audio_id": audio_id, "asr": asr_config})
            records[audio_id] = record
            write_transcript_jsonl(records, transcript_path)
        del model
    return records, str(transcript_path)


def _alignment_result_key(alignment):
    method = alignment.get("method", "") if isinstance(alignment, dict) else str(alignment)
    if method.startswith("charsiu"):
        return "charsiu"
    if method.startswith("levenshtein"):
        return "levenshtein"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method) or "unknown"


def _write_evaluation_result(
    result_file, result, result_key=None, result_variant=None
):
    """Atomically write one result or update a transcript-keyed result file."""
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    if result_key is None:
        saved_result = result
    else:
        if result_file.exists():
            with result_file.open(encoding="utf-8") as handle:
                saved_result = json.load(handle)
        else:
            saved_result = {}
        if result_variant is None:
            saved_result[result_key] = result
        else:
            existing = saved_result.get(result_key, {})
            if existing and "condition" in existing:
                # Migrate the previous transcript -> metrics layout without
                # dropping the already computed alignment result.
                existing = {
                    _alignment_result_key(existing.get("word_alignment")): existing
                }
            existing[result_variant] = result
            saved_result[result_key] = existing
    temporary = result_file.with_suffix(result_file.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(saved_result, handle, indent=2, sort_keys=True)
    temporary.replace(result_file)


def _evaluate_prepared(
    args, dataset, model, processor, score_token_id, word_label_names,
    binary_indices, output_dir, metric_prefix, metadata=None, result_file=None,
    result_key=None, result_variant=None, word_edit_operations=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
        bf16=args.bf16,
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=APA_LABEL_NAMES,
        dataloader_num_workers=args.num_workers,
        disable_tqdm=False,
        report_to=[],
    )
    trainer = APATrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWhisperAPA(
            processor,
            score_token_id,
            word_representation=getattr(
                model.config, "word_representation", "score-token"
            ),
        ),
        processing_class=processor.feature_extractor,
        compute_metrics=build_compute_metrics(
            word_label_names, binary_indices, word_edit_operations
        ),
    )
    metrics = trainer.evaluate(dataset, metric_key_prefix=metric_prefix)
    result = {
        key: float(value)
        for key, value in metrics.items()
        if key.endswith("_loss") or key.endswith("_pcc")
    }
    if metadata:
        result.update(metadata)
    if word_edit_operations is not None:
        print("\nSpeechOcean762 open word PCC (M/S/M+S)")
        print(f"{'score':<30} {'M':>9} {'S':>9} {'M+S':>9}")
        word_metric_stems = []
        for index, name in enumerate(word_label_names):
            if index in binary_indices:
                word_metric_stems.append("word_stress_multisyllabic")
            else:
                word_metric_stems.append(f"word_{name}")
        for stem in word_metric_stems:
            keys = [
                f"{metric_prefix}_{stem}_match_pcc",
                f"{metric_prefix}_{stem}_substitution_pcc",
                f"{metric_prefix}_{stem}_match_substitution_pcc",
            ]
            if all(key in result for key in keys):
                print(
                    f"{stem.removeprefix('word_'):<30} "
                    + " ".join(f"{result[key]:9.4f}" for key in keys)
                )
    result_file = Path(
        result_file
        or output_dir / args.split.replace("-", "_") / "results.json"
    )
    _write_evaluation_result(result_file, result, result_key, result_variant)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved SpeechOcean762 metrics to {result_file}")
    return result


def evaluate_speechocean762(
    args, model, processor, score_token_id, word_label_names, binary_indices,
    apa_config, output_dir,
):
    """Evaluate SpeechOcean762 with ground-truth and/or ASR transcripts."""
    print(f"Loading SpeechOcean762 dataset: {args.dataset}[{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    required = {"audio", "words", *UTTERANCE_LABELS}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"  samples={len(dataset)}")

    conditions = parse_speechocean_conditions(args.speechocean_conditions)
    split = args.split.replace("-", "_")
    open_data = None
    if "open" in conditions:
        records, transcript_reference = _prepare_open_transcripts(args, dataset, output_dir)
        transcript_metrics = _open_transcript_metrics(dataset, records)
        gt_alignments = asr_alignments = None
        alignment_reference = "levenshtein_reference_to_asr"
        if args.speechocean_word_alignment == "charsiu":
            gt_alignments, asr_alignments, alignment_reference = (
                _prepare_charsiu_alignments(
                    args, dataset, records, transcript_reference, output_dir
                )
            )
        open_features, matched_words, word_edit_operations = _prepare_open_features(
            dataset,
            records,
            processor,
            score_token_id,
            word_label_names,
            apa_config,
            gt_alignments=gt_alignments,
            asr_alignments=asr_alignments,
        )
        open_data = (
            open_features,
            matched_words,
            transcript_reference,
            alignment_reference,
            word_edit_operations,
        )

    if "close" in conditions:
        close_dataset = prepare_apa_dataset(
            dataset, processor, score_token_id, word_label_names,
            num_proc=args.preprocessing_num_workers,
            desc=f"Preprocessing close {args.split} split",
            normalize_word_labels=apa_config.get("normalize_word_labels", False),
            normalize_utterance_labels=apa_config.get("normalize_utterance_labels", False),
            binary_stress=apa_config.get("binary_stress", False),
            stress_multisyllabic_only=apa_config.get("stress_multisyllabic_only", False),
            text_normalization=apa_config.get("text_normalization", "none"),
            fluency_vad=apa_config.get("fluency_vad", "energy"),
            fluency_vad_threshold=apa_config.get("fluency_vad_threshold", 0.5),
            handcrafted_features=apa_config.get(
                "handcrafted_features",
                apa_config.get("shared_handcrafted_fusion", False),
            ),
        )
        _evaluate_prepared(
            args, close_dataset, model, processor, score_token_id, word_label_names,
            binary_indices, output_dir / "close", f"close_{split}",
            {"condition": "close", "transcript": "ground_truth"},
        )

    if "open" in conditions:
        (
            open_features,
            matched_words,
            transcript_reference,
            alignment_reference,
            word_edit_operations,
        ) = open_data
        _evaluate_prepared(
            args, open_features, model, processor, score_token_id, word_label_names,
            binary_indices,
            output_dir / "open",
            f"open_{split}",
            {
                "condition": "open",
                "transcript": transcript_reference,
                "word_alignment": alignment_reference,
                "matched_words": matched_words,
                "utterances": len(open_features),
                **transcript_metrics,
            },
            result_file=output_dir / "open" / "results.json",
            result_key=transcript_result_key(transcript_reference),
            result_variant=_alignment_result_key(alignment_reference),
            word_edit_operations=word_edit_operations,
        )
