"""Shared preprocessing, metrics, and Trainer behavior for Whisper APA."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import Seq2SeqTrainer

from data import (
    DEFAULT_WORD_LABELS,
    PREPROCESSING_SCHEMA_VERSION,
    UTTERANCE_LABELS,
    preprocess_example,
)


APA_LABEL_NAMES = [
    "word_labels",
    "word_label_mask",
    "word_syllable_counts",
    "word_stress_all_labels",
    "utterance_labels",
]


def _stable_preprocessing_cache_file(
    dataset,
    processor,
    score_token_id,
    word_label_names,
    *,
    normalize_word_labels,
    normalize_utterance_labels,
    binary_stress,
    stress_multisyllabic_only,
    text_normalization,
    fluency_vad,
    fluency_vad_threshold,
    handcrafted_features,
):
    """Return a checkpoint-independent cache path for deterministic map output."""
    cache_root = os.environ.get("HF_DATASETS_CACHE")
    if not cache_root:
        return None

    tokenizer_vocab = sorted(
        processor.tokenizer.get_vocab().items(), key=lambda item: item[0]
    )
    signature = {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_rows": len(dataset),
        "dataset_columns": list(dataset.column_names),
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "feature_extractor": processor.feature_extractor.to_dict(),
        "tokenizer_vocab": tokenizer_vocab,
        "score_token_id": int(score_token_id),
        "word_label_names": list(word_label_names),
        "normalize_word_labels": bool(normalize_word_labels),
        "normalize_utterance_labels": bool(normalize_utterance_labels),
        "binary_stress": bool(binary_stress),
        "stress_multisyllabic_only": bool(stress_multisyllabic_only),
        "text_normalization": text_normalization,
        "fluency_vad": fluency_vad,
        "fluency_vad_threshold": float(fluency_vad_threshold),
        "handcrafted_features": bool(handcrafted_features),
    }
    serialized = json.dumps(
        signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()[:24]
    cache_dir = Path(cache_root) / "openpa_preprocessed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"apa-{digest}.arrow"


def safe_pearson(prediction, target):
    if len(prediction) < 2 or np.std(prediction) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def build_compute_metrics(
    word_label_names, binary_word_label_indices=(), word_edit_operations=None
):
    binary_indices = set(binary_word_label_indices)

    def compute_metrics(eval_prediction):
        word_pred, utterance_pred = eval_prediction.predictions
        (
            word_labels,
            word_mask,
            word_syllable_counts,
            word_stress_all_labels,
            utterance_labels,
        ) = eval_prediction.label_ids
        valid_words = word_mask == 1
        edit_operation = None
        if word_edit_operations is not None:
            if len(word_edit_operations) != word_mask.shape[0]:
                raise ValueError(
                    "Word edit-operation rows do not match evaluation examples: "
                    f"{len(word_edit_operations)} != {word_mask.shape[0]}"
                )
            edit_operation = np.zeros(word_mask.shape, dtype=np.int8)
            operation_codes = {"match": 1, "substitute": 2}
            for row, operations in enumerate(word_edit_operations):
                width = min(len(operations), edit_operation.shape[1])
                edit_operation[row, :width] = [
                    operation_codes.get(value, 0) for value in operations[:width]
                ]
        metrics = {}
        macro_metrics = []

        def add_edit_scoped_metrics(key, prediction, target, valid):
            if edit_operation is None:
                return
            for suffix, scope in (
                ("match", edit_operation == 1),
                ("substitution", edit_operation == 2),
                ("match_substitution", edit_operation > 0),
            ):
                selected = valid & scope
                metrics[f"{key.removesuffix('_pcc')}_{suffix}_pcc"] = safe_pearson(
                    prediction[selected], target[selected]
                )

        for index, name in enumerate(word_label_names):
            prediction = word_pred[:, :, index]
            if index in binary_indices:
                prediction = 1.0 / (1.0 + np.exp(-np.clip(prediction, -60.0, 60.0)))
                multisyllabic = (
                    valid_words
                    & (word_syllable_counts >= 2)
                    & (word_stress_all_labels != -100)
                )
                key = "word_stress_multisyllabic_pcc"
                metrics[key] = safe_pearson(
                    prediction[multisyllabic], word_stress_all_labels[multisyllabic]
                )
                add_edit_scoped_metrics(
                    key, prediction, word_stress_all_labels, multisyllabic
                )
                macro_metrics.append(key)

                all_prediction = prediction.copy()
                all_prediction[word_syllable_counts == 1] = 1.0
                all_valid = (
                    valid_words
                    & (word_syllable_counts >= 1)
                    & (word_stress_all_labels != -100)
                )
                metrics["word_stress_all_pcc"] = safe_pearson(
                    all_prediction[all_valid], word_stress_all_labels[all_valid]
                )
                add_edit_scoped_metrics(
                    "word_stress_all_pcc",
                    all_prediction,
                    word_stress_all_labels,
                    all_valid,
                )
            else:
                valid = valid_words & (word_labels[:, :, index] != -100)
                key = f"word_{name}_pcc"
                metrics[key] = safe_pearson(
                    prediction[valid], word_labels[:, :, index][valid]
                )
                add_edit_scoped_metrics(
                    key, prediction, word_labels[:, :, index], valid
                )
                macro_metrics.append(key)

        for index, name in enumerate(UTTERANCE_LABELS):
            key = f"utterance_{name}_pcc"
            metrics[key] = safe_pearson(
                utterance_pred[:, index], utterance_labels[:, index]
            )
            macro_metrics.append(key)

        metrics["apa_macro_pcc"] = float(
            np.mean([metrics[name] for name in macro_metrics])
        )
        return metrics

    return compute_metrics


def prepare_apa_dataset(
    dataset,
    processor,
    score_token_id,
    word_label_names=DEFAULT_WORD_LABELS,
    num_proc=1,
    desc=None,
    normalize_word_labels=False,
    binary_stress=False,
    normalize_utterance_labels=False,
    stress_multisyllabic_only=False,
    text_normalization="none",
    fluency_vad="energy",
    fluency_vad_threshold=0.5,
    handcrafted_features=False,
):
    """Apply the same cached preprocessing in training and evaluation."""
    cache_file = _stable_preprocessing_cache_file(
        dataset,
        processor,
        score_token_id,
        word_label_names,
        normalize_word_labels=normalize_word_labels,
        normalize_utterance_labels=normalize_utterance_labels,
        binary_stress=binary_stress,
        stress_multisyllabic_only=stress_multisyllabic_only,
        text_normalization=text_normalization,
        fluency_vad=fluency_vad,
        fluency_vad_threshold=fluency_vad_threshold,
        handcrafted_features=handcrafted_features,
    )
    map_kwargs = {
        "function": preprocess_example,
        "fn_kwargs": {
            "processor": processor,
            "score_token_id": score_token_id,
            "word_label_names": word_label_names,
            "normalize_word_labels": normalize_word_labels,
            "normalize_utterance_labels": normalize_utterance_labels,
            "binary_stress": binary_stress,
            "stress_multisyllabic_only": stress_multisyllabic_only,
            "text_normalization": text_normalization,
            "fluency_vad": fluency_vad,
            "fluency_vad_threshold": fluency_vad_threshold,
            "handcrafted_features": handcrafted_features,
            "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        },
        "remove_columns": dataset.column_names,
        # datasets treats num_proc=1 as a multiprocessing pool too. Forking a
        # worker after Torch/Whisper initialization can deadlock on the first
        # decoded audio example, so use the in-process path for one worker.
        "num_proc": None if num_proc <= 1 else num_proc,
        "desc": desc,
    }
    if cache_file is not None:
        map_kwargs["cache_file_name"] = str(cache_file)
        cache_parts = list(cache_file.parent.glob(f"{cache_file.stem}*{cache_file.suffix}"))
        status = "hit" if cache_parts else "miss"
        print(f"OpenPA preprocessing cache {status}: {cache_file}", flush=True)
    prepared = dataset.map(**map_kwargs)
    required = {
        "input_features",
        "attention_mask",
        "labels",
        "word_labels",
        "word_syllable_counts",
        "word_stress_all_labels",
        "utterance_labels",
        "fluency_features",
        "prosodic_features",
    }
    missing = required.difference(prepared.column_names)
    if missing:
        print(f"Ignoring stale cache; missing columns: {sorted(missing)}", flush=True)
        prepared = dataset.map(**map_kwargs, load_from_cache_file=False)
    return prepared


class APATrainer(Seq2SeqTrainer):
    """Collect compact APA outputs instead of full ASR vocabulary logits."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            outputs = model(**inputs)
        loss = outputs.loss.detach().mean() if outputs.loss is not None else None
        if prediction_loss_only:
            return loss, None, None
        predictions = (outputs.word_logits.detach(), outputs.utterance_logits.detach())
        labels = tuple(inputs[name].detach() for name in APA_LABEL_NAMES)
        return loss, predictions, labels

    def floating_point_ops(self, inputs):
        return 0

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.state.best_global_step == self.state.global_step:
            metric = self.args.metric_for_best_model or "best_metric"
            if not metric.startswith("eval_"):
                metric = f"eval_{metric}"
            print(
                f"Best model saved: checkpoint={self.state.best_model_checkpoint}, "
                f"{metric}={self.state.best_metric}",
                flush=True,
            )
