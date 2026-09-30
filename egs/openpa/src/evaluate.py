#!/usr/bin/env python3
"""Common Stage 2 runner for Whisper APA test-set adapters."""

import argparse
import json
from pathlib import Path

import torch
import tqdm as tqdm_module
from peft import PeftConfig, PeftModel
from transformers import (
    WhisperConfig,
    WhisperProcessor,
    enable_full_determinism,
    set_seed,
)

from data import (
    FLUENCY_FEATURE_NAMES,
    PROSODIC_FEATURE_NAMES,
    SCORE_TOKEN,
    UTTERANCE_LABELS,
)
from modeling_whisper_apa import WhisperForPronunciationAssessment, initialize_score_token_weights
from multipa import evaluate_multipa
from speechocean import evaluate_speechocean762
from torgo import BALANCINGS, evaluate_torgo
from youtube import evaluate_youtube


# Batch evaluation does not need tqdm's background monitor thread.
tqdm_module.tqdm.monitor_interval = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--test-sets",
        default="speechocean762,multipa",
        help="Comma-separated adapters: speechocean762, multipa, torgo, youtube, or all",
    )
    parser.add_argument("--dataset", default="mispeech/speechocean762")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--speechocean-conditions",
        default="close,open",
        help="Comma-separated transcript conditions: close, open, or both",
    )
    parser.add_argument(
        "--speechocean-cache-dir",
        default="data/speechocean762",
        help="Shared transcript/alignment cache root, independent of APA checkpoints",
    )
    parser.add_argument("--speechocean-transcript-source", default=None)
    parser.add_argument("--speechocean-transcript-split", default="train")
    parser.add_argument("--speechocean-transcript-output", default=None)
    parser.add_argument("--speechocean-asr-model", default="medium.en")
    parser.add_argument(
        "--speechocean-asr-backend",
        choices=("faster-whisper", "whisperx", "qwen3-asr"),
        default="faster-whisper",
    )
    parser.add_argument("--speechocean-asr-batch-size", type=int, default=16)
    parser.add_argument("--speechocean-asr-device", default="cuda:0")
    parser.add_argument("--speechocean-asr-compute-type", default="float16")
    parser.add_argument("--speechocean-asr-beam-size", type=int, default=5)
    parser.add_argument(
        "--speechocean-word-alignment",
        choices=("charsiu", "levenshtein"),
        default="charsiu",
    )
    parser.add_argument(
        "--speechocean-charsiu-model", default="charsiu/en_w2v2_fc_10ms"
    )
    parser.add_argument("--speechocean-charsiu-device", default="cuda:0")
    parser.add_argument("--speechocean-charsiu-cache-dir", default=None)
    parser.add_argument(
        "--speechocean-charsiu-model-cache-dir", default=".cache/huggingface/hub"
    )
    parser.add_argument("--multipa-output-dir", default=None)
    parser.add_argument(
        "--multipa-cache-dir",
        default="data/multipa",
        help="Shared transcript cache root, independent of APA checkpoints",
    )
    parser.add_argument(
        "--multipa-word-output",
        choices=("accuracy", "total"),
        default="accuracy",
        help="Model word-level output used for MultiPA labels (default: accuracy)",
    )
    parser.add_argument("--multipa-transcript-source", default=None)
    parser.add_argument("--multipa-transcript-split", default="train")
    parser.add_argument("--multipa-transcript-output", default=None)
    parser.add_argument("--multipa-asr-model", default="medium.en")
    parser.add_argument(
        "--multipa-asr-backend",
        choices=("faster-whisper", "whisperx", "qwen3-asr"),
        default="faster-whisper",
    )
    parser.add_argument("--multipa-asr-batch-size", type=int, default=16)
    parser.add_argument("--multipa-asr-device", default="cuda:0")
    parser.add_argument("--multipa-asr-compute-type", default="float16")
    parser.add_argument("--multipa-asr-beam-size", type=int, default=5)
    parser.add_argument("--multipa-max-samples", type=int, default=None)
    parser.add_argument("--torgo-output-dir", default=None)
    parser.add_argument("--torgo-root", default="data/torgo/raw")
    parser.add_argument(
        "--torgo-manifest-root", default="data/torgo/pathbench",
        help="Cache for manifests pinned to the evaluated PathBench commit",
    )
    parser.add_argument(
        "--torgo-balancing",
        choices=("balanced", "unbalanced", "both"),
        default="both",
    )
    parser.add_argument(
        "--torgo-conditions",
        default="reference_text",
        help="Comma-separated: reference_text, reference_free, or both",
    )
    parser.add_argument("--torgo-word-output", default="accuracy")
    parser.add_argument(
        "--torgo-utterance-output", choices=UTTERANCE_LABELS, default="accuracy"
    )
    parser.add_argument("--torgo-batch-size", type=int, default=32)
    parser.add_argument("--torgo-asr-cache-dir", default="data/torgo")
    parser.add_argument("--torgo-asr-model", default="medium.en")
    parser.add_argument(
        "--torgo-asr-backend",
        choices=("faster-whisper", "whisperx"),
        default="faster-whisper",
    )
    parser.add_argument("--torgo-asr-batch-size", type=int, default=16)
    parser.add_argument("--torgo-asr-device", default="cuda:0")
    parser.add_argument("--torgo-asr-compute-type", default="float16")
    parser.add_argument("--torgo-asr-beam-size", type=int, default=5)
    parser.add_argument(
        "--torgo-max-samples", type=int, default=None,
        help="Unbalanced-only smoke-test limit per speaker and task",
    )
    parser.add_argument("--youtube-output-dir", default=None)
    parser.add_argument("--youtube-root", default="data/youtube/raw")
    parser.add_argument(
        "--youtube-manifest-root", default="data/youtube/pathbench",
        help="Cache for manifests pinned to the evaluated PathBench commit",
    )
    parser.add_argument("--youtube-batch-size", type=int, default=32)
    parser.add_argument("--youtube-max-samples", type=int, default=None)
    parser.add_argument(
        "--youtube-conditions", default="reference_text",
        help="Comma-separated: reference_text, reference_free, or both",
    )
    parser.add_argument("--youtube-asr-cache-dir", default="data/youtube")
    parser.add_argument("--youtube-asr-model", default="medium.en")
    parser.add_argument(
        "--youtube-asr-backend",
        choices=("faster-whisper", "whisperx", "qwen3-asr"),
        default="faster-whisper",
    )
    parser.add_argument("--youtube-asr-batch-size", type=int, default=16)
    parser.add_argument("--youtube-asr-device", default="cuda:0")
    parser.add_argument("--youtube-asr-compute-type", default="float16")
    parser.add_argument("--youtube-asr-beam-size", type=int, default=5)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_checkpoint(
    checkpoint, local_files_only=False, attn_implementation=None
):
    checkpoint = Path(checkpoint)
    processor_path = (
        checkpoint
        if (checkpoint / "tokenizer_config.json").exists()
        else checkpoint.parent
    )
    processor = WhisperProcessor.from_pretrained(
        processor_path, local_files_only=local_files_only
    )
    score_token_id = processor.tokenizer.convert_tokens_to_ids(SCORE_TOKEN)
    if score_token_id == processor.tokenizer.unk_token_id:
        raise RuntimeError(f"{processor_path} does not contain {SCORE_TOKEN}")

    metadata_path = checkpoint / "apa_config.json"
    if not metadata_path.exists():
        metadata_path = checkpoint.parent / "apa_config.json"
    if metadata_path.exists():
        with metadata_path.open() as handle:
            apa_config = json.load(handle)
        word_label_names = tuple(apa_config["word_label_names"])
    else:
        apa_config = {}
        word_label_names = ("accuracy",)
    binary_word_label_indices = tuple(
        apa_config.get("binary_word_label_indices", [])
    )

    if (checkpoint / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(checkpoint)
        base_model_path = peft_config.base_model_name_or_path
        config = WhisperConfig.from_pretrained(
            base_model_path, local_files_only=local_files_only
        )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        config.num_word_scores = len(word_label_names)
        config.word_representation = apa_config.get(
            "word_representation", "score-token"
        )
        config.binary_word_label_indices = list(binary_word_label_indices)
        config.stress_loss_weight = apa_config.get("stress_loss_weight", 1.0)
        config.stress_incorrect_weight = apa_config.get(
            "stress_incorrect_weight", 1.0
        )
        config.num_utterance_scores = len(UTTERANCE_LABELS)
        config.utterance_label_names = list(UTTERANCE_LABELS)
        config.utterance_context_layer = apa_config.get(
            "utterance_context_layer", False
        )
        config.utterance_pooling = apa_config.get("utterance_pooling", "mean")
        config.handcrafted_features = apa_config.get(
            "handcrafted_features",
            apa_config.get("shared_handcrafted_fusion", False),
        )
        config.utterance_representation_dropout = apa_config.get(
            "utterance_representation_dropout",
            apa_config.get("handcrafted_high_dim_dropout", 0.0),
        )
        config.num_prosodic_features = len(
            apa_config.get("prosodic_feature_names", PROSODIC_FEATURE_NAMES)
        )
        config.num_fluency_features = len(
            apa_config.get("fluency_feature_names", FLUENCY_FEATURE_NAMES)
        )
        config.score_token_id = score_token_id
        config.word_label_names = list(word_label_names)
        base_model = WhisperForPronunciationAssessment.from_pretrained(
            base_model_path, config=config, local_files_only=local_files_only
        )
        base_model.resize_token_embeddings(len(processor.tokenizer))
        initialize_score_token_weights(base_model, score_token_id)
        model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=False)
    else:
        load_kwargs = {"local_files_only": local_files_only}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        model = WhisperForPronunciationAssessment.from_pretrained(
            checkpoint, **load_kwargs
        )

    model.config.use_cache = True
    model.eval()
    word_label_names = tuple(getattr(model.config, "word_label_names", word_label_names))
    return (
        model,
        processor,
        score_token_id,
        word_label_names,
        binary_word_label_indices,
        apa_config,
    )


def parse_test_sets(value):
    selected = []
    for name in value.replace(" ", "").split(","):
        if name == "both":
            names = ("speechocean762", "multipa")
        elif name == "all":
            names = ("speechocean762", "multipa", "torgo", "youtube")
        elif name in {"speechocean762", "multipa", "torgo", "youtube"}:
            names = (name,)
        else:
            raise ValueError(
                "--test-sets must contain speechocean762, multipa, torgo, youtube, "
                "both, or all; "
                f"got {value!r}"
            )
        for selected_name in names:
            if selected_name not in selected:
                selected.append(selected_name)
    return tuple(selected)


def parse_torgo_conditions(value):
    selected = []
    for name in value.replace(" ", "").split(","):
        names = ("reference_text", "reference_free") if name == "both" else (name,)
        for condition in names:
            if condition not in {"reference_text", "reference_free"}:
                raise ValueError(
                    "--torgo-conditions must contain reference_text, "
                    f"reference_free, or both; got {value!r}"
                )
            if condition not in selected:
                selected.append(condition)
    return tuple(selected)


def parse_youtube_conditions(value):
    selected = []
    for name in value.replace(" ", "").split(","):
        names = ("reference_text", "reference_free") if name == "both" else (name,)
        for condition in names:
            if condition not in {"reference_text", "reference_free"}:
                raise ValueError(
                    "--youtube-conditions must contain reference_text, "
                    f"reference_free, or both; got {value!r}"
                )
            if condition not in selected:
                selected.append(condition)
    return tuple(selected)


def main():
    args = parse_args()
    if args.full_determinism:
        enable_full_determinism(args.seed)
    else:
        set_seed(args.seed)
    test_sets = parse_test_sets(args.test_sets)

    checkpoint = Path(args.checkpoint)
    # Checkpoints live under the experiment directory (for example,
    # EXP/checkpoint-final), while evaluation artifacts belong beside them.
    evaluation_root = checkpoint.parent / "evaluation"

    print(f"Loading checkpoint: {checkpoint}")
    model, processor, score_token_id, word_label_names, binary_indices, apa_config = (
        load_checkpoint(checkpoint, args.local_files_only)
    )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        "Text normalization: "
        f"{apa_config.get('text_normalization', 'none')}",
        flush=True,
    )

    if "speechocean762" in test_sets:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else evaluation_root / "speechocean762"
        )
        evaluate_speechocean762(
            args, model, processor, score_token_id, word_label_names,
            binary_indices, apa_config, output_dir,
        )

    if "multipa" in test_sets:
        if args.multipa_word_output not in word_label_names:
            raise ValueError(
                f"Checkpoint {checkpoint} cannot be evaluated on MultiPA: "
                f"word_label_names={list(word_label_names)!r} does not include "
                f"{args.multipa_word_output!r}"
            )
        output_dir = (
            Path(args.multipa_output_dir)
            if args.multipa_output_dir
            else evaluation_root / "multipa"
        )
        evaluate_multipa(
            model,
            processor,
            score_token_id,
            word_label_names,
            output_dir,
            cache_dir=args.multipa_cache_dir,
            word_output=args.multipa_word_output,
            text_normalization=apa_config.get("text_normalization", "none"),
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.num_workers,
            transcript_source=args.multipa_transcript_source,
            transcript_split=args.multipa_transcript_split,
            transcript_output=args.multipa_transcript_output,
            asr_model=args.multipa_asr_model,
            asr_device=args.multipa_asr_device,
            asr_compute_type=args.multipa_asr_compute_type,
            asr_beam_size=args.multipa_asr_beam_size,
            asr_backend=args.multipa_asr_backend,
            asr_batch_size=args.multipa_asr_batch_size,
            max_samples=args.multipa_max_samples,
            local_files_only=args.local_files_only,
            normalize_word_labels=apa_config.get("normalize_word_labels", False),
            normalize_utterance_labels=apa_config.get(
                "normalize_utterance_labels", False
            ),
            fluency_vad=apa_config.get("fluency_vad", "energy"),
            fluency_vad_threshold=apa_config.get("fluency_vad_threshold", 0.5),
            handcrafted_features=apa_config.get(
                "handcrafted_features",
                apa_config.get("shared_handcrafted_fusion", False),
            ),
        )

    if "torgo" in test_sets:
        output_dir = (
            Path(args.torgo_output_dir)
            if args.torgo_output_dir
            else evaluation_root / "torgo"
        )
        balancings = (
            BALANCINGS if args.torgo_balancing == "both"
            else (args.torgo_balancing,)
        )
        evaluate_torgo(
            model,
            processor,
            score_token_id,
            word_label_names,
            binary_indices,
            output_dir,
            conditions=parse_torgo_conditions(args.torgo_conditions),
            dataset_root=args.torgo_root,
            manifest_root=args.torgo_manifest_root,
            balancings=balancings,
            word_output=args.torgo_word_output,
            utterance_output=args.torgo_utterance_output,
            batch_size=args.torgo_batch_size,
            num_workers=args.num_workers,
            max_samples=args.torgo_max_samples,
            fluency_vad=apa_config.get("fluency_vad", "energy"),
            fluency_vad_threshold=apa_config.get("fluency_vad_threshold", 0.5),
            handcrafted_features=apa_config.get(
                "handcrafted_features",
                apa_config.get("shared_handcrafted_fusion", False),
            ),
            inference_dtype=(
                "float16" if args.fp16 else "bfloat16" if args.bf16 else None
            ),
            asr_cache_dir=args.torgo_asr_cache_dir,
            asr_model=args.torgo_asr_model,
            asr_backend=args.torgo_asr_backend,
            asr_batch_size=args.torgo_asr_batch_size,
            asr_device=args.torgo_asr_device,
            asr_compute_type=args.torgo_asr_compute_type,
            asr_beam_size=args.torgo_asr_beam_size,
            local_files_only=args.local_files_only,
        )

    if "youtube" in test_sets:
        output_dir = (
            Path(args.youtube_output_dir)
            if args.youtube_output_dir
            else evaluation_root / "youtube"
        )
        evaluate_youtube(
            model,
            processor,
            score_token_id,
            word_label_names,
            binary_indices,
            output_dir,
            conditions=parse_youtube_conditions(args.youtube_conditions),
            dataset_root=args.youtube_root,
            manifest_root=args.youtube_manifest_root,
            batch_size=args.youtube_batch_size,
            num_workers=args.num_workers,
            max_samples=args.youtube_max_samples,
            fluency_vad=apa_config.get("fluency_vad", "energy"),
            fluency_vad_threshold=apa_config.get("fluency_vad_threshold", 0.5),
            handcrafted_features=apa_config.get(
                "handcrafted_features",
                apa_config.get("shared_handcrafted_fusion", False),
            ),
            inference_dtype=(
                "float16" if args.fp16 else "bfloat16" if args.bf16 else None
            ),
            asr_cache_dir=args.youtube_asr_cache_dir,
            asr_model=args.youtube_asr_model,
            asr_backend=args.youtube_asr_backend,
            asr_batch_size=args.youtube_asr_batch_size,
            asr_device=args.youtube_asr_device,
            asr_compute_type=args.youtube_asr_compute_type,
            asr_beam_size=args.youtube_asr_beam_size,
            local_files_only=args.local_files_only,
        )


if __name__ == "__main__":
    main()
