#!/usr/bin/env python3
"""Score one audio file with reference-text or reference-free OpenPA."""

import argparse
import gc
import json
import re
import string
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from num2words import num2words
from peft import PeftConfig, PeftModel
from scipy.signal import resample_poly
from transformers import WhisperConfig, WhisperProcessor


MODEL_ID = "fuann/openpa"
SCORE_TOKEN = "<score>"
UTTERANCE_LABELS = ("accuracy", "fluency", "prosodic", "total")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Audio file to score")
    parser.add_argument(
        "--reference-text",
        "--transcript",
        dest="reference_text",
        help=(
            "Reference transcript. When omitted, WhisperX generates the "
            "transcript for reference-free inference"
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"Local model directory or Hugging Face repo (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="OpenPA and WhisperX device (default: cuda when available)",
    )
    parser.add_argument(
        "--asr-model",
        choices=("medium.en", "large-v3"),
        default="medium.en",
        help="WhisperX model used when --reference-text is omitted",
    )
    parser.add_argument(
        "--asr-compute-type",
        default=None,
        help="WhisperX compute type (default: float16 on CUDA, int8 on CPU)",
    )
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--asr-beam-size", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_model_path(model, local_files_only):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    return Path(
        snapshot_download(repo_id=model, local_files_only=local_files_only)
    )


def normalize_transcript(text):
    punctuation = string.punctuation.replace("'", "")
    text = str(text).translate(str.maketrans("", "", punctuation)).lower()
    compact = text.replace(" ", "")
    if compact.isdigit():
        text = " ".join(num2words(int(character)) for character in compact)
    else:
        text = " ".join(
            num2words(int(token)) if token.isdigit() else token
            for token in text.split()
        )
    return re.sub(r"\s+", " ", text).strip()


def load_audio(path, target_rate=16000):
    waveform, sampling_rate = sf.read(path, always_2d=False, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sampling_rate != target_rate:
        divisor = np.gcd(sampling_rate, target_rate)
        waveform = resample_poly(
            waveform,
            target_rate // divisor,
            sampling_rate // divisor,
        ).astype(np.float32)
    return np.ascontiguousarray(waveform), target_rate


def whisperx_device(device):
    """Convert a torch device into WhisperX/CTranslate2 device arguments."""
    if device.type == "cuda":
        return "cuda", 0 if device.index is None else device.index
    return "cpu", 0


def transcribe_with_whisperx(waveform, args, device):
    """Generate an English transcript without loading a separate VAD model."""
    try:
        import whisperx
        from whisperx.vads.vad import Vad
    except ImportError as error:
        raise RuntimeError(
            "Reference-free inference requires WhisperX. Install the model "
            "requirements or run `pip install whisperx`."
        ) from error

    class FullAudioVad(Vad):
        def __init__(self):
            super().__init__(0.5)

        @staticmethod
        def preprocess_audio(audio):
            return audio

        def __call__(self, audio, **kwargs):
            del kwargs
            duration = audio["waveform"].shape[-1] / audio["sample_rate"]
            return [SimpleNamespace(start=0.0, end=duration, speaker="UNKNOWN")]

    device_name, device_index = whisperx_device(device)
    compute_type = args.asr_compute_type
    if compute_type is None:
        compute_type = "float16" if device_name == "cuda" else "int8"

    asr_model = whisperx.load_model(
        args.asr_model,
        device=device_name,
        device_index=device_index,
        compute_type=compute_type,
        language="en",
        task="transcribe",
        asr_options={
            "beam_size": args.asr_beam_size,
            "condition_on_previous_text": False,
        },
        vad_model=FullAudioVad(),
        vad_method=None,
        local_files_only=args.local_files_only,
    )
    try:
        result = asr_model.transcribe(
            waveform,
            batch_size=args.asr_batch_size,
            language="en",
            task="transcribe",
            print_progress=False,
            verbose=False,
        )
        transcript = " ".join(
            segment["text"].strip() for segment in result["segments"]
        ).strip()
    finally:
        del asr_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not transcript:
        raise RuntimeError("WhisperX produced an empty transcript")
    return transcript, compute_type


def configure_model(model_path, device):
    sys.path.insert(0, str(model_path))
    from modeling_whisper_apa import (
        WhisperForPronunciationAssessment,
        initialize_score_token_weights,
    )

    with (model_path / "apa_config.json").open(encoding="utf-8") as handle:
        apa_config = json.load(handle)
    processor = WhisperProcessor.from_pretrained(model_path)
    score_token_id = processor.tokenizer.convert_tokens_to_ids(SCORE_TOKEN)
    if score_token_id == processor.tokenizer.unk_token_id:
        raise RuntimeError(f"Model tokenizer does not contain {SCORE_TOKEN}")

    peft_config = PeftConfig.from_pretrained(model_path)
    config = WhisperConfig.from_pretrained(peft_config.base_model_name_or_path)
    config.num_word_scores = len(apa_config["word_label_names"])
    config.word_label_names = list(apa_config["word_label_names"])
    config.word_representation = apa_config.get("word_representation", "score-token")
    config.binary_word_label_indices = apa_config.get(
        "binary_word_label_indices", []
    )
    config.num_utterance_scores = len(UTTERANCE_LABELS)
    config.utterance_label_names = list(UTTERANCE_LABELS)
    config.utterance_context_layer = apa_config.get(
        "utterance_context_layer", False
    )
    config.utterance_pooling = apa_config.get("utterance_pooling", "mean")
    config.handcrafted_features = apa_config.get("handcrafted_features", False)
    config.utterance_representation_dropout = apa_config.get(
        "utterance_representation_dropout", 0.0
    )
    config.num_fluency_features = len(apa_config.get("fluency_feature_names", []))
    config.num_prosodic_features = len(apa_config.get("prosodic_feature_names", []))
    config.score_token_id = score_token_id

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = WhisperForPronunciationAssessment.from_pretrained(
        peft_config.base_model_name_or_path,
        config=config,
        dtype=dtype,
    )
    base_model.resize_token_embeddings(len(processor.tokenizer))
    initialize_score_token_weights(base_model, score_token_id)
    model = PeftModel.from_pretrained(base_model, model_path)
    return model.to(device).eval(), processor, score_token_id, apa_config


def build_decoder_inputs(tokenizer, words, score_token_id, device):
    labels = list(tokenizer.prefix_tokens)
    start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if labels and labels[0] == start_token_id:
        labels = labels[1:]
    for index, word in enumerate(words):
        piece = word if index == 0 else " " + word
        piece_ids = tokenizer(piece, add_special_tokens=False).input_ids
        if not piece_ids:
            raise ValueError(f"Word produced no tokenizer pieces: {word!r}")
        labels.extend(piece_ids)
        labels.append(score_token_id)
    labels.append(tokenizer.eos_token_id)

    decoder_input_ids = torch.tensor(
        [[start_token_id] + labels[:-1]], dtype=torch.long, device=device
    )
    score_positions = decoder_input_ids.eq(score_token_id).nonzero()[:, 1]
    if score_positions.numel() != len(words):
        raise RuntimeError("Unexpected number of score-token positions")
    return (
        decoder_input_ids,
        torch.ones_like(decoder_input_ids),
        score_positions.unsqueeze(0),
        torch.ones((1, len(words)), dtype=torch.bool, device=device),
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")

    waveform, sampling_rate = load_audio(args.audio)
    if args.reference_text is None:
        raw_transcript, asr_compute_type = transcribe_with_whisperx(
            waveform, args, device
        )
        inference_mode = "reference-free"
    else:
        raw_transcript = args.reference_text
        asr_compute_type = None
        inference_mode = "reference-text"

    transcript = normalize_transcript(raw_transcript)
    words = transcript.split()
    if not words:
        raise ValueError("Transcript is empty after normalization")

    model_path = resolve_model_path(args.model, args.local_files_only)
    model, processor, score_token_id, apa_config = configure_model(
        model_path, device
    )
    acoustic = processor.feature_extractor(
        waveform,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )
    decoder_ids, decoder_mask, score_positions, word_mask = build_decoder_inputs(
        processor.tokenizer, words, score_token_id, device
    )
    if decoder_ids.size(1) > model.config.max_target_positions:
        raise ValueError(
            f"Transcript needs {decoder_ids.size(1)} decoder positions; "
            f"maximum is {model.config.max_target_positions}"
        )

    with torch.inference_mode():
        outputs = model(
            input_features=acoustic.input_features.to(
                device=device, dtype=next(model.parameters()).dtype
            ),
            attention_mask=acoustic.attention_mask.to(device),
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_mask,
            score_positions=score_positions,
            word_label_mask=word_mask,
            return_dict=True,
        )

    word_scores = outputs.word_logits[0].float().cpu().numpy()
    utterance_scores = outputs.utterance_logits[0].float().cpu().numpy()
    word_names = apa_config["word_label_names"]
    result = {
        "audio": str(Path(args.audio)),
        "mode": inference_mode,
        "transcript": transcript,
        "words": [
            {
                "word": word,
                **{
                    name: round(float(np.clip(value, 0.0, 10.0)), 4)
                    for name, value in zip(word_names, scores)
                },
            }
            for word, scores in zip(words, word_scores)
        ],
        "utterance": {
            name: round(float(np.clip(value, 0.0, 10.0)), 4)
            for name, value in zip(UTTERANCE_LABELS, utterance_scores)
        },
    }
    if inference_mode == "reference-free":
        result["asr"] = {
            "backend": "whisperx",
            "model": args.asr_model,
            "compute_type": asr_compute_type,
            "beam_size": args.asr_beam_size,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
