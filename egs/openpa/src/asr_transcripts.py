"""Reusable faster-whisper/WhisperX transcript loading and generation."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def faster_whisper_device(device):
    if device == "cpu":
        return "cpu", 0
    if device == "cuda":
        return "cuda", 0
    if device.startswith("cuda:"):
        return "cuda", int(device.split(":", 1)[1])
    return "cuda", int(device)


def transcript_fingerprint(record):
    stable = {
        "transcript": record["transcript"],
        "words": [
            {"word": word["word"], "start": word.get("start"), "end": word.get("end")}
            for word in record.get("words", [])
        ],
    }
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def transcript_result_key(transcript):
    source = str(transcript).split("[split=", 1)[0]
    name = Path(source).name
    for suffix in (".jsonl", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def write_transcript_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for audio_name in sorted(records):
            handle.write(json.dumps(records[audio_name], ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_transcript_record(row, require_timestamps=True):
    audio_name = row.get("audio_id") or row.get("audio")
    if isinstance(audio_name, dict):
        audio_name = Path(audio_name["path"]).name
    if not audio_name:
        raise ValueError("Transcript row requires audio_id or audio")

    raw_words = row.get("words") or []
    words = []
    for word in raw_words:
        text = str(word.get("word", word.get("text", ""))).strip()
        if not text:
            continue
        timestamp = word.get("timestamp") or [None, None]
        start = word.get("start", timestamp[0])
        end = word.get("end", timestamp[1])
        words.append({
            "word": text,
            "start": round(float(start), 3) if start is not None else None,
            "end": round(float(end), 3) if end is not None else None,
            "probability": float(word.get("probability", 1.0)),
        })

    transcript = str(row.get("transcript", row.get("hypothesis", ""))).strip()
    if not words and transcript and not require_timestamps:
        words = [
            {"word": word, "start": None, "end": None, "probability": 1.0}
            for word in transcript.split()
        ]
    if not words:
        requirement = " with start/end" if require_timestamps else ""
        raise ValueError(f"Transcript for {audio_name} has no words{requirement}")
    if require_timestamps and any(
        word["start"] is None or word["end"] is None for word in words
    ):
        raise ValueError(f"Transcript for {audio_name} has words without timestamps")

    record = dict(row)
    record["audio_id"] = Path(str(audio_name)).name
    record["transcript"] = transcript or " ".join(word["word"] for word in words)
    record["words"] = words
    record["fingerprint"] = transcript_fingerprint(record)
    return record


def load_transcript_records(source, split="train", require_timestamps=True):
    """Load stable transcript rows from local JSONL or a Hugging Face dataset."""
    source_path = Path(source)
    if source_path.exists():
        with source_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        if source_path.suffix in {".json", ".jsonl"}:
            raise FileNotFoundError(f"Transcript file not found: {source_path}")
        from datasets import load_dataset

        rows = load_dataset(source, split=split)
    records = {}
    for row in rows:
        record = normalize_transcript_record(row, require_timestamps)
        records[record["audio_id"]] = record
    return records


def transcribe_faster_whisper(model, audio, asr_config):
    segments, _ = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=asr_config["beam_size"],
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        repetition_penalty=asr_config.get("repetition_penalty", 1.0),
        no_repeat_ngram_size=asr_config.get("no_repeat_ngram_size", 0),
        max_new_tokens=asr_config.get("max_new_tokens"),
        word_timestamps=True,
    )
    segments = list(segments)
    words = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append({
                    "word": text,
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "probability": round(float(word.probability), 6),
                })
    record = {
        "transcript": " ".join(segment.text.strip() for segment in segments).strip(),
        "words": words,
    }
    record["fingerprint"] = transcript_fingerprint(record)
    return record


def transcribe_whisperx(model, audio, asr_config):
    """Transcribe one utterance with WhisperX's batched ASR pipeline."""
    asr_model = getattr(model, "asr_model", model)
    result = asr_model.transcribe(
        audio,
        batch_size=asr_config["batch_size"],
        language="en",
        task="transcribe",
        print_progress=False,
        verbose=False,
    )
    transcript = " ".join(
        segment["text"].strip() for segment in result["segments"]
    ).strip()
    if not transcript:
        transcript = "nospeech"
    if getattr(model, "align_model", None) is not None and transcript != "nospeech":
        import whisperx

        aligned = whisperx.align(
            result["segments"], model.align_model, model.align_metadata,
            audio, model.device, return_char_alignments=False,
        )
        words = [
            {
                "word": word["word"].strip(),
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "probability": round(float(word.get("score", 1.0)), 6),
            }
            for word in aligned.get("word_segments", [])
            if word.get("word", "").strip()
            and word.get("start") is not None and word.get("end") is not None
        ]
    else:
        words = [
            {
                "word": word,
                # MultiPA caches require timestamp fields. A zero-length
                # nospeech span is retained but receives no overlap target.
                "start": 0.0 if getattr(model, "align_model", None) is not None else None,
                "end": 0.0 if getattr(model, "align_model", None) is not None else None,
                "probability": 1.0,
            }
            for word in transcript.split()
        ]
    record = {"transcript": transcript, "words": words}
    record["fingerprint"] = transcript_fingerprint(record)
    return record


def transcribe_qwen3_asr(model, audio, asr_config):
    """Transcribe one utterance with native Transformers Qwen3-ASR."""
    inputs = model.processor.apply_transcription_request(
        audio=[audio], language="English"
    ).to(model.model.device, model.model.dtype)
    with torch.inference_mode():
        output_ids = model.model.generate(
            **inputs,
            max_new_tokens=asr_config["max_new_tokens"],
            do_sample=False,
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    transcript = model.processor.decode(
        generated_ids, return_format="transcription_only"
    )[0].strip()
    if not transcript:
        transcript = "nospeech"
    words = [
        {
            "word": word,
            "start": None,
            "end": None,
            "probability": 1.0,
        }
        for word in transcript.split()
    ]
    record = {"transcript": transcript, "words": words}
    record["fingerprint"] = transcript_fingerprint(record)
    return record


def load_whisperx_model(
    model_name, device, compute_type, beam_size, batch_size, local_files_only,
    word_timestamps=False,
):
    """Load WhisperX without a VAD model, treating each dataset row as one utterance."""
    import whisperx
    import ctranslate2

    from whisperx.vads.vad import Vad

    class FullAudioVad(Vad):
        def __init__(self):
            super().__init__(0.5)

        @staticmethod
        def preprocess_audio(audio):
            return audio

        def __call__(self, audio, **kwargs):
            duration = audio["waveform"].shape[-1] / audio["sample_rate"]
            return [SimpleNamespace(start=0.0, end=duration, speaker="UNKNOWN")]

    device_name, device_index = faster_whisper_device(device)
    model = whisperx.load_model(
        model_name,
        device=device_name,
        device_index=device_index,
        compute_type=compute_type,
        language="en",
        task="transcribe",
        asr_options={
            "beam_size": beam_size,
            "condition_on_previous_text": False,
        },
        vad_model=FullAudioVad(),
        vad_method=None,
        local_files_only=local_files_only,
    )
    if word_timestamps:
        align_model, align_metadata = whisperx.load_align_model(
            language_code="en", device=device_name,
        )
        model = SimpleNamespace(
            asr_model=model,
            align_model=align_model,
            align_metadata=align_metadata,
            device=device_name,
        )
    config = {
        "backend": "whisperx",
        "model": model_name,
        "device": device_name,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "batch_size": batch_size,
        "word_timestamps": word_timestamps,
        "condition_on_previous_text": False,
        "vad_method": "none",
        "random_seed": 0,
        "ctranslate2_version": ctranslate2.__version__,
    }
    return model, config


def load_faster_whisper_model(
    model_name, device, compute_type, beam_size, local_files_only
):
    from faster_whisper import WhisperModel
    import ctranslate2
    import faster_whisper

    device_name, device_index = faster_whisper_device(device)
    model = WhisperModel(
        model_name,
        device=device_name,
        device_index=device_index,
        compute_type=compute_type,
        local_files_only=local_files_only,
    )
    config = {
        "backend": "faster-whisper",
        "model": model_name,
        "device": device_name,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "word_timestamps": True,
        "random_seed": 0,
        "faster_whisper_version": faster_whisper.__version__,
        "ctranslate2_version": ctranslate2.__version__,
    }
    return model, config


def load_qwen3_asr_model(
    model_name, device, compute_type, local_files_only, max_new_tokens=256,
):
    """Load a Hugging Face Qwen3-ASR checkpoint for deterministic inference."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if compute_type not in {"float32", "float16", "bfloat16"}:
        raise ValueError(
            "Qwen3-ASR compute type must be float32, float16, or bfloat16"
        )
    device_name, device_index = faster_whisper_device(device)
    torch_device = torch.device(
        "cpu" if device_name == "cpu" else f"cuda:{device_index}"
    )
    if torch_device.type == "cpu" and compute_type != "float32":
        raise ValueError("Qwen3-ASR on CPU requires float32")
    dtype = getattr(torch, compute_type)
    processor = AutoProcessor.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    asr_model = AutoModelForMultimodalLM.from_pretrained(
        model_name, dtype=dtype, local_files_only=local_files_only
    ).to(torch_device).eval()
    model = SimpleNamespace(processor=processor, model=asr_model)
    config = {
        "backend": "qwen3-asr",
        "model": model_name,
        "device": str(torch_device),
        "compute_type": compute_type,
        "language": "English",
        "max_new_tokens": max_new_tokens,
        "word_timestamps": False,
        "random_seed": 0,
    }
    return model, config
