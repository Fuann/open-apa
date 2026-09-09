"""Model-specific SpeechOcean762 transcripts; lazy faster-whisper generation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

from tqdm.auto import tqdm

ASR_SETTINGS = dict(backend='faster-whisper', device='cuda', compute_type='float16',
                    beam_size=5, temperature=0.0, condition_on_previous_text=False,
                    vad_filter=False, word_timestamps=True, random_seed=0,
                    faster_whisper_version='1.2.1', ctranslate2_version='4.8.1')


def transcript_path(directory, model):
    if not model or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_' for c in model):
        raise ValueError(f'Invalid Whisper model name: {model}')
    return Path(directory) / f'faster-whisper-{model}-float16-beam5.jsonl'


def read_transcripts(path, model):
    records = {}
    if not path.exists():
        return records
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = Path(record['audio_id']).stem
            if key in records or not isinstance(record.get('transcript'), str):
                raise ValueError(f'Duplicate ID or invalid transcript: {path}:{line_number}')
            settings = record.get('asr', {})
            for name, expected in {**ASR_SETTINGS, 'model': model}.items():
                if settings.get(name) != expected:
                    raise ValueError(f'ASR setting mismatch for {key}: {name} must be {expected!r}')
            if settings.get('language', 'en') != 'en' or settings.get('task', 'transcribe') != 'transcribe':
                raise ValueError(f'Expected English transcription: {key}')
            records[key] = record
    return records


def load_generator(model, models_dir, device, threads):
    """Use exactly the engine versions/precision recorded in the supplied JSONL."""
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError('Missing transcript: install faster-whisper==1.2.1 and ctranslate2==4.8.1 in the Conda environment') from exc
    for package, version in [('faster-whisper', '1.2.1'), ('ctranslate2', '4.8.1')]:
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Missing transcript requires {package}=={version} to match supplied transcripts')
    # ASR remains CUDA/float16 even when Hippo/feature extraction uses CPU.
    index = int(device.split(':')[1]) if device.startswith('cuda:') else 0
    if ctranslate2.get_cuda_device_count() <= index or 'float16' not in ctranslate2.get_supported_compute_types('cuda', index):
        raise RuntimeError('Missing transcript requires CUDA float16 to match supplied transcripts; select an available GPU with --gpu')
    ctranslate2.set_random_seed(0)
    return WhisperModel(model, device='cuda', device_index=index, compute_type='float16',
                        cpu_threads=threads, download_root=str(Path(models_dir) / 'faster-whisper'))


def generate_record(generator, path, model):
    segments, _ = generator.transcribe(str(path), language='en', task='transcribe',
        beam_size=5, temperature=0.0, condition_on_previous_text=False,
        vad_filter=False, word_timestamps=True)
    segments = list(segments)  # faster-whisper performs decoding lazily.
    words = [dict(word=word.word.strip(), start=float(word.start), end=float(word.end),
                  probability=float(word.probability)) for segment in segments for word in (segment.words or [])]
    return dict(audio_id=path.name, transcript=''.join(segment.text for segment in segments).strip(),
                words=words, audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                asr={**ASR_SETTINGS, 'model': model, 'language': 'en', 'task': 'transcribe'})


def resolve_transcripts(ids, wav_dir, directory, model, models_dir, device='cpu', threads=4):
    """Read existing rows (including empty transcripts); generate only absent IDs."""
    import fcntl
    path = transcript_path(directory, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent experiments from duplicating IDs while filling the same file.
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = read_transcripts(path, model)
        missing = [key for key in ids if key not in records]
        generated = []
        if missing:
            generator = load_generator(model, models_dir, device, threads)
            import os
            with path.open('a') as handle:
                # An existing last record need not have a trailing newline.
                if path.stat().st_size:
                    with path.open('rb') as reader:
                        reader.seek(-1, 2)
                        if reader.read(1) != b'\n':
                            handle.write('\n')
                for key in tqdm(missing, desc=f'faster-whisper {model}', unit='utt', dynamic_ncols=True):
                    audio = Path(wav_dir) / f'{key}.wav'
                    record = generate_record(generator, audio, model)
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    records[key] = record
                    generated.append(key)
            del generator
        selected = {key: records[key]['transcript'] for key in ids}
        provenance = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          model=model, settings=ASR_SETTINGS, reused_count=len(ids)-len(generated),
                          generated_ids=generated)
    print(f'Transcripts {model}: {len(ids)-len(generated)} reused, {len(generated)} generated', flush=True)
    return selected, provenance
