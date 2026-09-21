"""Read and validate fixed SpeechOcean762 ASR transcripts."""
import hashlib
import json
from pathlib import Path

ASR_SETTINGS = dict(backend='faster-whisper', device='cuda', compute_type='float16',
                    beam_size=5, temperature=0.0, condition_on_previous_text=False,
                    vad_filter=False, word_timestamps=True, random_seed=0,
                    faster_whisper_version='1.2.1', ctranslate2_version='4.8.1')
WHISPERX_SETTINGS = dict(backend='whisperx', device='cuda', compute_type='float16',
                         beam_size=5, batch_size=16,
                         condition_on_previous_text=False, vad_method='none',
                         random_seed=0, ctranslate2_version='4.8.1')


def transcript_path(directory, model, backend='faster-whisper'):
    if not model or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_' for c in model):
        raise ValueError(f'Invalid Whisper model name: {model}')
    if backend not in ('faster-whisper', 'whisperx'):
        raise ValueError(f'Invalid ASR backend: {backend}')
    return Path(directory) / f'{backend}-{model}-float16-beam5.jsonl'


def read_transcripts(path, model, backend='faster-whisper'):
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
            expected_settings = ASR_SETTINGS if backend == 'faster-whisper' else WHISPERX_SETTINGS
            for name, expected in {**expected_settings, 'model': model}.items():
                if settings.get(name) != expected:
                    raise ValueError(f'ASR setting mismatch for {key}: {name} must be {expected!r}')
            if backend == 'whisperx' and not isinstance(settings.get('word_timestamps'), bool):
                raise ValueError(f'ASR setting mismatch for {key}: word_timestamps must be boolean')
            if settings.get('language', 'en') != 'en' or settings.get('task', 'transcribe') != 'transcribe':
                raise ValueError(f'Expected English transcription: {key}')
            records[key] = record
    return records


def resolve_transcripts(ids, wav_dir, directory, model, models_dir, device='cpu',
                        threads=4, backend='faster-whisper'):
    """Read a complete fixed transcript without modifying shared references."""
    path = transcript_path(directory, model, backend)
    if not path.is_file():
        raise FileNotFoundError(f'Missing fixed ASR transcripts: {path}')
    records = read_transcripts(path, model, backend)
    missing = [key for key in ids if key not in records]
    if missing:
        raise ValueError(f'Fixed ASR transcript is missing {len(missing)} IDs; first: {missing[0]}')
    selected = {key: records[key]['transcript'] for key in ids}
    settings = ASR_SETTINGS if backend == 'faster-whisper' else WHISPERX_SETTINGS
    provenance = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      backend=backend, model=model, settings=settings,
                      reused_count=len(ids), generated_ids=[])
    print(f'Transcripts {backend}/{model}: {len(ids)} reused', flush=True)
    return selected, provenance
