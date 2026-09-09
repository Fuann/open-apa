#!/usr/bin/env python3
"""Download raw SpeechOcean762 test audio/labels into this recipe."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parents[1] / 'data/speechocean762'
    parquet = hf_hub_download('mispeech/speechocean762', 'data/test-00000-of-00001.parquet',
                             repo_type='dataset', local_dir=str(root))
    wav = root / 'wav'
    wav.mkdir(parents=True, exist_ok=True)
    scores = {}
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            audio = row.pop('audio')
            filename = Path(audio['path']).with_suffix('.wav').name
            key = Path(filename).stem
            if key in scores or audio.get('bytes') is None:
                raise ValueError(f'Duplicate ID or missing audio: {key}')
            (wav / filename).write_bytes(audio['bytes'])
            scores[key] = row
    if len(scores) != 2500:
        raise ValueError(f'Expected 2500 test utterances, found {len(scores)}')
    (root / 'scores.json').write_text(json.dumps(scores) + '\n')
    print(f'Prepared {len(scores)} raw test utterances in {root}')


if __name__ == '__main__':
    main()
