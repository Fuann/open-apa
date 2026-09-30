# PathBench oral-cancer YouTube transfer evaluation

This adapter evaluates the 98-utterance oral-cancer YouTube release using the
PathBench protocol. It averages each OpenPA utterance head per speaker, then
reports speaker-level Pearson correlation against `spk2score` for all four
utterance heads: `accuracy`, `fluency`, `prosodic`, and `total`.

## Download

```bash
bash scripts/transfer/download_youtube.sh
```

The downloader supports `--stage` and `--stop-stage`:

- stage 0 downloads `youtube_utterances.zip` from the official Zenodo record;
- stage 1 extracts it without performing an expensive archive verification;
- stage 2 summarizes the extracted WAV files.

## Reference-text evaluation

```bash
bash scripts/transfer/evaluate_youtube.sh --stage 1 \
  --checkpoint exp/EXPERIMENT/checkpoint-final \
  --conditions reference_text
```

## Reference-free evaluation

```bash
bash scripts/transfer/evaluate_youtube.sh --stage 1 \
  --checkpoint exp/EXPERIMENT/checkpoint-final \
  --conditions reference_free \
  --asr-model medium.en \
  --asr-backend faster-whisper
```

Use `--conditions both` to run both protocols. ASR hypotheses are
cached incrementally under `data/youtube/transcripts/`; a stable run ID keeps
results from different ASR models, backends, compute types, and beam sizes
separate.

Outputs default to `exp/openpa-pretrained/transfer/youtube/` and can be changed
with `--output-dir`:

- `reference-text/results.json` contains the reference-text metrics;
- `reference-free/results.json` contains metrics keyed by ASR run ID;
- recording- and speaker-level CSV files preserve every utterance-head score.

PathBench manifests are pinned to commit
`ab85a8046851d608495dd61b9f90718a03e59184` and cached under
`data/youtube/pathbench/`.
