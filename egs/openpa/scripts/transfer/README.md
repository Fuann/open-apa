# Cross-domain transfer evaluation

These optional recipes evaluate the released OpenPA checkpoint without
fine-tuning on two out-of-domain corpora:

- TORGO: dysarthric speech, evaluated with the PathBench speaker-level protocol.
- YouTube: oral-cancer speech collected from public videos.

Protocol notes are documented in [TORGO.md](TORGO.md) and
[YOUTUBE.md](YOUTUBE.md).

The main `run.sh` remains limited to SpeechOcean762 and MultiPA. Transfer
results are written separately under `exp/openpa-pretrained/transfer/` and are
not included in the primary benchmark macro score.

## TORGO

```bash
bash scripts/transfer/evaluate_torgo.sh
```

## Oral-cancer YouTube

```bash
bash scripts/transfer/evaluate_youtube.sh
```

Each wrapper has two stages: Stage 0 downloads and validates the corpus; Stage
1 evaluates OpenPA. To evaluate an already prepared corpus or another
checkpoint:

```bash
bash scripts/transfer/evaluate_torgo.sh \
    --stage 1 \
    --checkpoint exp/EXPERIMENT/checkpoint-final
```

The downloaders remain separate because they are reusable, resumable data
preparation utilities and expose corpus-specific locations and validation.
They are also invoked automatically by Stage 0 of each evaluation wrapper:

```bash
bash scripts/transfer/download_torgo.sh
bash scripts/transfer/download_youtube.sh
```
