# PathBench TORGO dysarthric-speech transfer evaluation

This adapter evaluates OpenPA on TORGO using the PathBench pathological-speech
protocol. It scores each recording, averages predictions per dysarthric
speaker, and reports speaker-level Pearson correlation against the Frenchay
Dysarthria Assessment (FDA) scores supplied by PathBench.

The PathBench-compatible headline targets are:

- isolated-word recordings: OpenPA `word_accuracy`;
- sentence recordings: OpenPA `utterance_accuracy`.

These are the wrapper defaults. Select another saved OpenPA head with
`--word-output` or `--utterance-output`.

The recording- and speaker-level CSV files also retain the other OpenPA word
and utterance scores for analysis. These outputs are zero-shot transfer scores,
not calibrated clinical diagnoses.

## PathBench protocol

The implementation follows the pinned PathBench manifests and conventions:

- `wav_arrayMic` is used for every recording;
- word and sentence recordings are evaluated separately;
- `balanced` corresponds to Matched Content (MC);
- `unbalanced` corresponds to Extended (EX);
- recording predictions are averaged within each speaker;
- PCC is computed over the eight dysarthric speakers only;
- controls are not assigned an artificial normal FDA score;
- word and sentence tasks use their respective PathBench FDA targets.

The manifests are pinned to PathBench commit
`ab85a8046851d608495dd61b9f90718a03e59184` and cached under
`data/torgo/pathbench/`.

## Download

```bash
bash scripts/transfer/download_torgo.sh
```

The downloader defaults to `--pathbench-only true` and therefore downloads
only `F.tar.bz2` and `M.tar.bz2`, which contain the eight dysarthric speakers
used for PathBench PCC. Control archives and supplemental files are omitted.

The downloader supports `--stage` and `--stop-stage`:

- stage 0 downloads the selected official TORGO archives;
- stage 1 extracts the archives using completion markers;
- stage 2 validates the eight required speakers and reports audio/prompt counts.

To download and validate the complete 15-speaker release instead:

```bash
bash scripts/transfer/download_torgo.sh --pathbench-only false
```

TORGO is distributed for academic, non-profit use. Publications using it must
follow the citation requirements on the official TORGO website.

## Reference-text evaluation

Reference text is the primary PathBench-compatible condition. OpenPA receives
the intended TORGO prompt as decoder input.

```bash
bash scripts/transfer/evaluate_torgo.sh --stage 1 \
    --checkpoint exp/EXPERIMENT/checkpoint-final \
    --conditions reference_text \
    --balancing both \
    --word-output accuracy \
    --utterance-output accuracy
```

Use `--balancing balanced` for MC only or `--balancing unbalanced` for EX only.

## Reference-free evaluation

Reference-free evaluation is an additional OpenPA robustness experiment, not
the original PathBench Reference-Text setting. It replaces the prompt with a
fixed ASR transcript while retaining the same speaker-level aggregation.

```bash
bash scripts/transfer/evaluate_torgo.sh --stage 1 \
    --checkpoint exp/EXPERIMENT/checkpoint-final \
    --conditions reference_free \
    --balancing both \
    --asr-model medium.en \
    --asr-backend whisperx
```

Use `--conditions both` to run both transcript conditions. ASR hypotheses are
cached incrementally under `data/torgo/transcripts/`; backend, model, compute
type, and beam size are included in the run ID.

## Outputs

Outputs default to `exp/openpa-pretrained/transfer/torgo/` and can be changed
with `--output-dir`:

- `reference-text/results.json` contains MC/EX word and sentence PCC;
- `reference-free/results.json` contains results keyed by ASR run ID;
- recording-level CSV files preserve every OpenPA prediction;
- speaker-level CSV files contain the averaged scores and FDA ground truth.

Because each PCC contains only eight dysarthric-speaker observations, results
should be presented as exploratory cross-domain transfer evidence rather than
per-utterance clinical validation.
