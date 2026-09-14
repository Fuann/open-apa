# HiPPO SpeechOcean762 evaluation

Extract pronunciation features from audio, run the released HiPPO checkpoint,
and evaluate utterance and word scores in open or closed response mode.

## Setup

Create the dedicated environment and install the dependencies with:

```bash
conda create -n hippo python=3.9
conda activate hippo
python -m pip install -r requirements.txt
```

`path.sh` activates `hippo` automatically. It discovers Conda from `PATH`,
`CONDA_EXE`, or the usual `~/miniconda3` and `~/miniforge3` locations. Override
the executable with `HIPPO_CONDA_EXE` or the environment name with
`HIPPO_CONDA_ENV`. Model caches remain inside this recipe, while Hugging Face
authentication uses the token created by the standard `hf auth login` command.

Model files and feature-model caches are stored under `pretrained-models/`.
Stage 0 downloads the released checkpoint from [fuann/hippo](https://huggingface.co/fuann/hippo)
into `pretrained-models/hippo/best_audio_model.pth`.
It also downloads the original 40-token CTC-GOP model and matching processor
from [fuann/ctc-gop](https://huggingface.co/fuann/ctc-gop) into
`pretrained-models/ctc-gop/checkpoint-8000/` and
`pretrained-models/ctc-gop/processor_config_gop/`.
The CTC-GOP files originally come from
[CTC-based-GOP](https://github.com/frank613/CTC-based-GOP/tree/main/is24/models).
Generic wav2vec2 weights are not interchangeable.

The timestamp word evaluator also requires the sibling `../multipa` recipe,
its Charsiu dependencies, cached Charsiu models under `~/.cache/huggingface/`,
and SpeechOcean762 reference alignments under
`../multipa/data/speechocean762/gt-alignments/`.

## Running

```bash
# Open response: medium.en followed by large-v3.
bash run.sh

# One ASR model, with GPU extraction and inference.
bash run.sh --whisper-model large-v3 --gpu 0

# Use the provided WhisperX medium.en transcript.
bash run.sh --asr-backend whisperx --whisper-model medium.en --gpu 0

# Closed response uses ground-truth text.
bash run.sh --response-mode closed --gpu 0

# Repeat inference and evaluation with existing features.
bash run.sh --stage 2 --response-mode closed \
  --exp-dir exp/hippo/decode_speechocean762_closed
bash run.sh --stage 2 --whisper-model large-v3 \
  --exp-dir exp/hippo/decode_speechocean762_open_whisperx_large-v3

# CPU smoke run in a separate directory.
bash run.sh --limit 2 --whisper-model medium.en \
  --exp-dir exp/hippo/smoke
```

Extraction and inference default to `cuda:0` (GPU 0). `--gpu N` selects another
CUDA device; `--device cpu` remains available. `--device` is honored by
inference without `DataParallel`. See
`bash run.sh --help` for all options.

| Stage | Work |
| --- | --- |
| 0 | Download HiPPO, CTC-GOP and feature models; prepare raw test audio/labels if absent |
| 1 | Load provided transcripts and extract GOP, SSL, language features and labels |
| 2 | Check feature provenance, audit inputs, and run the released checkpoint |
| 3 | Prepare word alignment targets and write evaluation reports |

Default experiment directories are:

- `exp/hippo/decode_speechocean762_closed`
- `exp/hippo/decode_speechocean762_open_whisperx_medium.en`
- `exp/hippo/decode_speechocean762_open_whisperx_large-v3`

A two-model run treats `--exp-dir` as a parent for the two named open directories.
Stage 1 recomputes features and refuses to overwrite an existing feature directory.
Failed extractions remain in `features.incomplete-*`; only successful extractions
publish `features/`. Resume stages 2/3 with the original ASR, response mode, scores
and feature configuration. After changing the model, resume from stage 2.

## Transcripts and normalization

Open mode uses the model-specific JSONL files in the repository-level shared
`references` directory, matched by `audio_id`. When `path.sh` is sourced, it
creates `references -> ../../references` in this recipe directory:

- `whisperx-medium.en-float16-beam5.jsonl`
- `whisperx-large-v3-float16-beam5.jsonl`

WhisperX is the default evaluation backend. With no model specified, the recipe
runs both versioned references. `--asr-backend faster-whisper` remains available
for a compatible local transcript, but faster-whisper transcripts are not
versioned in this repository.

The WhisperX references use CTranslate2 4.8.1, CUDA float16, beam size 5,
batch size 16, seed 0, `condition_on_previous_text=False`, and no word
timestamps. Their metadata is validated before extraction.
`run.sh` reports an error when the selected fixed transcript is unavailable;
it does not generate or modify shared references. Settings and transcript
provenance are recorded in `features/features.json`.

Text normalization follows MultiPA: remove punctuation except apostrophes,
lowercase, then convert numeric tokens. All-numeric sentences are spelled digit
by digit; numeric tokens in mixed sentences use cardinal numbers. G2P also strips
accents and normalizes tokens. Older features without `text_normalization` retain
their original normalization; reevaluation alone does not change their inputs.

SpeechOcean762 open word PCC uses deterministic Levenshtein alignment. Results
are reported in three columns: match-only (`M`), substitution-only (`S`), and
their union (`M+S`). Insertions and deletions are excluded. Coverage, WER, and M/S/I/D counts are
written to `word_evaluation.statistics.json` in the experiment directory.
Stage 2's printed word PCC and `result.csv` use this same protocol as stage 3's
`word_open`: predictions are capped at 10, and failed utterances are excluded.
Stages 2 and 3 report PCC directly for the released checkpoint; no cross-seed
mean or standard deviation is computed. Targets are prepared before stage 2,
including runs with `--stop-stage 2`. Direct open-response calls to `inference.py` require
`--word-evaluation` and the matching `--scores` (if using custom scores).
The saved native prediction/target arrays and `word_native` diagnostic retain
the original labels. This evaluation does not use Charsiu or ground-truth word
timestamps.

## Checkpoint and failure conventions

The released checkpoint requires a phone-ID offset of `+1` before 42-class one-hot
encoding. `src/models/hippo.py` implements this convention. Registered parameters
needed for strict checkpoint loading are retained, including auxiliary layers
unused by inference. The run records the offset, checkpoint hash, model source
hash and feature fingerprint in `feature_provenance.json`.

Open transcripts with no text, no phones, or more than `--max-phones` phones use
fallback predictions. Their IDs/reasons are retained in `features/failures.json`,
and padded rows bypass all feature encoders and HiPPO inference. No transcript is
silently truncated. The default padding is 65 phones for open and 50 for closed;
changing it can affect predictions. Closed overflow and invalid input features
remain errors.

Fallback values on the 0–10 scale are:

- Utterance accuracy/fluency/prosodic/total: **1/0/0/0**.
- Reference-word accuracy/stress/total: **0/5/1**.

Completeness has no fallback and is excluded from the main report. Failed
reference timestamp alignment is recorded separately in `evaluation_failures`;
it uses fallback in the main utterance and timestamp-word reports, while the
sequence-word report retains available model predictions.

## Reading results

`result.txt` and `metrics.json.summary` show PCC for the released checkpoint.
`--limit` is for smoke tests; full benchmark evaluation covers 2,500 test
utterances.

| Group | Evaluation rule |
| --- | --- |
| `utterance_full_test_fallback` | All test utterances, including fallback |
| `word_hippo_sequence` | Sequence alignment: exclude deletions; insertion/substitution targets are zero |
| `word_multipa` | Timestamp alignment: mean overlapping reference scores, or 0/5/1 without overlap |

Both word groups cap model predictions at 10 and include ASR fallback. Charsiu
may omit unaligned words, so their sample counts can differ. Timestamp targets
and selected word indices are cached in `word_evaluation.json` with input
fingerprints. Native uncapped, successful-only metrics remain in `metrics.json`
for diagnostics. Error metrics use 0–10 for utterances/words and 0–2 for phones.

| File | Content |
| --- | --- |
| `features/et.csv` | Input IDs, transcripts, audio paths and hashes |
| `features/features.json` | Feature provenance and fingerprints |
| `audit.json` | Input coverage, labels and array inventory |
| `result.txt` | Main PCC tables |
| `metrics.json` | Checkpoint metrics and report metadata |
| `predictions.jsonl` | Per-utterance/word scores and fallback reasons |
| `preds/` | Raw model predictions and targets |

## Source layout and tests

```text
run.sh / path.sh             Pipeline and environment setup
inference.py                Checkpoint inference
src/models/hippo.py         HiPPO model
src/feats_extract/          Feature extraction, transcripts and pure helpers
src/prepare_data.py         Raw SpeechOcean762 preparation
src/prepare_word_evaluation.py  MultiPA timestamp targets
src/evaluate_speechocean762.py  Auditing and metrics
tests/                      Regression tests
```

```bash
. ./path.sh
python -m unittest discover -s tests -v
```

Downloaded data, results, model caches and the local `.deps` dependency directory
are ignored by Git. Third-party package license notices remain with their packages.
