# MultiPA paper reproduction

This recipe runs the released MultiPA model on either the 50-file MultiPA pilot
set or the 2,500-file SpeechOcean762 test set. Data is downloaded from Hugging
Face (`yuwchen/multipa` or `mispeech/speechocean762`), and the official HuBERT
Base and RoBERTa Base backbones are downloaded automatically.

In this document, **MultiPA model** means the pronunciation-assessment method,
while **MultiPA pilot set** means its 50-file evaluation dataset. The command
line identifier for the pilot set remains `multipa`.


```text
egs/multipa/
├── run.sh
├── path.sh
├── requirements.txt
├── README.md
└── src/
    ├── test_open.py
    ├── evaluate_multipa.py
    ├── evaluate_speechocean762.py
    ├── prepare_open_transcripts.py
    ├── prepare_speechocean762.py
    └── supporting model/alignment modules
```

### Dependencies:
Install the Python dependencies with:

```bash
conda create -n multipa python=3.9
conda activate multipa
python -m pip install --upgrade "pip<24.1"
python -m pip install -r requirements.txt
```

The pip version constraint is required by `fairseq==0.12.2`: its OmegaConf
2.0.x dependency contains legacy metadata that pip 24.1 and newer reject. If a
previous installation failed with `ResolutionImpossible`, downgrade pip with
the command above and rerun the requirements installation.

`requirements.txt` also pins NumPy 1.26.4 because this legacy stack is not ABI
compatible with NumPy 2.x. If the environment was already installed with
NumPy 2.x and reports `numpy.dtype size changed`, repair the compiled packages
with:

```bash
python -m pip install --force-reinstall "numpy==1.26.4" pandas pyarrow
python -m pip check
```


### Usage:
`run.sh` automatically activates the `multipa` Conda environment. If `conda`
is not on `PATH`, provide its executable without editing `path.sh`:

```bash
MULTIPA_CONDA_EXE=/path/to/conda bash run.sh --test-data multipa
```

To use a differently named environment, set `MULTIPA_CONDA_ENV`:

```bash
MULTIPA_CONDA_ENV=my-environment bash run.sh --test-data multipa
```

Run **MultiPA open**. This is the only supported mode for the MultiPA pilot set
because it does not provide ground-truth transcripts:

```bash
bash run.sh --test-data multipa
```

By default, SpeechOcean762 runs **SpeechOcean762 close** followed by
**SpeechOcean762 open**:

```bash
bash run.sh --test-data speechocean762
```

Select only one mode when needed:

```bash
bash run.sh --test-data speechocean762 --response-mode close
bash run.sh --test-data speechocean762 --response-mode open
```

SpeechOcean762 close uses the dataset's ground-truth transcript as the
reference. MultiPA open and SpeechOcean762 open use fixed faster-whisper ASR
transcripts under the repository-level `references` directory. When
`path.sh` is sourced, it creates `references -> ../../references` in this
recipe directory. The default ASR transcript is `medium.en`; `large-v3` is
also selectable with `--whisper-model`. The required files are:

```text
references/multipa/transcripts/faster-whisper-<model>-float16-beam5.jsonl
references/speechocean762/test/transcripts/faster-whisper-<model>-float16-beam5.jsonl
```

If `references` already exists but is not that symlink, rename or remove it
before running `run.sh`. Both modes still use Whisper `base.en` for the
comparison ASR transcript.

The open word metric uses deterministic Levenshtein alignment and reports PCC
in three columns: match-only (`M`), substitution-only (`S`), and their union
(`M+S`). Insertions and deletions are excluded. Coverage, WER, and M/S/I/D counts are written beside the transcript as
`*.word-evaluation.json`. Ground-truth Charsiu alignments are no longer required
for open evaluation; the MultiPA model itself still uses its native alignment
features during inference.

With data and models prepared:

```bash
bash run.sh --stage 2 --test-data speechocean762 --response-mode open --whisper-model medium.en
```

`closed` is accepted as an alias for `close`.


The released assessment checkpoint is downloaded from `yuwchen/multipa` and
extracted under `pretrained-models/model_assessment_val9_r1`. For example, the
default SpeechOcean762 open configuration writes to:

```text
exp/model_assessment_val9_r1/decode_speechocean762_open_faster-whisper-medium.en-float16-beam5/
├── test_mb.txt
└── result.txt
```

Stage 3 reports the checkpoint's PCC directly in each decode directory's
`result.txt`; it does not aggregate mean or standard deviation across seeds.

## Implementation results

Model: `model_assessment_val9_r1`

The reported open results use Whisper `medium.en`.
All values are Pearson correlation coefficients (PCC) from the corresponding
`exp/model_assessment_val9_r1/decode_*/result.txt` files.

### SpeechOcean762 open/close

| Mode | Wrd-Acc | Wrd-Stress | Wrd-Total | Utt-Acc | Utt-Fluency | Utt-Prosody | Utt-Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Close | 0.5124 | 0.1866 | 0.5237 | 0.7310 | 0.7985 | 0.7952 | 0.7498 |
| Open | 0.4097 | 0.2119 | 0.4187 | 0.6960 | 0.7639 | 0.7614 | 0.7139 |

### MultiPA open

| Wrd-Score | Utt-Acc | Utt-Fluency | Utt-Prosody |
| ---: | ---: | ---: | ---: |
| 0.3703 | 0.6122 | 0.6567 | 0.4739 |

The MultiPA annotations do not contain ground-truth utterance total, word
stress, or word total scores, so those metrics are not reported.
