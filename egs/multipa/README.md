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
    ├── summarize_pcc.py
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
bash run.sh --test-data speechocean762 --evaluation-mode close
bash run.sh --test-data speechocean762 --evaluation-mode open
```

SpeechOcean762 close uses the dataset's ground-truth transcript as the
reference. SpeechOcean762 open uses a fixed faster-whisper transcript from
`data/speechocean762/transcript/test`. The default is `medium.en`; `large-v3`
is also supported. If the selected transcript file is absent or incomplete,
stage 2 generates the missing rows once with faster-whisper using FP16, beam
size 5, and temperature 0. Both modes still use Whisper `base.en` for the
comparison ASR transcript.

With data and models prepared:

```bash
bash run.sh --stage 2 --test-data speechocean762 --evaluation-mode open --whisper-model medium.en
```

`--response-mode` is an alias for `--evaluation-mode`; `closed` is accepted as
an alias for `close`.


The five assessment checkpoints are downloaded from the private model repo
`fuann/multipa-model` and kept under `pretrained-models/multipa-model/{0..4}`.
Predictions are separated by seed. For example, the default SpeechOcean762 open
configuration writes to:

```text
exp/multipa-model/0/decode_speechocean762_open_faster-whisper-medium.en-float16-beam5/
├── test_mb.txt
├── result.txt
└── pcc.json
```

Stage 3 evaluates all five seeds and writes PCC mean and sample standard
deviation to `exp/multipa-model/evaluate_*/result_mean_std.txt`.

## Implementation results

Model: `model_assessment_val9_r1`

The reported open results use Whisper `medium.en`, not the new `large-v3` default.
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
