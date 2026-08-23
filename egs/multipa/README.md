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
reference, while SpeechOcean762 open uses Whisper `medium.en`. Both modes still
use Whisper `base.en` for the comparison ASR transcript. Close skips loading
Whisper `medium.en`.


The experiment outputs follow
`exp/{pretrained_model}/decode_{test_data}_{evaluation_mode}`.
For the default configuration, predictions and final metrics are written to:

```text
exp/model_assessment_val9_r1/decode_multipa_open/
├── test_mb.txt
└── result.txt
```

## Implementation results

Model: `model_assessment_val9_r1`

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
