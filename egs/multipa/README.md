# MultiPA paper reproduction

This recipe runs the released MultiPA model on either the 50-file MultiPA pilot
set or the 2,500-file SpeechOcean762 test set. Data is downloaded from Hugging
Face (`yuwchen/multipa` or `mispeech/speechocean762`), and the official HuBERT
Base and RoBERTa Base backbones are downloaded automatically.


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
python -m pip install -r requirements.txt
```


### Usage:
Run MultiPA open-response inference:

```bash
bash run.sh --test-data multipa
```

Run SpeechOcean762 open-response inference:

```bash
bash run.sh --test-data speechocean762
```

The experiment outputs follow `exp/{pretrained_model}/decode_{test_data}`.
For the default configuration, predictions and final metrics are written to:

```text
exp/model_assessment_val9_r1/decode_multipa/
├── test_mb.txt
└── result.txt
```

## Implementation results

Model: `model_assessment_val9_r1`

### SpeechOcean762

All 2,500 predictions are included; six invalid predictions use the fallback
scores defined by the original evaluation protocol. Word labels are mapped by
ground-truth timestamp overlap.

#### Utterance-level Score (PCC)

| Accuracy | Fluency | Prosody | Total |
| :---: | :---: | :---: | :---: |
| 0.7050 | 0.7750 | 0.7727 | 0.7300 |

#### Word-level Score (PCC)

| Accuracy | Stress | Total |
| :---: | :---: | :---: |
| 0.4133 | 0.2622 | 0.4237 |


### MultiPA

All 50 predictions are valid and matched with annotations. Utterance-level
human scores are averaged across five annotators.

#### Utterance-level Score (PCC)

| Accuracy | Fluency | Prosody |
| :---: | :---: | :---: |
| 0.6095 | 0.6448 | 0.4626 |

#### Word-level Score (PCC)

| Accuracy |
| :---: |
| 0.3713 |

The MultiPA annotations do not contain ground-truth utterance total, word
stress, or word total scores, so those metrics are not reported.