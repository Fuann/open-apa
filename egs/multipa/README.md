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

### MultiPA

All 50 predictions are valid and matched with annotations. Utterance-level
human scores are averaged across five annotators.

| Level | Metric | N | PCC | SRCC |
| --- | --- | ---: | ---: | ---: |
| Utterance | Accuracy | 50 | 0.6095 | 0.6645 |
| Utterance | Fluency | 50 | 0.6448 | 0.6572 |
| Utterance | Prosody | 50 | 0.4626 | 0.5141 |
| Word | Accuracy | 1,439 | 0.3713 | 0.3704 |

The MultiPA annotations do not contain ground-truth utterance total, word
stress, or word total scores, so those metrics are not reported.

### SpeechOcean762

All 2,500 predictions are included; six invalid predictions use the fallback
scores defined by the original evaluation protocol. Word labels are mapped by
ground-truth timestamp overlap.

| Level | Metric | N | PCC | SRCC |
| --- | --- | ---: | ---: | ---: |
| Utterance | Accuracy | 2,500 | 0.7050 | 0.6875 |
| Utterance | Fluency | 2,500 | 0.7750 | 0.7601 |
| Utterance | Prosody | 2,500 | 0.7727 | 0.7593 |
| Utterance | Total | 2,500 | 0.7300 | 0.7155 |
| Word | Accuracy | 16,139 | 0.4133 | 0.3539 |
| Word | Stress | 16,139 | 0.2622 | 0.1605 |
| Word | Total | 16,139 | 0.4237 | 0.3740 |
