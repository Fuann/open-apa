# OpenPA

This recipe provides single-audio inference and reproducible evaluation of the
released OpenPA model on SpeechOcean762 and MultiPA.

## Setup

Create the environment and install the dependencies:

```bash
conda create -n openpa python=3.10
conda activate openpa
python -m pip install -r requirements.txt
```

All workflow scripts source `path.sh` automatically. By default it activates
the `openpa` Conda environment and keeps downloaded models and generated caches
inside the recipe. The locations can be overridden when needed:

```bash
export OPENPA_CONDA_ENV=openpa
export OPENPA_MODELS_DIR=/path/to/models
export OPENPA_CACHE_DIR=/path/to/cache
```

## Usage

### A. Download the pretrained model

```bash
hf download fuann/openpa --local-dir pretrained-models/openpa
```

### B. Single-wav inference

```bash
# Reference-text inference.
python inference.py \
    --model pretrained-models/openpa \
    --audio ../../audios/sample.wav \
    --reference-text "I always thought he was a good person"

# Reference-free inference with WhisperX medium.en.
python inference.py \
    --model pretrained-models/openpa \
    --audio ../../audios/sample.wav \
    --asr-model medium.en
```

### C. Benchmark evaluation

```bash
# Reproduce SpeechOcean762 closed/open and MultiPA open results using
# the versioned WhisperX medium.en transcripts under ../../references.
bash run.sh \
    --test-sets speechocean762,multipa \
    --speechocean-conditions close,open \
    --eval-asr-backend whisperx \
    --asr-model medium.en \
    --gpu 0
```
Results are written to:

```bash
exp/openpa-pretrained/evaluation/
├── speechocean762/
│   ├── close/
│   └── open/
└── multipa/
```
