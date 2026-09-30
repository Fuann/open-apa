#!/usr/bin/env bash

# Evaluate the released OpenPA checkpoint.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_dir"
. "$repo_dir/path.sh"

stage=0
stop_stage=2
model_repo=fuann/openpa
checkpoint=pretrained-models/openpa
exp_dir=exp/openpa-pretrained
test_sets=speechocean762,multipa
gpu=0
seed=168
full_determinism=true
num_workers=8
per_device_eval_batch_size=64
eval_fp16=true

dataset=mispeech/speechocean762
eval_split=test
asr_model=medium.en
eval_asr_backend=whisperx
asr_device=cuda:0
asr_compute_type=float16
asr_beam_size=5
asr_batch_size=16

speechocean_conditions=close,open
speechocean_cache_dir=data/speechocean762
speechocean_transcript_source=
speechocean_transcript_split=test
speechocean_word_alignment=levenshtein
speechocean_charsiu_model=charsiu/en_w2v2_fc_10ms
charsiu_hf_cache_dir=.cache/huggingface/hub
multipa_cache_dir=data/multipa
multipa_word_output=total
multipa_transcript_source=
multipa_transcript_split=test
multipa_max_samples=

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: bash run.sh [options]

Evaluate the released OpenPA model on the public benchmark datasets.

Runtime:
    --stage N                       first stage (default: $stage)
    --stop-stage N                  last stage (default: $stop_stage)
    --model-repo REPO               Hugging Face repository (default: $model_repo)
    --checkpoint DIR                local checkpoint (default: $checkpoint)
    --exp-dir DIR                   evaluation output (default: $exp_dir)
    --test-sets SETS                speechocean762, multipa, or both (default: $test_sets)
    --gpu ID                        CUDA device ID (default: $gpu)
    --seed N                        deterministic seed (default: $seed)
    --full-determinism BOOL         deterministic evaluation (default: $full_determinism)
    --num-workers N                 DataLoader workers (default: $num_workers)
    --per-device-eval-batch-size N  evaluation batch size (default: $per_device_eval_batch_size)
    --eval-fp16 BOOL                FP16 inference (default: $eval_fp16)

Open-response ASR:
    --asr-model MODEL               transcript model (default: $asr_model)
    --eval-asr-backend BACKEND      whisperx, faster-whisper, or qwen3-asr
    --asr-device DEVICE             ASR device (default: $asr_device)
    --asr-compute-type TYPE         ASR compute type (default: $asr_compute_type)
    --asr-beam-size N               beam size (default: $asr_beam_size)
    --asr-batch-size N              ASR batch size (default: $asr_batch_size)

Evaluation:
    --speechocean-conditions MODES  close, open, or close,open
    --speechocean-transcript-source PATH
    --speechocean-word-alignment MODE  levenshtein or charsiu
    --multipa-transcript-source PATH
    --multipa-word-output LABEL     accuracy or total
    --multipa-max-samples N         optional smoke-test limit

Stages:
    0  Download the released model from Hugging Face
    1  Verify the checkpoint and create the output directory
    2  Run inference and calculate benchmark metrics

The private model repository requires an authenticated Hugging Face account.
EOF
}

help_message="$(usage)"
. ./parse_options.sh || exit 1

if [[ $# -ne 0 ]]; then
    echo -e "${RED}Unexpected positional arguments: $*${NC}" >&2
    exit 2
fi
if ! [[ "$stage" =~ ^[0-9]+$ && "$stop_stage" =~ ^[0-9]+$ ]] || \
        (( stage > stop_stage || stop_stage > 2 )); then
    echo -e "${RED}Stages must satisfy 0 <= stage <= stop-stage <= 2.${NC}" >&2
    exit 2
fi
if [[ "$full_determinism" != true && "$full_determinism" != false ]]; then
    echo -e "${RED}--full-determinism must be true or false.${NC}" >&2
    exit 2
fi
if [[ "$eval_fp16" != true && "$eval_fp16" != false ]]; then
    echo -e "${RED}--eval-fp16 must be true or false.${NC}" >&2
    exit 2
fi
case "$eval_asr_backend" in
    whisperx|faster-whisper|qwen3-asr) ;;
    *) echo -e "${RED}Unsupported --eval-asr-backend: $eval_asr_backend${NC}" >&2; exit 2 ;;
esac
case "$multipa_word_output" in
    accuracy|total) ;;
    *) echo -e "${RED}--multipa-word-output must be accuracy or total.${NC}" >&2; exit 2 ;;
esac

if [[ -z $speechocean_transcript_source ]]; then
    speechocean_transcript_source=$(openpa_reference_transcript \
        speechocean762 "$eval_asr_backend" "$asr_model" \
        "$asr_compute_type" "$asr_beam_size" || true)
fi
if [[ -z $multipa_transcript_source ]]; then
    multipa_transcript_source=$(openpa_reference_transcript \
        multipa "$eval_asr_backend" "$asr_model" \
        "$asr_compute_type" "$asr_beam_size" || true)
fi

determinism_args=(--no-full-determinism)
determinism_env=()
if [[ "$full_determinism" == true ]]; then
    determinism_args=(--full-determinism)
    determinism_env=("PYTHONHASHSEED=$seed" "CUBLAS_WORKSPACE_CONFIG=:4096:8")
fi

if (( stage <= 0 && stop_stage >= 0 )); then
    echo -e "${GREEN}Stage 0: Download released OpenPA model${NC}"
    if [[ -f "$checkpoint/apa_config.json" ]] && \
            [[ -f "$checkpoint/config.json" || -f "$checkpoint/adapter_config.json" ]]; then
        echo -e "${YELLOW}Using existing checkpoint: $checkpoint${NC}"
    else
        command -v hf >/dev/null 2>&1 || {
            echo -e "${RED}Missing Hugging Face CLI command: hf${NC}" >&2
            exit 1
        }
        mkdir -p "$checkpoint"
        hf download "$model_repo" --local-dir "$checkpoint"
    fi
    echo
fi

if (( stage <= 1 && stop_stage >= 1 )); then
    echo -e "${GREEN}Stage 1: Verify benchmark configuration${NC}"
    [[ -f "$checkpoint/config.json" || -f "$checkpoint/adapter_config.json" ]] || {
        echo -e "${RED}Checkpoint is incomplete: $checkpoint${NC}" >&2
        echo -e "${YELLOW}Run Stage 0 or pass --checkpoint DIR.${NC}" >&2
        exit 1
    }
    mkdir -p "$exp_dir/evaluation"
    echo "Checkpoint: $checkpoint"
    echo "Test sets: $test_sets"
    echo "SpeechOcean762 transcript: ${speechocean_transcript_source:-generated}"
    echo "MultiPA transcript: ${multipa_transcript_source:-generated}"
    echo "Output: $exp_dir/evaluation"
    echo
fi

if (( stage <= 2 && stop_stage >= 2 )); then
    echo -e "${GREEN}Stage 2: Evaluate released OpenPA model${NC}"
    precision_args=(--no-fp16)
    [[ "$eval_fp16" != true ]] || precision_args=(--fp16)
    multipa_limit_args=()
    [[ -z "$multipa_max_samples" ]] || \
        multipa_limit_args=(--multipa-max-samples "$multipa_max_samples")

    env "${determinism_env[@]}" CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
        python src/evaluate.py \
        --checkpoint "$checkpoint" \
        --test-sets "$test_sets" \
        --dataset "$dataset" \
        --split "$eval_split" \
        --output-dir "$exp_dir/evaluation/speechocean762" \
        --multipa-output-dir "$exp_dir/evaluation/multipa" \
        --seed "$seed" \
        "${determinism_args[@]}" \
        --per-device-eval-batch-size "$per_device_eval_batch_size" \
        --num-workers "$num_workers" \
        --speechocean-conditions "$speechocean_conditions" \
        --speechocean-cache-dir "$speechocean_cache_dir" \
        --speechocean-transcript-source "$speechocean_transcript_source" \
        --speechocean-transcript-split "$speechocean_transcript_split" \
        --speechocean-asr-model "$asr_model" \
        --speechocean-asr-backend "$eval_asr_backend" \
        --speechocean-asr-batch-size "$asr_batch_size" \
        --speechocean-asr-device "$asr_device" \
        --speechocean-asr-compute-type "$asr_compute_type" \
        --speechocean-asr-beam-size "$asr_beam_size" \
        --speechocean-word-alignment "$speechocean_word_alignment" \
        --speechocean-charsiu-model "$speechocean_charsiu_model" \
        --speechocean-charsiu-device "$asr_device" \
        --speechocean-charsiu-model-cache-dir "$charsiu_hf_cache_dir" \
        --multipa-cache-dir "$multipa_cache_dir" \
        --multipa-word-output "$multipa_word_output" \
        --multipa-transcript-source "$multipa_transcript_source" \
        --multipa-transcript-split "$multipa_transcript_split" \
        --multipa-asr-model "$asr_model" \
        --multipa-asr-backend "$eval_asr_backend" \
        --multipa-asr-batch-size "$asr_batch_size" \
        --multipa-asr-device "$asr_device" \
        --multipa-asr-compute-type "$asr_compute_type" \
        --multipa-asr-beam-size "$asr_beam_size" \
        "${multipa_limit_args[@]}" \
        "${precision_args[@]}"
    echo
fi

echo -e "${GREEN}Finished benchmark stages ${stage} through ${stop_stage}.${NC}"
