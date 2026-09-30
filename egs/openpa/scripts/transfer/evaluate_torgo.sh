#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
. "$repo_dir/path.sh"

stage=0
stop_stage=1
checkpoint=pretrained-models/openpa
output_dir=exp/openpa-pretrained/transfer/torgo
conditions=both
balancing=both
word_output=accuracy
utterance_output=accuracy
asr_model=large-v3
asr_backend=whisperx
gpu=0
batch_size=32
num_workers=8
max_samples=

usage() {
    cat <<EOF
Usage: bash scripts/transfer/evaluate_torgo.sh [options]

Evaluate OpenPA on PathBench TORGO dysarthric speech.

    --stage N              first stage (default: $stage)
    --stop-stage N         last stage (default: $stop_stage)
    --checkpoint DIR       OpenPA checkpoint (default: $checkpoint)
    --output-dir DIR       result directory (default: $output_dir)
    --conditions MODES     reference_text, reference_free, or both
    --balancing MODE       balanced, unbalanced, or both
    --word-output LABEL    OpenPA word head (default: $word_output)
    --utterance-output LABEL
                          accuracy, fluency, prosodic, or total
                          (default: $utterance_output)
    --asr-model MODEL      reference-free ASR model (default: $asr_model)
    --asr-backend BACKEND  whisperx or faster-whisper (default: $asr_backend)
    --gpu ID               CUDA device ID (default: $gpu)
    --batch-size N         OpenPA inference batch size (default: $batch_size)
    --num-workers N        DataLoader workers (default: $num_workers)
    --max-samples N        unbalanced smoke-test limit per speaker

Stages:
    0  Download, extract, and validate TORGO
    1  Run zero-shot OpenPA evaluation
EOF
}

help_message="$(usage)"
. ./parse_options.sh || exit 1
[[ $# -eq 0 ]] || { echo "Unexpected arguments: $*" >&2; exit 2; }
[[ "$stage" =~ ^[0-1]$ && "$stop_stage" =~ ^[0-1]$ ]] && \
    (( stage <= stop_stage )) || {
    echo "Stages must satisfy 0 <= stage <= stop-stage <= 1." >&2
    exit 2
}
case "$conditions" in reference_text|reference_free|both) ;; *)
    echo "Invalid --conditions: $conditions" >&2; exit 2 ;; esac
case "$balancing" in balanced|unbalanced|both) ;; *)
    echo "Invalid --balancing: $balancing" >&2; exit 2 ;; esac
[[ -n "$word_output" ]] || { echo "--word-output must not be empty" >&2; exit 2; }
case "$utterance_output" in accuracy|fluency|prosodic|total) ;; *)
    echo "Invalid --utterance-output: $utterance_output" >&2; exit 2 ;; esac
case "$asr_backend" in whisperx|faster-whisper) ;; *)
    echo "Invalid --asr-backend: $asr_backend" >&2; exit 2 ;; esac

if (( stage <= 0 && stop_stage >= 0 )); then
    bash scripts/transfer/download_torgo.sh
fi
if (( stage <= 1 && stop_stage >= 1 )); then
    [[ -f "$checkpoint/config.json" || -f "$checkpoint/adapter_config.json" ]] || {
        echo "Missing OpenPA checkpoint: $checkpoint" >&2
        exit 1
    }
    limit_args=()
    [[ -z "$max_samples" ]] || limit_args=(--torgo-max-samples "$max_samples")
    CUDA_VISIBLE_DEVICES="$gpu" python src/evaluate.py \
        --checkpoint "$checkpoint" \
        --test-sets torgo \
        --torgo-output-dir "$output_dir" \
        --torgo-root data/torgo/raw \
        --torgo-manifest-root data/torgo/pathbench \
        --torgo-conditions "$conditions" \
        --torgo-balancing "$balancing" \
        --torgo-word-output "$word_output" \
        --torgo-utterance-output "$utterance_output" \
        --torgo-batch-size "$batch_size" \
        --torgo-asr-model "$asr_model" \
        --torgo-asr-backend "$asr_backend" \
        --torgo-asr-device cuda:0 \
        --num-workers "$num_workers" \
        "${limit_args[@]}"
fi
