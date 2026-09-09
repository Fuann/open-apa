#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
stage=0
stop_stage=3
whisper_model=
transcript_dir=data/speechocean762/transcript/test
response_mode=open
device=cpu
gpu=
threads=4
batch_size=32
max_phones=
limit=0
seeds="0 1 2 3 4"
wav_dir=data/speechocean762/wav
scores=data/speechocean762/scores.json
exp_dir=
feature_dir=
usage() {
  cat <<EOF
Usage: bash run.sh [options]
  --whisper-model NAME  Select one model (default: medium.en then large-v3)
  --transcript-dir PATH Transcript JSONL directory
  --response-mode MODE  open or closed (default: open)
  --stage N            First stage (default: 0)
  --stop-stage N       Last stage (default: 3)
  --device DEVICE      Extraction/inference device (default: cpu)
  --gpu N              Use GPU N (sets --device cuda:0)
  --threads N          PyTorch CPU threads (default: 4)
  --batch-size N       Hippo inference batch size (default: 32)
  --max-phones N       Padded sequence length (default: 65 open / 50 closed)
  --limit N            First N utterances for smoke tests (default: 0 = all)
  --seeds "0 1 ..."     Released Hippo checkpoints (default: all five)
  --wav-dir PATH       Input mono 16 kHz WAV directory
  --scores PATH        SpeechOcean762 scores.json
  --exp-dir PATH       Output directory; required when resuming stage 2 or 3
  --feature-dir PATH   Fresh feature output (default: <exp-dir>/features)

Stages: 0 download models/prepare raw test data; 1 transcripts (generate if missing) and all features;
        2 audit and Hippo inference; 3 evaluation.
Default runs use stable decode names and refuse to overwrite existing features.
Models and model download caches are stored under pretrained-models/.
EOF
}
original_args=("$@")
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0;;
  esac
  if (($# < 2)) || [[ $2 == --* ]]; then
    echo "Missing value for $1" >&2; exit 2
  fi
  case "$1" in
    --stage) stage=$2;;
    --stop-stage) stop_stage=$2;;
    --whisper-model) whisper_model=$2;;
    --transcript-dir) transcript_dir=$2;;
    --response-mode) response_mode=$2;;
    --device) device=$2;;
    --gpu) gpu=$2;;
    --threads) threads=$2;;
    --batch-size) batch_size=$2;;
    --max-phones) max_phones=$2;;
    --limit) limit=$2;;
    --seeds) seeds=$2;;
    --wav-dir) wav_dir=$2;;
    --scores) scores=$2;;
    --exp-dir) exp_dir=$2;;
    --feature-dir) feature_dir=$2;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
  shift 2
done
if [[ -z $max_phones ]]; then
  max_phones=65
  [[ $response_mode != closed ]] || max_phones=50
fi
for value in "$stage" "$stop_stage" "$threads" "$batch_size" "$max_phones" "$limit"; do
  [[ $value =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Expected nonnegative integer: $value" >&2; exit 2; }
done
((stage<=stop_stage && stop_stage<=3 && threads>0 && batch_size>0 && max_phones>0)) || { echo "Invalid stage or numeric settings" >&2; exit 2; }
[[ $response_mode == open || $response_mode == closed ]] || { echo "Invalid response mode" >&2; exit 2; }
[[ -z $whisper_model || $whisper_model =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Use a Whisper model name, not a file path" >&2; exit 2; }
[[ -n $seeds ]] || { echo "At least one seed is required" >&2; exit 2; }
for seed in $seeds; do
  [[ $seed =~ ^[0-4]$ ]] || { echo "Unknown released seed: $seed" >&2; exit 2; }
done
if [[ -n $gpu ]]; then
  [[ $gpu =~ ^[0-9]+$ ]] || { echo "Invalid GPU index" >&2; exit 2; }
  device=cuda:0
fi
if ((stage>=2)) && [[ -z $exp_dir ]]; then
  echo "Resuming requires --exp-dir pointing to a completed extraction run" >&2; exit 2
fi
if [[ -z $whisper_model ]]; then
  if [[ $response_mode == closed ]]; then
    whisper_model=large-v3
  else
    base_exp=${exp_dir:-exp/hippo}
    for selected_model in medium.en large-v3; do
      child_args=("${original_args[@]}" --whisper-model "$selected_model" --exp-dir "$base_exp/decode_speechocean762_open_$selected_model")
      if [[ -n $feature_dir ]]; then
        child_args+=(--feature-dir "$feature_dir/$selected_model")
      fi
      bash "$0" "${child_args[@]}"
    done
    exit 0
  fi
fi
. ./path.sh

decode_name=decode_speechocean762_$response_mode
[[ $response_mode != open ]] || decode_name=${decode_name}_$whisper_model
exp_dir=${exp_dir:-exp/hippo/$decode_name}
feature_dir=${feature_dir:-$exp_dir/features}
models=$PWD/pretrained-models
extract_args=(--models "$models" --whisper-model "$whisper_model" --transcript-dir "$transcript_dir" --response-mode "$response_mode"
  --wav-dir "$wav_dir" --scores "$scores" --output "$feature_dir" --device "$device"
  --threads "$threads" --max-phones "$max_phones" --limit "$limit")
if ((stage<=0 && stop_stage>=0)); then
  echo "Stage 0: models and raw SpeechOcean762 data"
  python src/feats_extract/extract_features.py download "${extract_args[@]}"
  if [[ ! -d $wav_dir || ! -f $scores ]]; then
    if [[ $wav_dir != data/speechocean762/wav || $scores != data/speechocean762/scores.json ]]; then
      echo "Supply existing files for custom --wav-dir and --scores" >&2; exit 1
    fi
    python src/prepare_data.py
  fi
fi
if ((stage<=1 && stop_stage>=1)); then
  echo "Stage 1: transcripts, fresh GOP, SSL and ModernBERT features"
  mkdir -p "$exp_dir"
  python -u src/feats_extract/extract_features.py extract "${extract_args[@]}" 2>&1 | tee "$exp_dir/features.log"
fi
eval_args=(--manifest "$feature_dir/et.csv" --scores "$scores" --feature-dir "$feature_dir"
  --exp-dir "$exp_dir" --response-mode "$response_mode" --asr-variant fresh --seeds "$seeds")
if ((stop_stage>=2)); then
  python src/feats_extract/extract_features.py check "${extract_args[@]}"
fi
if ((stage<=2 && stop_stage>=2)); then
  echo "Stage 2: feature audit and Hippo inference"
  python src/evaluate_speechocean762.py "${eval_args[@]}" --audit-only
  for seed in $seeds; do
    mkdir -p "$exp_dir/$seed"
    python -u inference.py --seed "$seed" --feature-dir "$feature_dir" \
      --checkpoint "$models/hippo/$seed/models/best_audio_model.pth" --exp-dir "$exp_dir/$seed" \
      --response-mode "$response_mode" --asr-variant fresh --batch_size "$batch_size" --threads "$threads" --device "$device" \
      2>&1 | tee "$exp_dir/$seed/inference.log"
  done
fi
if ((stage<=3 && stop_stage>=3)); then
  echo "Stage 3: evaluation"
  python src/prepare_word_evaluation.py --feature-dir "$feature_dir" --output "$exp_dir/word_evaluation.json" --scores "$scores"
  python src/evaluate_speechocean762.py "${eval_args[@]}" --word-evaluation "$exp_dir/word_evaluation.json"
fi
echo "Done through stage $stop_stage. Outputs: $exp_dir"
