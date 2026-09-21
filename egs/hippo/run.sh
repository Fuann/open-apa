#!/usr/bin/env bash

# Run the released Hippo model on SpeechOcean762 or the MultiPA pilot set.
#
# Stages:
#   0: Download data and pretrained models
#   1: Extract features using the selected response transcript
#   2: Audit features and run Hippo inference
#   3: Evaluate utterance- and word-level PCC

set -euo pipefail
cd "$(dirname "$0")"

# Experiment configuration
stage=0
stop_stage=3
whisper_model=
asr_backend=whisperx
test_data=speechocean762
transcript_dir=
response_mode=open
device=cuda:0
gpu=0
threads=4
batch_size=32
max_phones=
limit=0
wav_dir=
scores=
annotations=
multipa_word_alignment=whisperx
exp_dir=
feature_dir=

# Model locations
hippo_github=https://github.com/bicheng1225/HIPPO
ctc_gop_github=https://github.com/frank613/CTC-based-GOP
hippo_model_url=https://raw.githubusercontent.com/bicheng1225/HIPPO/main/pretrained-models/hippo/best_audio_model.pth
ctc_model_base=https://raw.githubusercontent.com/frank613/CTC-based-GOP/main/is24/models
ctc_weights_url=https://media.githubusercontent.com/media/frank613/CTC-based-GOP/main/is24/models/checkpoint-8000/pytorch_model.bin
models=$PWD/pretrained-models
checkpoint=$models/hippo/best_audio_model.pth

GREEN='\033[0;32m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: bash run.sh [options]
    --whisper-model NAME  Select one model (default: medium.en then large-v3)
    --asr-backend NAME    faster-whisper or whisperx (default: $asr_backend)
    --test-data SET       speechocean762 or multipa (default: $test_data)
    --response-mode MODE  open or closed (default: open)
    --stage N            First stage (default: 0)
    --stop-stage N       Last stage (default: 3)
    --device DEVICE      Extraction/inference device (default: cuda:0)
    --gpu N              Use GPU N (sets --device cuda:0)
    --threads N          PyTorch CPU threads (default: 4)
    --batch-size N       Hippo inference batch size (default: 32)
    --max-phones N       Padded sequence length (default: 65 open / 50 closed)
    --limit N            First N utterances for smoke tests (default: 0 = all)
    --wav-dir PATH       Input mono 16 kHz WAV directory
    --scores PATH        Prepared dataset scores JSON
    --annotations PATH   MultiPA annotation.csv
    --transcript-dir PATH  Directory containing fixed ASR transcript JSONL
    --multipa-word-alignment NAME  whisperx or ctc (default: $multipa_word_alignment)
    --exp-dir PATH       Output directory; required when resuming stage 2 or 3
    --feature-dir PATH   Fresh feature output (default: <exp-dir>/features)

Stages: 0 download models/prepare raw test data; 1 load provided transcripts and extract all features;
                2 audit and Hippo inference; 3 evaluation.
Default runs use stable decode names and refuse to overwrite existing features.
Models and model download caches are stored under pretrained-models/.
EOF
}

download_file() {
    local url=$1
    local destination=$2
    local partial=$destination.part
    if [[ -s $destination ]]; then
        echo "Already downloaded: $destination"
        return
    fi
    mkdir -p "$(dirname -- "$destination")"
    curl --fail --location --retry 3 --continue-at - \
        --output "$partial" "$url"
    mv "$partial" "$destination"
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
        --asr-backend) asr_backend=$2;;
        --test-data) test_data=$2;;
        --response-mode) response_mode=$2;;
        --device) device=$2;;
        --gpu) gpu=$2;;
        --threads) threads=$2;;
        --batch-size) batch_size=$2;;
        --max-phones) max_phones=$2;;
        --limit) limit=$2;;
        --wav-dir) wav_dir=$2;;
        --scores) scores=$2;;
        --annotations) annotations=$2;;
        --transcript-dir) transcript_dir=$2;;
        --multipa-word-alignment) multipa_word_alignment=$2;;
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
[[ $asr_backend == faster-whisper || $asr_backend == whisperx ]] || { echo "Invalid ASR backend: $asr_backend" >&2; exit 2; }
[[ $test_data == speechocean762 || $test_data == multipa ]] || { echo "Invalid test data: $test_data" >&2; exit 2; }
[[ $multipa_word_alignment == whisperx || $multipa_word_alignment == ctc ]] || { echo "Invalid MultiPA word alignment: $multipa_word_alignment" >&2; exit 2; }
[[ $test_data != multipa || $response_mode == open ]] || { echo "MultiPA supports open response only" >&2; exit 2; }
[[ -z $whisper_model || $whisper_model =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Use a Whisper model name, not a file path" >&2; exit 2; }
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
            child_name="decode_${test_data}_open_${asr_backend}_$selected_model"
            [[ $test_data != multipa ]] || child_name+=_${multipa_word_alignment}
            child_args=("${original_args[@]}" --whisper-model "$selected_model"
                --exp-dir "$base_exp/$child_name")
            if [[ -n $feature_dir ]]; then
                child_args+=(--feature-dir "$feature_dir/$selected_model")
            fi
            bash "$0" "${child_args[@]}"
        done
        exit 0
    fi
fi
. ./path.sh

if [[ $test_data == multipa ]]; then
    transcript_dir=${transcript_dir:-references/multipa/transcripts}
    wav_dir=${wav_dir:-data/multipa/wav}
    annotations=${annotations:-data/multipa/annotation.csv}
    scores=${scores:-data/multipa/scores/${asr_backend}-${whisper_model}-${multipa_word_alignment}.json}
else
    transcript_dir=${transcript_dir:-references/speechocean762/test/transcripts}
    wav_dir=${wav_dir:-data/speechocean762/wav}
    scores=${scores:-data/speechocean762/scores.json}
fi

if [[ $response_mode == open ]]; then
    transcript_file="$transcript_dir/${asr_backend}-${whisper_model}-float16-beam5.jsonl"
    if [[ ! -f $transcript_file && $test_data == multipa ]]; then
        local_transcript="data/multipa/transcripts/${asr_backend}-${whisper_model}-float16-beam5.jsonl"
        if [[ -f $local_transcript ]]; then
            echo "Shared transcript unavailable; using recipe-local cache: $local_transcript"
            transcript_file=$local_transcript
            transcript_dir=$(dirname -- "$local_transcript")
        fi
    fi
    [[ -f $transcript_file ]] || {
        echo "Missing fixed ASR transcripts: $transcript_file" >&2
        exit 1
    }
fi

decode_name=decode_${test_data}_$response_mode
[[ $response_mode != open ]] || decode_name=${decode_name}_${asr_backend}_${whisper_model}
[[ $test_data != multipa ]] || decode_name=${decode_name}_${multipa_word_alignment}
exp_dir=${exp_dir:-exp/hippo/$decode_name}
feature_dir=${feature_dir:-$exp_dir/features}
extract_args=(--models "$models" --whisper-model "$whisper_model" --asr-backend "$asr_backend"
    --transcript-dir "$transcript_dir" --response-mode "$response_mode"
    --wav-dir "$wav_dir" --scores "$scores" --output "$feature_dir" --device "$device"
    --threads "$threads" --max-phones "$max_phones" --limit "$limit")
if ((stage<=0 && stop_stage>=0)); then
    echo -e "${GREEN}Stage 0: Download data and pretrained models${NC}"
    command -v curl >/dev/null 2>&1 || { echo "Missing command: curl" >&2; exit 1; }
    echo "Hippo model source: $hippo_github"
    echo "CTC-GOP model source: $ctc_gop_github"
    download_file "$hippo_model_url" "$checkpoint"
    download_file "$ctc_model_base/checkpoint-8000/config.json" \
        "$models/ctc-gop/checkpoint-8000/config.json"
    download_file "$ctc_weights_url" \
        "$models/ctc-gop/checkpoint-8000/pytorch_model.bin"
    [[ $(stat -c %s "$checkpoint") -gt 1000000 ]] || {
        echo "Invalid Hippo checkpoint downloaded from $hippo_github" >&2; exit 1
    }
    [[ $(stat -c %s "$models/ctc-gop/checkpoint-8000/pytorch_model.bin") -gt 1000000000 ]] || {
        echo "Invalid CTC-GOP weights (possibly a Git LFS pointer) from $ctc_gop_github" >&2; exit 1
    }
    for file in preprocessor_config.json special_tokens_map.json tokenizer_config.json vocab.json; do
        download_file "$ctc_model_base/processor_config_gop/$file" \
            "$models/ctc-gop/processor_config_gop/$file"
    done
    python src/feats_extract/extract_features.py download "${extract_args[@]}"
    if [[ $test_data == multipa ]]; then
        command -v hf >/dev/null 2>&1 || { echo "Missing command: hf" >&2; exit 1; }
        [[ -f $annotations && -d $wav_dir ]] || {
            hf download yuwchen/multipa wav/ annotation.csv --local-dir data/multipa --quiet
        }
        python src/prepare_multipa.py --annotations "$annotations" \
            --transcript "$transcript_file" --models "$models" --output "$scores" \
            --word-alignment "$multipa_word_alignment" --wav-dir "$wav_dir" --device "$device"
    elif [[ ! -d $wav_dir || ! -f $scores ]]; then
        if [[ $wav_dir != data/speechocean762/wav || $scores != data/speechocean762/scores.json ]]; then
            echo "Supply existing files for custom --wav-dir and --scores" >&2; exit 1
        fi
        python src/prepare_data.py
    fi
fi
if ((stage<=1 && stop_stage>=1)); then
    echo -e "${GREEN}Stage 1: Extract GOP, SSL, and ModernBERT features${NC}"
    mkdir -p "$exp_dir"
    python -u src/feats_extract/extract_features.py extract "${extract_args[@]}" 2>&1 | tee "$exp_dir/features.log"
fi
eval_args=(--manifest "$feature_dir/et.csv" --scores "$scores" --feature-dir "$feature_dir"
    --exp-dir "$exp_dir" --response-mode "$response_mode" --asr-variant fresh)
if ((stop_stage>=2)); then
    python src/feats_extract/extract_features.py check "${extract_args[@]}"
fi
word_evaluation_args=()
if ((stop_stage>=2)); then
    if [[ $response_mode == open ]]; then
        statistics_file="$exp_dir/word_evaluation.statistics.json"
        word_evaluation_file="$exp_dir/word_evaluation.json"
        python src/prepare_word_evaluation.py --feature-dir "$feature_dir" \
            --output "$word_evaluation_file" --statistics-output "$statistics_file" \
            --transcript "$transcript_file" --scores "$scores"
        word_evaluation_args=(--word-evaluation "$word_evaluation_file")
    fi
fi
if ((stage<=2 && stop_stage>=2)); then
    echo -e "${GREEN}Stage 2: Audit features and run Hippo inference${NC}"
    [[ -f $checkpoint ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
    python src/evaluate_speechocean762.py "${eval_args[@]}" --audit-only
    python -u inference.py --seed 0 --feature-dir "$feature_dir" \
        --checkpoint "$checkpoint" --exp-dir "$exp_dir" \
        --response-mode "$response_mode" --asr-variant fresh \
        --batch_size "$batch_size" --threads "$threads" --device "$device" \
        --scores "$scores" "${word_evaluation_args[@]}" \
        2>&1 | tee "$exp_dir/inference.log"
fi
if ((stage<=3 && stop_stage>=3)); then
    echo -e "${GREEN}Stage 3: Evaluate utterance- and word-level PCC${NC}"
    if [[ $test_data == multipa ]]; then
        python src/evaluate_multipa.py --manifest "$feature_dir/et.csv" \
            --scores "$scores" --feature-dir "$feature_dir" --exp-dir "$exp_dir"
    else
        python src/evaluate_speechocean762.py "${eval_args[@]}" "${word_evaluation_args[@]}"
    fi
fi
echo -e "${GREEN}Done through stage $stop_stage. Outputs: $exp_dir${NC}"
