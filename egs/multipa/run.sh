#!/usr/bin/env bash

# Run the MultiPA model on the MultiPA pilot set or SpeechOcean762 test set.
#
# Stages:
#   0: Download the selected data and all pretrained models
#   1: Extract pretrained models and create the evaluation file list
#   2: Run MultiPA model inference in the selected evaluation mode
#   3: Evaluate utterance- and word-level PCC

set -euo pipefail

. ./path.sh

# Download locations
multipa_repo=yuwchen/multipa
speechocean_repo=mispeech/speechocean762
hubert_url=https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
roberta_url=https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz

# All model files and archives live here.
pretrained_dir=pretrained-models
pretrained_model=model_assessment_val9_r1
assessment_archive=$pretrained_dir/$pretrained_model.zip
checkpoint_dir=$pretrained_dir/$pretrained_model
fairseq_base_model=$pretrained_dir/fairseq_hubert/hubert_base_ls960.pt
roberta_archive=$pretrained_dir/roberta.base.tar.gz
fairseq_roberta=$pretrained_dir/roberta.base

# Experiment configuration
test_data=multipa
response_mode=both
whisper_model=medium.en
gpu=0
verbose=false

# Stage configuration
stage=0
stop_stage=3

GREEN='\033[0;32m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --stage N       First stage to run (default: $stage)
  --stop-stage N  Last stage to run (default: $stop_stage)
  --gpu N         CUDA device index (default: $gpu)
  --test-data SET MultiPA pilot set (multipa) or SpeechOcean762 test set
                  (speechocean762; default: $test_data)
  --response-mode MODE
                  open, close, or both (default: $response_mode).
                  MultiPA always runs open; SpeechOcean762 defaults to both.
                  closed is accepted as an alias for close.
  --whisper-model NAME Main transcript Whisper model (default: $whisper_model)
                       MultiPA open and SpeechOcean762 open support medium.en
                       and large-v3, using fixed JSONL under references/.
  --verbose       Print each audio prediction (default: disabled)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --stop-stage) stop_stage=$2; shift 2 ;;
        --gpu) gpu=$2; shift 2 ;;
        --test-data) test_data=$2; shift 2 ;;
        --response-mode) response_mode=$2; shift 2 ;;
        --whisper-model) whisper_model=$2; shift 2 ;;
        --verbose) verbose=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$test_data" in
    multipa)
        dataset_dir=data/multipa
        data_dir=$dataset_dir/wav
        annotation_file=$dataset_dir/annotation.csv
        expected_samples=50
        ;;
    speechocean762)
        dataset_dir=data/speechocean762
        data_dir=$dataset_dir/wav
        parquet_file=$dataset_dir/data/test-00000-of-00001.parquet
        scores_file=$dataset_dir/scores.json
        gt_alignment_dir=$dataset_dir/gt-alignments
        expected_samples=2500
        ;;
    *) echo "Unsupported --test-data: $test_data" >&2; exit 2 ;;
esac

[[ $response_mode == closed ]] && response_mode=close

case "$response_mode" in
    open|both) ;;
    close)
        [[ $test_data == speechocean762 ]] || {
            echo "Close evaluation is only available for speechocean762" >&2
            exit 2
        }
        ;;
    *) echo "Unsupported --response-mode: $response_mode" >&2; exit 2 ;;
esac

if [[ $test_data == multipa ]]; then
    response_modes=(open)
elif [[ $response_mode == both ]]; then
    response_modes=(close open)
else
    response_modes=("$response_mode")
fi

data_list=data/${test_data}_test.txt

open_transcript_file=
if [[ " ${response_modes[*]} " == *" open "* ]]; then
    case "$whisper_model" in
        medium.en|large-v3) ;;
        *)
            echo "$test_data open supports --whisper-model medium.en or large-v3" >&2
            exit 2
            ;;
    esac
    if [[ $test_data == multipa ]]; then
        open_transcript_file=references/multipa/transcripts/faster-whisper-${whisper_model}-float16-beam5.jsonl
    else
        open_transcript_file=references/speechocean762/test/transcripts/faster-whisper-${whisper_model}-float16-beam5.jsonl
    fi
    [[ -f $open_transcript_file ]] || {
        echo "Missing fixed ASR transcripts: $open_transcript_file" >&2
        exit 1
    }
fi

decode_directory() {
    local mode=$1
    local name=decode_${test_data}_${mode}
    if [[ $test_data == speechocean762 && $mode == open ]]; then
        name+=_faster-whisper-${whisper_model}-float16-beam5
    elif [[ $mode == open && $whisper_model != medium.en ]]; then
        name+=_${whisper_model//\//_}
    fi
    echo "exp/$pretrained_model/$name"
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

if [[ $stage -le 0 && $stop_stage -ge 0 ]]; then
    echo -e "${GREEN}Stage 0: Download data and pretrained models${NC}"
    command -v hf >/dev/null 2>&1 || { echo "Missing command: hf" >&2; exit 1; }
    command -v curl >/dev/null 2>&1 || { echo "Missing command: curl" >&2; exit 1; }

    mkdir -p "$dataset_dir" "$pretrained_dir"
    if [[ $test_data == multipa ]]; then
        hf download "$multipa_repo" wav/ annotation.csv annotation.xlsx \
            --local-dir "$dataset_dir" --quiet
    else
        hf download "$speechocean_repo" data/test-00000-of-00001.parquet \
            --repo-type dataset --local-dir "$dataset_dir"
    fi
    hf download "$multipa_repo" model_assessment_val9_r1.zip \
        --local-dir "$pretrained_dir"
    download_file "$hubert_url" "$fairseq_base_model"
    download_file "$roberta_url" "$roberta_archive"
fi

if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
    echo -e "${GREEN}Stage 1: Extract models and prepare the file list${NC}"
    [[ -f $assessment_archive ]] || { echo "Missing archive: $assessment_archive" >&2; exit 1; }
    [[ -f $roberta_archive ]] || { echo "Missing archive: $roberta_archive" >&2; exit 1; }
    [[ -f $fairseq_base_model ]] || { echo "Missing HuBERT model: $fairseq_base_model" >&2; exit 1; }

    unzip -q -o "$assessment_archive" -d "$pretrained_dir"
    tar -xzf "$roberta_archive" -C "$pretrained_dir"
    if [[ $test_data == multipa ]]; then
        [[ -d $data_dir ]] || { echo "Missing audio directory: $data_dir" >&2; exit 1; }
        [[ -f $annotation_file ]] || { echo "Missing annotations: $annotation_file" >&2; exit 1; }
        find "$data_dir" -maxdepth 1 -type f -iname '*.wav' -printf '%f\n' \
            | LC_ALL=C sort > "$data_list"
    else
        [[ -f $parquet_file ]] || { echo "Missing parquet: $parquet_file" >&2; exit 1; }
        python -c 'import pyarrow' 2>/dev/null || {
            echo "Missing Python package: pyarrow" >&2
            echo "Install it with: python -m pip install pyarrow" >&2
            exit 1
        }
        python ./src/prepare_speechocean762.py \
            --parquet "$parquet_file" \
            --wav-dir "$data_dir" \
            --data-list "$data_list" \
            --scores "$scores_file"
    fi
    chmod -R a+rX "$dataset_dir"

    sample_count=$(wc -l < "$data_list")
    [[ $sample_count -eq $expected_samples ]] || {
        echo "Expected $expected_samples WAV files, found $sample_count" >&2
        exit 1
    }
fi

if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
    [[ -f $checkpoint_dir/PRO/best ]] || {
        echo "Missing checkpoint: $checkpoint_dir/PRO/best" >&2
        exit 1
    }
    [[ -f $fairseq_base_model ]] || { echo "Missing HuBERT model: $fairseq_base_model" >&2; exit 1; }
    [[ -f $fairseq_roberta/model.pt && -f $fairseq_roberta/dict.txt ]] || {
        echo "Missing RoBERTa model.pt or dict.txt in: $fairseq_roberta" >&2
        exit 1
    }
    [[ -f $data_list ]] || { echo "Missing data list: $data_list" >&2; exit 1; }

    while IFS= read -r audio_file; do
        [[ -r $data_dir/$audio_file ]] || {
            echo "Audio file is not readable: $data_dir/$audio_file" >&2
            exit 1
        }
    done < "$data_list"

    for current_mode in "${response_modes[@]}"; do
        if [[ $test_data == multipa ]]; then
            evaluation_name="MultiPA open"
        else
            evaluation_name="SpeechOcean762 $current_mode"
        fi
        output_dir=$(decode_directory "$current_mode")
        prediction_file=$output_dir/test_mb.txt

        echo -e "${GREEN}Stage 2: $evaluation_name inference${NC}"
        mkdir -p "$output_dir"
        if [[ -s $prediction_file ]] && [[ $(wc -l < "$prediction_file") -eq $expected_samples ]]; then
            echo "Complete prediction exists, skipping: $prediction_file"
            continue
        fi
        verbose_option=()
        [[ $verbose == true ]] && verbose_option+=(--verbose)
        transcript_option=()
        if [[ $current_mode == close ]]; then
            [[ -f $scores_file ]] || { echo "Missing scores/transcripts: $scores_file" >&2; exit 1; }
            transcript_option+=(--transcripts "$scores_file")
        else
            transcript_option+=(--open-transcripts "$open_transcript_file")
        fi
        CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1984 \
        CUDA_VISIBLE_DEVICES=$gpu python ./src/test_open.py \
            --fairseq_base_model "$fairseq_base_model" \
            --fairseq_roberta "$fairseq_roberta" \
            --datadir "$data_dir" \
            --datalist "$data_list" \
            --ckptdir "$checkpoint_dir" \
            --output-file "$prediction_file" \
            --whisper-model "$whisper_model" \
            "${transcript_option[@]}" \
            "${verbose_option[@]}" || exit 1
    done
fi

if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
    if [[ $test_data == speechocean762 && " ${response_modes[*]} " == *" close "* ]]; then
        CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1984 \
        python ./src/align_speechocean762.py \
            --scores "$scores_file" \
            --wav-dir "$data_dir" \
            --output-dir "$gt_alignment_dir"
    fi

    for current_mode in "${response_modes[@]}"; do
        if [[ $test_data == multipa ]]; then
            evaluation_name="MultiPA open"
        else
            evaluation_name="SpeechOcean762 $current_mode"
        fi
        output_dir=$(decode_directory "$current_mode")
        prediction_file=$output_dir/test_mb.txt
        result_file=$output_dir/result.txt

        echo -e "${GREEN}Stage 3: $evaluation_name evaluation${NC}"
        [[ -f $prediction_file ]] || { echo "Missing predictions: $prediction_file" >&2; exit 1; }
        mkdir -p "$output_dir"
        if [[ $test_data == multipa ]]; then
            python ./src/evaluate_multipa.py \
                --predictions "$prediction_file" \
                --annotations "$annotation_file" | tee "$result_file"
        else
            word_evaluation_args=()
            if [[ $current_mode == open ]]; then
                word_evaluation_args=(--open-transcripts "$open_transcript_file"
                    --word-evaluation-json "$output_dir/word-evaluation.json")
            fi
            python ./src/evaluate_speechocean762.py \
                --predictions "$prediction_file" \
                --scores "$scores_file" \
                --gt-alignments "$gt_alignment_dir" \
                --evaluation-mode "$current_mode" \
                "${word_evaluation_args[@]}" | tee "$result_file"
        fi
    done
fi

if [[ $stop_stage -ge 3 ]]; then
    echo -e "${GREEN}Done. Results:${NC}"
    for current_mode in "${response_modes[@]}"; do
        echo "  $(decode_directory "$current_mode")/result.txt"
    done
else
    echo -e "${GREEN}Done through stage $stop_stage.${NC}"
fi
