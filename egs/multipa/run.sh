#!/usr/bin/env bash

# Run the MultiPA model on the MultiPA pilot set or SpeechOcean762 test set.
#
# Stages:
#   0: Download the selected data and all pretrained models
#   1: Extract pretrained models and create the evaluation file list
#   2: Run MultiPA model inference in the selected evaluation mode
#   3: Evaluate all five seeds and report PCC mean/std

set -euo pipefail

. ./path.sh

# Download locations
multipa_repo=yuwchen/multipa
assessment_repo=fuann/multipa-model
speechocean_repo=mispeech/speechocean762
hubert_url=https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
roberta_url=https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz

# All model files and archives live here.
pretrained_dir=pretrained-models
pretrained_model=multipa-model
checkpoint_root=$pretrained_dir/$pretrained_model
model_seeds=(0 1 2 3 4)
fairseq_base_model=$pretrained_dir/fairseq_hubert/hubert_base_ls960.pt
roberta_archive=$pretrained_dir/roberta.base.tar.gz
fairseq_roberta=$pretrained_dir/roberta.base

# Experiment configuration
test_data=multipa
evaluation_mode=both
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
  --evaluation-mode MODE
                  open, close, or both (default: $evaluation_mode).
                  MultiPA always runs open; SpeechOcean762 defaults to both.
  --response-mode MODE Alias for --evaluation-mode (closed also accepted)
  --whisper-model NAME Main transcript Whisper model (default: $whisper_model)
                       SpeechOcean762 open supports medium.en and large-v3;
                       it reuses data/speechocean762/transcript/test JSONL.
  --verbose       Print each audio prediction (default: disabled)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --stop-stage) stop_stage=$2; shift 2 ;;
        --gpu) gpu=$2; shift 2 ;;
        --test-data) test_data=$2; shift 2 ;;
        --evaluation-mode|--response-mode) evaluation_mode=$2; shift 2 ;;
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

[[ $evaluation_mode == closed ]] && evaluation_mode=close

case "$evaluation_mode" in
    open|both) ;;
    close)
        [[ $test_data == speechocean762 ]] || {
            echo "Close evaluation is only available for speechocean762" >&2
            exit 2
        }
        ;;
    *) echo "Unsupported --evaluation-mode: $evaluation_mode" >&2; exit 2 ;;
esac

if [[ $test_data == multipa ]]; then
    evaluation_modes=(open)
elif [[ $evaluation_mode == both ]]; then
    evaluation_modes=(close open)
else
    evaluation_modes=("$evaluation_mode")
fi

data_list=data/${test_data}_test.txt

decode_directory() {
    local seed=$1
    local mode=$2
    local name=decode_${test_data}_${mode}
    if [[ $test_data == speechocean762 && $mode == open ]]; then
        name+=_faster-whisper-${whisper_model}-float16-beam5
    elif [[ $mode == open && $whisper_model != medium.en ]]; then
        name+=_${whisper_model//\//_}
    fi
    echo "exp/$pretrained_model/$seed/$name"
}

summary_directory() {
    local mode=$1
    local name=evaluate_${test_data}_${mode}
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
    hf download "$assessment_repo" \
        --include '*/PRO/best' \
        --local-dir "$checkpoint_root"
    download_file "$hubert_url" "$fairseq_base_model"
    download_file "$roberta_url" "$roberta_archive"
fi

if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
    echo -e "${GREEN}Stage 1: Extract models and prepare the file list${NC}"
    [[ -f $roberta_archive ]] || { echo "Missing archive: $roberta_archive" >&2; exit 1; }
    [[ -f $fairseq_base_model ]] || { echo "Missing HuBERT model: $fairseq_base_model" >&2; exit 1; }

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
    for seed in "${model_seeds[@]}"; do
        checkpoint_dir=$checkpoint_root/$seed
        [[ -f $checkpoint_dir/PRO/best ]] || {
            echo "Missing checkpoint: $checkpoint_dir/PRO/best" >&2
            exit 1
        }
    done
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

    open_transcript_file=
    if [[ $test_data == speechocean762 && " ${evaluation_modes[*]} " == *" open "* ]]; then
        case "$whisper_model" in
            medium.en|large-v3) ;;
            *)
                echo "SpeechOcean762 open supports --whisper-model medium.en or large-v3" >&2
                exit 2
                ;;
        esac
        open_transcript_file=$dataset_dir/transcript/test/faster-whisper-${whisper_model}-float16-beam5.jsonl
        echo -e "${GREEN}Preparing fixed SpeechOcean762 open transcripts: $whisper_model${NC}"
        CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
        CUDA_VISIBLE_DEVICES=$gpu python ./src/prepare_open_transcripts.py \
            --model "$whisper_model" \
            --wav-dir "$data_dir" \
            --datalist "$data_list" \
            --output "$open_transcript_file"
    fi

    for seed in "${model_seeds[@]}"; do
        checkpoint_dir=$checkpoint_root/$seed
        for current_mode in "${evaluation_modes[@]}"; do
            if [[ $test_data == multipa ]]; then
                evaluation_name="MultiPA open"
            else
                evaluation_name="SpeechOcean762 $current_mode"
            fi
            output_dir=$(decode_directory "$seed" "$current_mode")
            prediction_file=$output_dir/test_mb.txt

            echo -e "${GREEN}Stage 2: $evaluation_name inference, seed $seed${NC}"
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
            elif [[ $test_data == speechocean762 ]]; then
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
    done
fi

if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
    if [[ $test_data == speechocean762 ]]; then
        CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1984 \
        python ./src/align_speechocean762.py \
            --scores "$scores_file" \
            --wav-dir "$data_dir" \
            --output-dir "$gt_alignment_dir"
    fi

    for current_mode in "${evaluation_modes[@]}"; do
        if [[ $test_data == multipa ]]; then
            evaluation_name="MultiPA open"
        else
            evaluation_name="SpeechOcean762 $current_mode"
        fi
        pcc_files=()
        for seed in "${model_seeds[@]}"; do
            output_dir=$(decode_directory "$seed" "$current_mode")
            prediction_file=$output_dir/test_mb.txt
            result_file=$output_dir/result.txt
            pcc_file=$output_dir/pcc.json

            echo -e "${GREEN}Stage 3: $evaluation_name evaluation, seed $seed${NC}"
            [[ -f $prediction_file ]] || { echo "Missing predictions: $prediction_file" >&2; exit 1; }
            mkdir -p "$output_dir"
            if [[ $test_data == multipa ]]; then
                python ./src/evaluate_multipa.py \
                    --predictions "$prediction_file" \
                    --annotations "$annotation_file" \
                    --pcc-json "$pcc_file" | tee "$result_file"
            else
                python ./src/evaluate_speechocean762.py \
                    --predictions "$prediction_file" \
                    --scores "$scores_file" \
                    --gt-alignments "$gt_alignment_dir" \
                    --pcc-json "$pcc_file" | tee "$result_file"
            fi
            pcc_files+=("$pcc_file")
        done
        summary_dir=$(summary_directory "$current_mode")
        mkdir -p "$summary_dir"
        python ./src/summarize_pcc.py \
            --results "${pcc_files[@]}" \
            --output "$summary_dir/result_mean_std.txt"
    done
fi

if [[ $stop_stage -ge 3 ]]; then
    echo -e "${GREEN}Done. Results:${NC}"
    for current_mode in "${evaluation_modes[@]}"; do
        echo "  $(summary_directory "$current_mode")/result_mean_std.txt"
    done
else
    echo -e "${GREEN}Done through stage $stop_stage.${NC}"
fi
