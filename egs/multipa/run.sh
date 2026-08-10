#!/usr/bin/env bash

# Run MultiPA inference/evaluation on MultiPA or SpeechOcean762.
#
# Stages:
#   0: Download the selected data and all pretrained models
#   1: Extract pretrained models and create the evaluation file list
#   2: Run MultiPA open-response inference
#   3: Evaluate utterance- and word-level correlations

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
  --test-data SET multipa or speechocean762 (default: $test_data)
  --verbose       Print each audio prediction (default: disabled)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --stop-stage) stop_stage=$2; shift 2 ;;
        --gpu) gpu=$2; shift 2 ;;
        --test-data) test_data=$2; shift 2 ;;
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

data_list=data/${test_data}_test.txt
output_dir=exp/$pretrained_model/decode_$test_data
prediction_file=$output_dir/test_mb.txt
result_file=$output_dir/result.txt

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
            --local-dir "$dataset_dir"
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
    echo -e "${GREEN}Stage 2: $test_data open-response inference${NC}"
    [[ -f $checkpoint_dir/PRO/best ]] || { echo "Missing checkpoint: $checkpoint_dir/PRO/best" >&2; exit 1; }
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

    mkdir -p "$output_dir"
    verbose_option=()
    [[ $verbose == true ]] && verbose_option+=(--verbose)
    CUDA_VISIBLE_DEVICES=$gpu python ./src/test_open.py \
        --fairseq_base_model "$fairseq_base_model" \
        --fairseq_roberta "$fairseq_roberta" \
        --datadir "$data_dir" \
        --datalist "$data_list" \
        --ckptdir "$checkpoint_dir" \
        --output-file "$prediction_file" \
        "${verbose_option[@]}" || exit 1
fi

if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
    echo -e "${GREEN}Stage 3: $test_data evaluation${NC}"
    [[ -f $prediction_file ]] || { echo "Missing predictions: $prediction_file" >&2; exit 1; }
    mkdir -p "$output_dir"
    if [[ $test_data == multipa ]]; then
        python ./src/evaluate_multipa.py \
            --predictions "$prediction_file" \
            --annotations "$annotation_file" | tee "$result_file"
    else
        python ./src/align_speechocean762.py \
            --scores "$scores_file" \
            --wav-dir "$data_dir" \
            --output-dir "$gt_alignment_dir"
        python ./src/evaluate_speechocean762.py \
            --predictions "$prediction_file" \
            --scores "$scores_file" \
            --gt-alignments "$gt_alignment_dir" | tee "$result_file"
    fi
fi

if [[ $stop_stage -ge 3 ]]; then
    echo -e "${GREEN}Done. Results: $result_file${NC}"
else
    echo -e "${GREEN}Done through stage $stop_stage.${NC}"
fi
