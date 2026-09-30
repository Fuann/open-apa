#!/usr/bin/env bash

# Shared environment setup for the OpenPA recipe.
if [[ ${OPENPA_PATH_INITIALIZED:-false} == true ]]; then
    return 0
fi
export LC_ALL=C

openpa_recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# All benchmark recipes share the versioned references at
# open-apa/references through this relative symlink.
openpa_references_link=$openpa_recipe_dir/references
if [[ ! -e $openpa_references_link && ! -L $openpa_references_link ]]; then
    ln -s ../../references "$openpa_references_link"
elif [[ ! -L $openpa_references_link || $(readlink "$openpa_references_link") != ../../references ]]; then
    echo "Expected $openpa_references_link to be a symlink to ../../references" >&2
    return 1
fi
if [[ ! -d $openpa_references_link ]]; then
    echo "Shared benchmark references are unavailable." >&2
    echo "Expected open-apa/references through $openpa_references_link." >&2
    return 1
fi
export OPENPA_REFERENCES_DIR=$openpa_references_link

openpa_conda_env=${OPENPA_CONDA_ENV:-openpa}
openpa_conda_exe=${OPENPA_CONDA_EXE:-${CONDA_EXE:-}}

if [[ ${CONDA_DEFAULT_ENV:-} == "$openpa_conda_env" ]]; then
    :
elif [[ -n $openpa_conda_exe && -x $openpa_conda_exe ]]; then
    eval "$("$openpa_conda_exe" shell.bash hook)"
    conda activate "$openpa_conda_env"
elif command -v conda >/dev/null 2>&1; then
    openpa_conda_exe=$(command -v conda)
    eval "$("$openpa_conda_exe" shell.bash hook)"
    conda activate "$openpa_conda_env"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    openpa_conda_exe="$HOME/miniconda3/bin/conda"
    eval "$("$openpa_conda_exe" shell.bash hook)"
    conda activate "$openpa_conda_env"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    openpa_conda_exe="$HOME/miniforge3/bin/conda"
    eval "$("$openpa_conda_exe" shell.bash hook)"
    conda activate "$openpa_conda_env"
else
    echo "Conda was not found. Add it to PATH or set OPENPA_CONDA_EXE." >&2
    return 1
fi

# Keep downloaded models and generated preprocessing caches separate. The
# token remains in the user's standard Hugging Face location and is never
# stored in this repository.
openpa_models_dir=${OPENPA_MODELS_DIR:-$openpa_recipe_dir/pretrained-models}
openpa_cache_dir=${OPENPA_CACHE_DIR:-$openpa_recipe_dir/.cache}

export HF_HOME="$openpa_models_dir/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$openpa_models_dir/xet"
export HF_DATASETS_CACHE="$openpa_cache_dir/huggingface/datasets"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$openpa_models_dir/torch"
export XDG_CACHE_HOME="$openpa_cache_dir"
export NLTK_DATA="$openpa_models_dir/nltk_data"
export PYTHONPATH="$openpa_recipe_dir/src${PYTHONPATH:+:$PYTHONPATH}"

openpa_reference_transcript() {
    local dataset_name=$1
    local backend_name=$2
    local model_name=$3
    local compute_type=$4
    local beam_size=$5
    local transcript_dir filename model_slug

    model_slug=${model_name//\//_}
    case "$dataset_name" in
        speechocean762)
            transcript_dir=$OPENPA_REFERENCES_DIR/speechocean762/test/transcripts
            ;;
        multipa)
            transcript_dir=$OPENPA_REFERENCES_DIR/multipa/transcripts
            ;;
        *)
            return 1
            ;;
    esac
    if [[ $backend_name == qwen3-asr ]]; then
        filename="${backend_name}-${model_slug}-${compute_type}.jsonl"
    else
        filename="${backend_name}-${model_slug}-${compute_type}-beam${beam_size}.jsonl"
    fi
    [[ -f $transcript_dir/$filename ]] || return 1
    printf '%s\n' "$transcript_dir/$filename"
}

mkdir -p \
    "$HF_HUB_CACHE" \
    "$HF_XET_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$TORCH_HOME" \
    "$NLTK_DATA"

OPENPA_PATH_INITIALIZED=true
unset openpa_recipe_dir openpa_references_link openpa_conda_env openpa_conda_exe
unset openpa_models_dir openpa_cache_dir
