#!/usr/bin/env bash

# Environment setup for the Hippo recipe.
export LC_ALL=C

# Keep shared, versioned ASR references available from this recipe directory.
hippo_recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hippo_references_link=$hippo_recipe_dir/references
if [[ ! -e $hippo_references_link && ! -L $hippo_references_link ]]; then
    ln -s ../../references "$hippo_references_link"
elif [[ ! -L $hippo_references_link || $(readlink "$hippo_references_link") != ../../references ]]; then
    echo "Expected $hippo_references_link to be a symlink to ../../references" >&2
    return 1
fi

hippo_conda_env=${HIPPO_CONDA_ENV:-hippo}
hippo_conda_exe=${HIPPO_CONDA_EXE:-${CONDA_EXE:-}}

if [[ -n $hippo_conda_exe && -x $hippo_conda_exe ]]; then
    :
elif command -v conda >/dev/null 2>&1; then
    hippo_conda_exe=$(command -v conda)
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    hippo_conda_exe="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    hippo_conda_exe="$HOME/miniforge3/bin/conda"
else
    echo "Conda was not found. Add it to PATH or set HIPPO_CONDA_EXE." >&2
    return 1
fi

eval "$("$hippo_conda_exe" shell.bash hook)"
conda activate "$hippo_conda_env"

unset hippo_conda_exe hippo_conda_env

# Resolve paths relative to this file, including when sourced from another directory.
hippo_models_dir=$hippo_recipe_dir/pretrained-models
export HF_HOME="$hippo_models_dir/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
# Keep credentials in the standard user location while model caches stay local.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_XET_CACHE="$hippo_models_dir/xet"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$hippo_models_dir/torch"
export XDG_CACHE_HOME="$hippo_models_dir/cache"
export NLTK_DATA="$hippo_models_dir/nltk_data"
export PYTHONPATH="$hippo_recipe_dir/.deps:$hippo_recipe_dir/src${PYTHONPATH:+:$PYTHONPATH}"

# run.sh parses and validates --gpu before sourcing this file.
if [[ -n ${gpu:-} ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu"
fi
unset hippo_recipe_dir hippo_models_dir hippo_references_link
