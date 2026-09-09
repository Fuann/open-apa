#!/usr/bin/env bash

# Environment setup for the Hippo recipe (Conda setup from ../multipa/path.sh).
export LC_ALL=C

multipa_conda_env=${MULTIPA_CONDA_ENV:-multipa}

if command -v conda >/dev/null 2>&1; then
    multipa_conda_exe=$(command -v conda)
elif [[ -n ${CONDA_EXE:-} && -x $CONDA_EXE ]]; then
    multipa_conda_exe=$CONDA_EXE
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    multipa_conda_exe="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    multipa_conda_exe="$HOME/miniforge3/bin/conda"
else
    echo "Conda was not found. Install Conda or add it to PATH." >&2
    return 1
fi

eval "$("$multipa_conda_exe" shell.bash hook)"
conda activate "$multipa_conda_env"

unset multipa_conda_exe multipa_conda_env

# Resolve paths relative to this file, including when sourced from another directory.
hippo_recipe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
hippo_models_dir=$hippo_recipe_dir/pretrained-models
export HF_HOME="$hippo_models_dir/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
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
unset hippo_recipe_dir hippo_models_dir
