#!/usr/bin/env bash

# Environment setup for the MultiPA reproduction recipe.
export LC_ALL=C

multipa_conda_env=${MULTIPA_CONDA_ENV:-multipa}

if command -v conda >/dev/null 2>&1; then
    multipa_conda_exe=$(command -v conda)
elif [[ -x /share/homes/fuann/miniconda3/bin/conda ]]; then
    multipa_conda_exe=/share/homes/fuann/miniconda3/bin/conda
else
    echo "Conda was not found. Install Conda or add it to PATH." >&2
    return 1
fi

eval "$("$multipa_conda_exe" shell.bash hook)"
conda activate "$multipa_conda_env"

unset multipa_conda_exe multipa_conda_env
