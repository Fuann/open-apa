#!/usr/bin/env bash

# Environment setup for the MultiPA reproduction recipe.
export LC_ALL=C

multipa_conda_env=${MULTIPA_CONDA_ENV:-multipa}
multipa_conda_exe=${MULTIPA_CONDA_EXE:-${CONDA_EXE:-}}

if [[ -n $multipa_conda_exe && -x $multipa_conda_exe ]]; then
    :
elif command -v conda >/dev/null 2>&1; then
    multipa_conda_exe=$(command -v conda)
else
    echo "Conda was not found. Add it to PATH or set MULTIPA_CONDA_EXE." >&2
    return 1
fi

eval "$("$multipa_conda_exe" shell.bash hook)"
conda activate "$multipa_conda_env"

unset multipa_conda_exe multipa_conda_env
