#!/usr/bin/env bash

# Environment setup for the MultiPA reproduction recipe.
export LC_ALL=C

# Keep shared, versioned ASR references available from this recipe directory.
multipa_recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
multipa_references_link=$multipa_recipe_dir/references
if [[ ! -e $multipa_references_link && ! -L $multipa_references_link ]]; then
    ln -s ../../references "$multipa_references_link"
elif [[ ! -L $multipa_references_link || $(readlink "$multipa_references_link") != ../../references ]]; then
    echo "Expected $multipa_references_link to be a symlink to ../../references" >&2
    return 1
fi

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

unset multipa_conda_exe multipa_conda_env multipa_recipe_dir multipa_references_link
