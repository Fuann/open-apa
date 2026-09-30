#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
. "$repo_dir/path.sh"

# TORGO public release downloader.
# Stages:
#   0: download PathBench-required archives (or the complete release)
#   1: extract the archives
#   2: validate the extracted speaker/audio/prompt structure

stage=0
stop_stage=2
download_dir=data/torgo/archives
output_dir=data/torgo/raw
base_url=https://www.cs.toronto.edu/~complingweb/data/TORGO
pathbench_only=true

pathbench_archives=(F.tar.bz2 M.tar.bz2)
complete_archives=(F.tar.bz2 FC.tar.bz2 M.tar.bz2 MC.tar.bz2)
# These two files live under doc/ on the official server, but are stored next
# to the archives locally.
supplemental=(doc/ERRORS.xls doc/CoilLocations.pdf)
pathbench_speakers=(F01 F03 F04 M01 M02 M03 M04 M05)
complete_speakers=(F01 F03 F04 FC01 FC02 FC03 M01 M02 M03 M04 M05 MC01 MC02 MC03 MC04)

usage() {
    cat <<'EOF'
Usage: bash scripts/transfer/download_torgo.sh [options]

Options:
    --stage N             First stage to run (default: 0)
    --stop-stage N        Last stage to run (default: 2)
    --download-dir DIR    Archive directory (default: data/torgo/archives)
    --output-dir DIR      Extraction directory (default: data/torgo/raw)
    --base-url URL        TORGO download base URL
    --pathbench-only BOOL Download only F/M dysarthric archives required for
                          PathBench PCC (default: true)
    --help                Show this help

Stages:
    0  Download PathBench-required archives, or all archives when requested
    1  Extract archives (completed archives are skipped on rerun)
    2  Validate the selected speakers and report WAV/prompt counts

The public release is free for academic, non-profit use. Publications must
cite a paper listed on the official TORGO website.
EOF
}

while (( $# )); do
    case "$1" in
        --stage) stage=${2:?"--stage requires a value"}; shift 2 ;;
        --stop-stage) stop_stage=${2:?"--stop-stage requires a value"}; shift 2 ;;
        --download-dir) download_dir=${2:?"--download-dir requires a value"}; shift 2 ;;
        --output-dir) output_dir=${2:?"--output-dir requires a value"}; shift 2 ;;
        --base-url) base_url=${2:?"--base-url requires a value"}; shift 2 ;;
        --pathbench-only) pathbench_only=${2:?"--pathbench-only requires a value"}; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$stage" =~ ^[0-2]$ && "$stop_stage" =~ ^[0-2]$ ]]; then
    echo "--stage and --stop-stage must be integers from 0 through 2." >&2
    exit 2
fi
if (( stage > stop_stage )); then
    echo "--stage ($stage) cannot be greater than --stop-stage ($stop_stage)." >&2
    exit 2
fi
if [[ "$pathbench_only" != true && "$pathbench_only" != false ]]; then
    echo "--pathbench-only must be true or false." >&2
    exit 2
fi

if [[ "$pathbench_only" == true ]]; then
    archives=("${pathbench_archives[@]}")
    expected_speakers=("${pathbench_speakers[@]}")
    extra_downloads=()
else
    archives=("${complete_archives[@]}")
    expected_speakers=("${complete_speakers[@]}")
    extra_downloads=("${supplemental[@]}")
fi

download_file() {
    local remote_path=$1
    local name=${remote_path##*/}
    local destination=$download_dir/$name
    local partial=$destination.part
    local url=${base_url%/}/$remote_path

    if [[ -s "$destination" ]]; then
        echo "Reusing existing file: $destination"
        return
    fi
    if command -v curl >/dev/null 2>&1; then
        # --continue-at resumes a partial file; --retry handles transient failures.
        curl --fail --location --retry 5 --retry-delay 5 \
            --continue-at - --output "$partial" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --tries=5 --waitretry=5 --output-document="$partial" "$url"
    else
        echo "Neither curl nor wget is installed." >&2
        exit 1
    fi
    mv "$partial" "$destination"
}

if (( stage <= 0 && stop_stage >= 0 )); then
    echo "Stage 0: downloading the TORGO public release (pathbench_only=$pathbench_only)"
    mkdir -p "$download_dir"
    for remote_path in "${archives[@]}" "${extra_downloads[@]}"; do
        echo "Downloading $remote_path"
        download_file "$remote_path"
    done
fi

if (( stage <= 1 && stop_stage >= 1 )); then
    echo "Stage 1: extracting archives"
    mkdir -p "$output_dir"
    for name in "${archives[@]}"; do
        archive=$download_dir/$name
        marker=$output_dir/.${name%.tar.bz2}.extracted
        if [[ -f "$marker" ]]; then
            echo "Skipping $name (marker exists: $marker)"
            continue
        fi
        if [[ ! -s "$archive" ]]; then
            echo "Missing archive: $archive" >&2
            echo "Run Stage 0 first." >&2
            exit 1
        fi
        echo "Extracting $name"
        tar -xjf "$archive" -C "$output_dir"
        touch "$marker"
    done
fi

if (( stage <= 2 && stop_stage >= 2 )); then
    echo "Stage 2: validating extracted data"
    missing=()
    for speaker in "${expected_speakers[@]}"; do
        if ! find "$output_dir" -type d -name "$speaker" -print -quit | grep -q .; then
            missing+=("$speaker")
        fi
    done
    if (( ${#missing[@]} )); then
        echo "Missing expected speaker directories: ${missing[*]}" >&2
        exit 1
    fi

    wav_count=$(find "$output_dir" -type f \( -iname '*.wav' \) | wc -l)
    prompt_count=$(find "$output_dir" -type f -path '*/prompts/*' | wc -l)
    phone_count=$(find "$output_dir" -type f \( -iname '*.phn' \) | wc -l)
    echo "Found all ${#expected_speakers[@]} expected speakers."
    echo "WAV files: $wav_count"
    echo "Prompt files: $prompt_count"
    echo "PHN files: $phone_count"
    if (( wav_count == 0 || prompt_count == 0 )); then
        echo "Validation failed: WAV or prompt files were not found." >&2
        exit 1
    fi
    echo "TORGO is ready at: $output_dir"
fi
