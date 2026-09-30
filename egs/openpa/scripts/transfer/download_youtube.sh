#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
. "$repo_dir/path.sh"

# PathBench oral-cancer YouTube evaluation-set downloader.
# Stages:
#   0: download the official Zenodo archive
#   1: extract the archive
#   2: report the extracted audio-file counts

stage=0
stop_stage=2
download_dir=data/youtube/archives
output_dir=data/youtube/raw
archive_name=youtube_utterances.zip
archive_url=https://zenodo.org/api/records/18738598/files/youtube_utterances.zip/content

usage() {
    cat <<'EOF'
Usage: bash scripts/transfer/download_youtube.sh [options]

Options:
    --stage N             First stage to run (default: 0)
    --stop-stage N        Last stage to run (default: 2)
    --download-dir DIR    Archive directory (default: data/youtube/archives)
    --output-dir DIR      Extraction directory (default: data/youtube/raw)
    --archive-url URL     Override the official Zenodo archive URL
    --help                Show this help

Stages:
    0  Download the PathBench oral-cancer YouTube utterance archive
    1  Extract the archive (an existing completion marker is respected)
    2  Report extracted WAV/audio counts

The download is about 223 MB. This script intentionally does not verify or
checksum the compressed archive.
EOF
}

while (( $# )); do
    case "$1" in
        --stage) stage=${2:?"--stage requires a value"}; shift 2 ;;
        --stop-stage) stop_stage=${2:?"--stop-stage requires a value"}; shift 2 ;;
        --download-dir) download_dir=${2:?"--download-dir requires a value"}; shift 2 ;;
        --output-dir) output_dir=${2:?"--output-dir requires a value"}; shift 2 ;;
        --archive-url) archive_url=${2:?"--archive-url requires a value"}; shift 2 ;;
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

download_file() {
    local destination=$download_dir/$archive_name
    local partial=$destination.part

    if [[ -s "$destination" ]]; then
        echo "Reusing existing file: $destination"
        return
    fi
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 5 --retry-delay 5 \
            --continue-at - --output "$partial" "$archive_url"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --tries=5 --waitretry=5 \
            --output-document="$partial" "$archive_url"
    else
        echo "Neither curl nor wget is installed." >&2
        exit 1
    fi
    mv "$partial" "$destination"
}

if (( stage <= 0 && stop_stage >= 0 )); then
    echo "Stage 0: downloading the oral-cancer YouTube corpus"
    mkdir -p "$download_dir"
    download_file
fi

if (( stage <= 1 && stop_stage >= 1 )); then
    echo "Stage 1: extracting the corpus"
    archive=$download_dir/$archive_name
    marker=$output_dir/.youtube_utterances.extracted
    if [[ -f "$marker" ]]; then
        echo "Skipping extraction (marker exists: $marker)"
    else
        if [[ ! -s "$archive" ]]; then
            echo "Missing archive: $archive" >&2
            echo "Run Stage 0 first." >&2
            exit 1
        fi
        if ! command -v unzip >/dev/null 2>&1; then
            echo "The unzip command is required for Stage 1." >&2
            exit 1
        fi
        mkdir -p "$output_dir"
        # -o makes a rerun continue cleanly after an interrupted extraction.
        unzip -oq "$archive" -d "$output_dir"
        touch "$marker"
    fi
fi

if (( stage <= 2 && stop_stage >= 2 )); then
    echo "Stage 2: summarizing extracted data"
    if [[ ! -d "$output_dir" ]]; then
        echo "Missing extraction directory: $output_dir" >&2
        echo "Run Stage 1 first." >&2
        exit 1
    fi
    wav_count=$(find "$output_dir" -type f -iname '*.wav' | wc -l)
    audio_count=$(find "$output_dir" -type f \( \
        -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' -o -iname '*.m4a' \
        \) | wc -l)
    echo "WAV files: $wav_count"
    echo "All recognized audio files: $audio_count"
    echo "YouTube oral-cancer corpus is ready at: $output_dir"
fi
