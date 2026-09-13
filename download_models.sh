#!/usr/bin/env bash
# Download all Lineart_Painter model weights via setup_models.py
# Requires: pip install huggingface_hub
# Existing files are skipped automatically (supports resumable download).
# Usage:
#   ./download_models.sh                      # download all optional models
#   ./download_models.sh --only sd15 cn_anime # download only the listed items
#   ./download_models.sh --skip midas animatediff   # skip the listed items
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/setup_models.py" "$@"
