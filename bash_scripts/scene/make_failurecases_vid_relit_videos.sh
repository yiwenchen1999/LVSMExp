#!/usr/bin/env bash
# Wrapper for make_failurecases_vid_relit_videos.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/make_failurecases_vid_relit_videos.py" "$@"

# Examples:
# bash bash_scripts/scene/make_failurecases_vid_relit_videos.sh --dry-run
# bash bash_scripts/scene/make_failurecases_vid_relit_videos.sh --iters iter_00000001 --overwrite
