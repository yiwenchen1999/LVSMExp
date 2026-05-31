#!/usr/bin/env bash
# Wrapper for make_polyhaven_pair_and_grid_videos.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/make_polyhaven_pair_and_grid_videos.py" "$@"

# Examples:
# bash bash_scripts/video_making/make_polyhaven_pair_and_grid_videos.sh --dry-run
# bash bash_scripts/video_making/make_polyhaven_pair_and_grid_videos.sh --seed 42 --overwrite
