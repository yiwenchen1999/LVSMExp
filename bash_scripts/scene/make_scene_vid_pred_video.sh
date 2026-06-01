#!/usr/bin/env bash
# Wrapper for make_scene_vid_pred_video.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/make_scene_vid_pred_video.py" "$@"

# Examples:
# bash bash_scripts/scene/make_scene_vid_pred_video.sh --dry-run
# bash bash_scripts/scene/make_scene_vid_pred_video.sh --iters iter_00000001 --overwrite
