#!/usr/bin/env bash
# Wrapper for reorganize_scene_vid_previews_input_gt_pred.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/reorganize_scene_vid_previews_input_gt_pred.py" "$@"

# Examples:
# bash bash_scripts/scene/reorganize_scene_vid_previews_input_gt_pred.sh --dry-run
# bash bash_scripts/scene/reorganize_scene_vid_previews_input_gt_pred.sh --base result_previews/polyhaven_all
