#!/usr/bin/env bash
# Wrapper for make_failurecases_vid_gt_pred_pair.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/make_failurecases_vid_gt_pred_pair.py" "$@"

# Examples:
# bash bash_scripts/scene/make_failurecases_vid_gt_pred_pair.sh --dry-run
# bash bash_scripts/scene/make_failurecases_vid_gt_pred_pair.sh --fps 24 --overwrite
