#!/usr/bin/env bash
# Wrapper for reorganize_failurecases_vid_previews.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/reorganize_failurecases_vid_previews.py" "$@"

# Examples:
# bash bash_scripts/scene/reorganize_failurecases_vid_previews.sh --dry-run
# bash bash_scripts/scene/reorganize_failurecases_vid_previews.sh --iters iter_00000001 iter_00000002
