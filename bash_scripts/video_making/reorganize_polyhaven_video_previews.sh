#!/usr/bin/env bash
# Wrapper for reorganize_polyhaven_video_previews.py
# Reconstructs chunked supervision_* strips into flat montages + per-frame single images.
# Pass-through args: --base DIR, --num-input-views N, --iters iter_00000001 ..., --dry-run, etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/reorganize_polyhaven_video_previews.py" "$@"

# Examples:
# bash bash_scripts/video_making/reorganize_polyhaven_video_previews.sh --dry-run
# bash bash_scripts/video_making/reorganize_polyhaven_video_previews.sh \
#   --base result_previews/videos/polyhaven
# bash bash_scripts/video_making/reorganize_polyhaven_video_previews.sh \
#   --iters iter_00000001 --no-flattened
