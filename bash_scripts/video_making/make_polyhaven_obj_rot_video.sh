#!/usr/bin/env bash
# Wrapper for make_polyhaven_obj_rot_video.py
# Build per-scene/per-iter mp4 videos from frame_*/view_relit_pred.jpg.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/make_polyhaven_obj_rot_video.py" "$@"

# Examples:
# bash bash_scripts/video_making/make_polyhaven_obj_rot_video.sh --dry-run
# bash bash_scripts/video_making/make_polyhaven_obj_rot_video.sh --fps 24
# bash bash_scripts/video_making/make_polyhaven_obj_rot_video.sh --iters iter_00000001 --overwrite
