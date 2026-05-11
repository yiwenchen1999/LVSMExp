#!/usr/bin/env bash
# Pull mfml1 .../ckpt/infer_512_dpt/iter_* → repo/result_previews/resolution_comparisons/infer_512_dpt
# Reuses pull_relight_iter_previews_from_sony.sh (same rsync pattern as infer_256).
#
# Usage:
#   bash bash_scripts/img_quality_refinement/pull_infer_512_dpt_previews_from_sony.sh
#   REMOTE_HOST=mfml1 REMOTE_CKPT_DIR=... DEST_DIR=... bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export REMOTE_HOST="${REMOTE_HOST:-mfml1}"
export REMOTE_CKPT_DIR="${REMOTE_CKPT_DIR:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/infer_512_dpt}"
export DEST_DIR="${DEST_DIR:-${REPO_ROOT}/result_previews/resolution_comparisons/infer_512_dpt}"

exec bash "${SCRIPT_DIR}/pull_relight_iter_previews_from_sony.sh"
