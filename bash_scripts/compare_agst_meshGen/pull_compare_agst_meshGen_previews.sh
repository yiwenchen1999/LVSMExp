#!/usr/bin/env bash
# Pull iter_* preview folders from Sony (mfml1) into result_previews/compare_agst_meshGen/<run_name>/.
#
# Defaults (two runs):
#   test_relight_2view
#   test_relight_2view_objaverse
#
# Remote layout:
#   /music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/<run_name>/iter_*
#
# Usage:
#   bash bash_scripts/compare_agst_meshGen/pull_compare_agst_meshGen_previews.sh
#   REMOTE_HOST=mfm1 RUN_NAMES="test_relight_2view" bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-mfml1}"
REMOTE_CKPT_ROOT="${REMOTE_CKPT_ROOT:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}/result_previews/compare_agst_meshGen}"

# Space-separated list of run folder names under ckpt/
RUN_NAMES="${RUN_NAMES:-test_relight_1view_objaverse}"

# Local shell must not expand *; keep it literal for remote expansion.
for RUN_NAME in ${RUN_NAMES}; do
  REMOTE_CKPT_DIR="${REMOTE_CKPT_ROOT}/${RUN_NAME}"
  DEST_DIR="${DEST_ROOT}/${RUN_NAME}"
  mkdir -p "${DEST_DIR}"
  RSYNC_SRC="${REMOTE_HOST}:${REMOTE_CKPT_DIR}/iter_"'*'

  echo "Pulling: ${REMOTE_HOST}:${REMOTE_CKPT_DIR}/iter_*"
  echo "     -> ${DEST_DIR}/"

  rsync -av --human-readable --partial --progress \
    -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
    "${RSYNC_SRC}" \
    "${DEST_DIR}/"
done

echo "Done. Previews under ${DEST_ROOT}"
