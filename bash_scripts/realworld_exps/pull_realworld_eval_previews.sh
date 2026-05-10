#!/usr/bin/env bash
# Pull iter_* preview folders from northeastern-fileTransfer for realworld eval.
#
# Defaults are set for:
#   /projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/test_relight_stanfordORB
# and destination:
#   result_previews/realworld_eval/test_relight_stanfordORB
#
# Usage examples:
#   bash bash_scripts/realworld_exps/pull_realworld_eval_previews.sh
#   REMOTE_CKPT_DIR=/remote/ckpt/test_xxx RUN_NAME=test_xxx \
#     bash bash_scripts/realworld_exps/pull_realworld_eval_previews.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-northeastern-fileTransfer}"
RUN_NAME="${RUN_NAME:-test_relight_stanfordORB}"
REMOTE_CKPT_DIR="${REMOTE_CKPT_DIR:-/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/${RUN_NAME}}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}/result_previews/realworld_eval}"
DEST_DIR="${DEST_DIR:-${DEST_ROOT}/${RUN_NAME}}"

mkdir -p "${DEST_DIR}"

# Local shell must not expand *; keep it literal for remote expansion.
RSYNC_SRC="${REMOTE_HOST}:${REMOTE_CKPT_DIR}/iter_"'*'

echo "Pulling from: ${REMOTE_HOST}:${REMOTE_CKPT_DIR}"
echo "Destination : ${DEST_DIR}"

rsync -av --human-readable --partial --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  "${RSYNC_SRC}" \
  "${DEST_DIR}"

echo "Done. Pulled previews to ${DEST_DIR}"
