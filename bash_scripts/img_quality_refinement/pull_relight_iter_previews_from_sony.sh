#!/usr/bin/env bash
# Pull remote .../ckpt/<run>/iter_* only → repo/result_previews/resolution_comparisons
# Env: REMOTE_HOST (default mfml1), REMOTE_CKPT_DIR, DEST_DIR — prefix on the command or export.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${DEST_DIR:-${REPO_ROOT}/result_previews/resolution_comparisons}"
REMOTE_HOST="${REMOTE_HOST:-mfml1}"
REMOTE_CKPT_DIR="${REMOTE_CKPT_DIR:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/relight_result_512}"

mkdir -p "${DEST_DIR}"

# Literal * at end: local shell must not glob; remote expands iter_*.
RSYNC_SRC="${REMOTE_HOST}:${REMOTE_CKPT_DIR}/iter_"'*'

# macOS rsync 2.6.9: use --human-readable, not -h (conflicts with --help).
rsync -av --human-readable --partial --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  "${RSYNC_SRC}" \
  "${DEST_DIR}"
