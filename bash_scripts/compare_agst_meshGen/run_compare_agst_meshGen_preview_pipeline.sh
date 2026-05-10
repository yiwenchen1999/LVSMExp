#!/usr/bin/env bash
# 1) Pull iter_* from mfml1 -> result_previews/compare_agst_meshGen/<run>/
# 2) Flatten + single_image (same layout as realworld_exps preview pipeline)
#
# Env (optional):
#   REMOTE_HOST=mfml1
#   REMOTE_CKPT_ROOT=/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt
#   DEST_ROOT=.../result_previews/compare_agst_meshGen
#   RUN_NAMES="test_relight_2view test_relight_2view_objaverse"
#
# Extra args are forwarded only to the reorganize step (e.g. --dry-run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If first arg looks like reorganize-only flag, only reorganize.
if [[ "${1:-}" == "--reorganize-only" ]]; then
  shift
  bash "${SCRIPT_DIR}/reorganize_compare_agst_meshGen_previews.sh" "$@"
  exit 0
fi

bash "${SCRIPT_DIR}/pull_compare_agst_meshGen_previews.sh"

# Forward remaining args to reorganize
bash "${SCRIPT_DIR}/reorganize_compare_agst_meshGen_previews.sh" "$@"
