#!/usr/bin/env bash
# Pull ckpt result dirs from Sony (mfml1) and build flattened + single_image layouts
# (same convention as result_previews/resolution_comparisons).
#
# Default runs:
#   test_relight_2view
#   test_relight_2view_objaverse
#
# Mesh-gen / 2-view Obj configs use training.num_input_views = 2 (override with NUM_INPUT_VIEWS).
#
# Usage:
#   bash bash_scripts/compare_agst_meshGen/pull_and_reorganize_meshgen_previews.sh
#   REMOTE_HOST=mfml1 SKIP_RSYNC=1 bash ...   # only reorganize (data already local)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-mfml1}"
REMOTE_CKPT_BASE="${REMOTE_CKPT_BASE:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt}"
DEST_DIR="${DEST_DIR:-${REPO_ROOT}/result_previews/compare_agst_meshGen}"
NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-2}"

# Space-separated; export RUN_NAMES="foo bar" to override.
RUN_NAMES_DEFAULT=(test_relight_2view test_relight_2view_objaverse)
RUN_NAMES=(${RUN_NAMES:-})

SSH_OPTS=(ssh
  -o ServerAliveInterval=60
  -o ServerAliveCountMax=3
  -o TCPKeepAlive=yes
)

if [[ ${#RUN_NAMES[@]} -eq 0 ]]; then
  RUN_NAMES=("${RUN_NAMES_DEFAULT[@]}")
fi

mkdir -p "${DEST_DIR}"

if [[ "${SKIP_RSYNC:-0}" != "1" ]]; then
  for name in "${RUN_NAMES[@]}"; do
    echo "=== rsync ${REMOTE_HOST}:${REMOTE_CKPT_BASE}/${name}/ -> ${DEST_DIR}/${name}/"
    mkdir -p "${DEST_DIR}/${name}"
    rsync -av --human-readable --partial --progress \
      -e "${SSH_OPTS[*]}" \
      "${REMOTE_HOST}:${REMOTE_CKPT_BASE}/${name}/" \
      "${DEST_DIR}/${name}/"
  done
else
  echo "SKIP_RSYNC=1: skipping pull"
fi

INFER_ARGS=()
for name in "${RUN_NAMES[@]}"; do
  INFER_ARGS+=("${name}")
done

echo "=== reorganize (flattened + single_image)"
python3 "${REPO_ROOT}/bash_scripts/img_quality_refinement/reorganize_resolution_comparison_previews.py" \
  --base "${DEST_DIR}" \
  --infer "${INFER_ARGS[@]}" \
  --num-input-views "${NUM_INPUT_VIEWS}"

echo "Done. Output root: ${DEST_DIR}"
