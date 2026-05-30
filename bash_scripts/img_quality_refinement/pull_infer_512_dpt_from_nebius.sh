#!/usr/bin/env bash
# Pull yiwen@204.12.169.196 ~/LVSMExp/ckpt/infer_512_dpt/iter_* and reorganize previews.
#
# Usage:
#   bash bash_scripts/img_quality_refinement/pull_infer_512_dpt_from_nebius.sh
#   REMOTE_HOST=yiwen@204.12.169.196 REMOTE_CKPT_DIR=... INFER_NAME=... DEST_DIR=... bash ...
#   PULL_ONLY=1 bash ...
#   NUM_INPUT_VIEWS=2 bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export REMOTE_HOST="${REMOTE_HOST:-yiwen@204.12.169.196}"
export REMOTE_CKPT_DIR="${REMOTE_CKPT_DIR:-/home/yiwen/LVSMExp/ckpt/infer_512_dpt}"

INFER_NAME="${INFER_NAME:-infer_512_dpt}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/result_previews/resolution_comparisons}"
export DEST_DIR="${DEST_DIR:-${BASE_DIR}/${INFER_NAME}}"

PULL_ONLY="${PULL_ONLY:-0}"
NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-4}"

echo "=============================================="
echo "Pulling iter previews from ${REMOTE_HOST}"
echo "REMOTE_CKPT_DIR: ${REMOTE_CKPT_DIR}"
echo "DEST_DIR: ${DEST_DIR}"
echo "=============================================="

bash "${SCRIPT_DIR}/pull_relight_iter_previews_from_sony.sh"

if [[ "${PULL_ONLY}" == "1" ]]; then
  echo "PULL_ONLY=1, skip reorganize."
  exit 0
fi

echo "=============================================="
echo "Reorganizing previews for ${INFER_NAME}"
echo "BASE_DIR: ${BASE_DIR}"
echo "NUM_INPUT_VIEWS: ${NUM_INPUT_VIEWS}"
echo "=============================================="

bash "${SCRIPT_DIR}/reorganize_resolution_comparison_previews.sh" \
  --base "${BASE_DIR}" \
  --infer "${INFER_NAME}" \
  --num-input-views "${NUM_INPUT_VIEWS}"

echo "Done. Organized outputs under ${BASE_DIR}."
