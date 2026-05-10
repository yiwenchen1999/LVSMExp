#!/usr/bin/env bash
# Compute overall relight metrics for realworld eval previews.
# Outputs:
#   result_previews/realworld_eval/single_image/metrics_relit_<run_name>.csv
#   result_previews/realworld_eval/single_image/metrics_relit_<run_name>_summary.json
#
# Default computes PSNR/SSIM only (LPIPS skipped).
#
# Usage:
#   bash bash_scripts/realworld_exps/compute_realworld_eval_metrics.sh
#   RUN_NAME=test_relight_obj-with-light \
#     bash bash_scripts/realworld_exps/compute_realworld_eval_metrics.sh
#   # include LPIPS:
#   SKIP_LPIPS=0 bash bash_scripts/realworld_exps/compute_realworld_eval_metrics.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_DIR="${BASE_DIR:-${REPO_ROOT}/result_previews/realworld_eval/single_image}"
RUN_NAME="${RUN_NAME:-test_relight_stanfordORB}"
SKIP_LPIPS="${SKIP_LPIPS:-1}"

if python3 -c "import torch" >/dev/null 2>&1; then
  EXTRA_ARGS=()
  if [[ "${SKIP_LPIPS}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-lpips)
  fi
  python3 "${REPO_ROOT}/bash_scripts/img_quality_refinement/compute_single_image_relit_metrics.py" \
    --base "${BASE_DIR}" \
    --infer "${RUN_NAME}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
else
  echo "torch not found; using numpy fallback for PSNR/SSIM."
  python3 "${REPO_ROOT}/bash_scripts/realworld_exps/compute_realworld_eval_metrics_no_torch.py" \
    --base "${BASE_DIR}" \
    --infer "${RUN_NAME}" \
    "$@"
fi
