#!/bin/bash
# Compute PSNR, SSIM, LPIPS for GT (white bg) vs TensoIR (*_nobg.png) for all scene pairs.
# Usage:
#   bash scripts/eval_all_gt_vs_tensoIR.sh [--no-lpips] [-o results.json]
#   bash scripts/eval_all_gt_vs_tensoIR.sh /path/to/gt_samples /path/to/TensoIR [options]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$REPO/result_previews/eval/gt_samples" ]]; then
  GT_SAMPLES="$REPO/result_previews/eval/gt_samples"
  TENSOIR="$REPO/result_previews/eval/TensoIR"
else
  GT_SAMPLES="$REPO/eval/gt_samples"
  TENSOIR="$REPO/eval/TensoIR"
fi
if [[ -d "${1:-}" && -d "${2:-}" ]]; then
  GT_SAMPLES="$1"
  TENSOIR="$2"
  shift 2
fi
exec python "$REPO/scripts/eval_gt_vs_tensoIR.py" "$GT_SAMPLES" "$TENSOIR" --all "$@"
