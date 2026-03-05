#!/bin/bash
# Compute PSNR, SSIM, LPIPS for GT (black bg, masked) vs NVDiff for all scene pairs.
# Usage:
#   bash scripts/eval_all_gt_vs_nvdiff.sh [--tone-map] [--masked] [-o results.json]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$REPO/result_previews/eval/gt_samples" ]]; then
  GT_SAMPLES="$REPO/result_previews/eval/gt_samples"
  NVDIFF="$REPO/result_previews/eval/NVDiff"
else
  GT_SAMPLES="$REPO/eval/gt_samples"
  NVDIFF="$REPO/eval/NVDiff"
fi
if [[ -d "${1:-}" && -d "${2:-}" ]]; then
  GT_SAMPLES="$1"
  NVDIFF="$2"
  shift 2
fi
exec python "$REPO/scripts/eval_gt_vs_nvdiff.py" "$GT_SAMPLES" "$NVDIFF" --all "$@"
