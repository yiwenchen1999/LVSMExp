#!/bin/bash
# Compute PSNR, SSIM, LPIPS for all scene pairs in eval/gt_samples vs eval/lightSwitch.
# Usage:
#   bash scripts/eval_all_gt_vs_lightswitch.sh [--tone-map] [-o results.json]
#   bash scripts/eval_all_gt_vs_lightswitch.sh /path/to/gt_samples /path/to/lightSwitch [--tone-map] [-o out.json]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GT_SAMPLES="$REPO/result_previews/eval/gt_samples"
LIGHT_SWITCH="$REPO/result_previews/eval/lightSwitch"
if [[ -d "${1:-}" && -d "${2:-}" ]]; then
  GT_SAMPLES="$1"
  LIGHT_SWITCH="$2"
  shift 2
fi
exec python "$REPO/scripts/eval_gt_vs_lightswitch_masked.py" "$GT_SAMPLES" "$LIGHT_SWITCH" --all "$@"
