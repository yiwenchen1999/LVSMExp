#!/usr/bin/env bash
# PSNR / SSIM / LPIPS for relit_gt vs relit_pred under single_image/infer_*
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/compute_single_image_relit_metrics.py" "$@"
