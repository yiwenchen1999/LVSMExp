#!/usr/bin/env bash
# End-to-end helper:
#   1) pull iter_* previews from northeastern-fileTransfer
#   2) generate flattened + single_image folders
#   3) compute overall PSNR / SSIM (LPIPS skipped by default)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/pull_realworld_eval_previews.sh" "$@"
bash "${SCRIPT_DIR}/reorganize_realworld_eval_previews.sh"
bash "${SCRIPT_DIR}/compute_realworld_eval_metrics.sh"
