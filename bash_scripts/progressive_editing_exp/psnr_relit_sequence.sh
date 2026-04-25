#!/usr/bin/env bash
# PSNR between relit_gt_* and relit_pred_* per sequence index + CSV + plot.
# Uses the local venv from shortcut.sh (lines 2–3).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate
cd "$REPO_ROOT"

exec python data_postprocess/psnr_relit_sequence.py "$@"

# usage:
#   # 默认 root = result_previews/progressive_stability/finetuned_ckpt
#   bash bash_scripts/progressive_editing_exp/psnr_relit_sequence.sh
#   # 自定义根目录 + 输出目录与文件前缀
  bash bash_scripts/progressive_editing_exp/psnr_relit_sequence.sh \
    result_previews/progressive_stability/latent_space_progression_cycledTraining \
    -o result_previews/progressive_stability/latent_space_progression_cycledTraining/relit_psnr_metrics \
    --basename objaverse_relit_psnr
#   # 仅从已有 CSV 重绘曲线（默认 PNG 与 CSV 同目录、同名 .png）
#   bash bash_scripts/progressive_editing_exp/psnr_relit_sequence.sh \
#     --from-csv result_previews/progressive_stability/og_ckpt_objaverse/relit_psnr_metrics/objaverse_relit_psnr.csv
#   bash bash_scripts/progressive_editing_exp/psnr_relit_sequence.sh --help
