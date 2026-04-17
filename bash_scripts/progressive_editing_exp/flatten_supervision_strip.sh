#!/usr/bin/env bash
# Flatten supervision_* horizontal strips into per-view folders and merged PNGs.
# Uses the local venv from shortcut.sh (lines 2–3).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate
cd "$REPO_ROOT"

exec python data_postprocess/flatten_progressive_supervision.py "$@"

#usage:
# bash bash_scripts/progressive_editing_exp/flatten_supervision_strip.sh result_previews/progressive_stability/finetuned_ckpt

# 视角数 n（任选一种：-n / --num-views / --n-views）
# bash bash_scripts/progressive_editing_exp/flatten_supervision_strip.sh -n 8
# bash bash_scripts/progressive_editing_exp/flatten_supervision_strip.sh --num-views 4
# # 只检查、不写文件
# bash bash_scripts/progressive_editing_exp/flatten_supervision_strip.sh --dry-run
# # 瓦片尺寸（默认 256）
# bash bash_scripts/progressive_editing_exp/flatten_supervision_strip.sh --tile-w 256 --tile-h 256
# # 组合：目录 + 视角数 + dry-run
# bash bash_scripts/progressive_editing_exp/flatten_supervision_strip.sh \
#   result_previews/progressive_stability/finetuned_ckpt \
#   -n 8 \
#   --dry-run
