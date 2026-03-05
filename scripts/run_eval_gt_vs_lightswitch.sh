#!/bin/bash
# Run eval_gt_vs_lightswitch_masked.py with local venv (from shortcut.sh)
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /Users/yiwenchen/Desktop/ResearchProjects/scripts/venv/bin/activate
cd "$REPO"
python scripts/eval_gt_vs_lightswitch_masked.py "$@"
