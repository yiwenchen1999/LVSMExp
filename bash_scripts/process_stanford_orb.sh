#!/usr/bin/env bash
set -euo pipefail

# Use the venv from shortcut.sh (lines 2-4)
cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate

cd /Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp

python preprocess_scripts/preprocess_stanford_orb.py \
  --input-root data_samples/stanford_ORB \
  --output-root data_samples/stanford_ORB_processed \
  --split both \
  --target-size 512 \
  --target-fov 30
