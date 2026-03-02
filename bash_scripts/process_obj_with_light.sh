#!/usr/bin/env bash
set -euo pipefail

# Use the venv from shortcut.sh
cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate

cd /Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp

python preprocess_scripts/preprocess_obj_with_light.py \
  --input-root data_samples/obj_with_light \
  --output-root data_samples/obj_with_light_processed \
  --split test \
  --target-size 512 \
  --target-fov 30
