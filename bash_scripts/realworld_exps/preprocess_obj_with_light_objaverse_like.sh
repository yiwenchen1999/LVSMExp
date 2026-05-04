#!/usr/bin/env bash
set -euo pipefail

# Use the same local venv setup as shortcut.sh (lines 2-4).
cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate

cd /Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp

INPUT_ROOT="/Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp/data_samples/obj_with_light"
OUTPUT_ROOT="/Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp/data_samples/processed_obj_with_light_objaverse_like"

python preprocess_scripts/preprocess_obj_with_light_objaverse_like.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"
