#!/usr/bin/env bash
set -euo pipefail

# Use the same local venv setup as shortcut.sh (lines 2-4).
cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate

cd /Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp

# Explicit input/output paths.
INPUT_ROOT="/projects/vig/Datasets/stanfordORB/blender_LDR"
OUTPUT_ROOT="/projects/vig/Datasets/stanfordORB/lvsm_stanford_orb"

python preprocess_scripts/preprocess_stanford_orb_objaverse_like.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --target-size 512 \
  --no-adjust-fov
