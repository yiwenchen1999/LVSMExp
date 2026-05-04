#!/usr/bin/env bash
set -euo pipefail

# Explicit input/output paths.
INPUT_ROOT="/projects/vig/Datasets/stanfordORB/blender_LDR"
OUTPUT_ROOT="/projects/vig/Datasets/stanfordORB/lvsm_stanford_orb"

python preprocess_scripts/preprocess_stanford_orb_objaverse_like.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --target-size 512 \
  --no-adjust-fov
