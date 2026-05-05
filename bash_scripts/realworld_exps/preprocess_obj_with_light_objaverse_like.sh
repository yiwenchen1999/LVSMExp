#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="/projects/vig/Datasets/obj-with-light/dataset"
OUTPUT_ROOT="/projects/vig/Datasets/obj-with-light/lvsm_format"

python preprocess_scripts/preprocess_obj_with_light_objaverse_like.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"
