#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=preprocess_robustness_test
#SBATCH --mem=32
#SBATCH --ntasks=16
#SBATCH --output=myjob.preprocess_robustness_test.out
#SBATCH --error=myjob.preprocess_robustness_test.err

set -euo pipefail

BASE_INPUT=/projects/vig/Datasets/objaverse/hf-objaverse-v1
OUTPUT_BASE=/projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsmPlus_objaverse_robustTest
SPLIT=test
HDRI_DIR=/projects/vig/Datasets/objaverse/envmaps_256/hdirs

ROBUST_TAGS=(far near normal)

for tag in "${ROBUST_TAGS[@]}"; do
  input_dir="${BASE_INPUT}/rendered_dense_robustTest_${tag}"
  output_dir="${OUTPUT_BASE}_${tag}"
  echo "Processing robustness source: ${input_dir}"
  echo "Output directory: ${output_dir}"
  python preprocess_scripts/preprocess_objaverse.py \
    --input "${input_dir}" \
    --output "${output_dir}" \
    --no-output-tar \
    --split "${SPLIT}" \
    --hdri-dir "${HDRI_DIR}"

  # Regenerate full_list for each robustness bucket separately.
  python preprocess_scripts/preprocess_objaverse.py \
    --output "${output_dir}" \
    --no-output-tar \
    --split "${SPLIT}" \
    --full-list-only
done

# Merge full_list/metadata from 3 robustness buckets into one list for training.
python preprocess_scripts/merge_robustness_full_list.py \
  --input-roots "${OUTPUT_BASE}_far" "${OUTPUT_BASE}_near" "${OUTPUT_BASE}_normal" \
  --output-root "${OUTPUT_BASE}_merged" \
  --split "${SPLIT}" \
  --prefer-image-tar
