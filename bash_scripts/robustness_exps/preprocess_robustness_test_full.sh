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
OUTPUT_ROOT=/projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsmPlus_objaverse_robustTest
# Tar mirror (location B) disabled via --no-output-tar in python calls below.
# To re-enable: set a path and pass --output-tar "${OUTPUT_TAR_ROOT}" instead of --no-output-tar.
# OUTPUT_TAR_ROOT=/scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse_robustTest_tar
SPLIT=test
HDRI_DIR=/projects/vig/Datasets/objaverse/envmaps_256/hdirs

INPUT_DIRS=(
  "${BASE_INPUT}/rendered_dense_robustTest_far"
  "${BASE_INPUT}/rendered_dense_robustTest_near"
  "${BASE_INPUT}/rendered_dense_robustTest_normal"
)

for input_dir in "${INPUT_DIRS[@]}"; do
  echo "Processing robustness source: ${input_dir}"
  python preprocess_scripts/preprocess_objaverse.py \
    --input "${input_dir}" \
    --output "${OUTPUT_ROOT}" \
    --no-output-tar \
    --split "${SPLIT}" \
    --hdri-dir "${HDRI_DIR}"
done

# Regenerate an integrated list after all sources are merged.
python preprocess_scripts/preprocess_objaverse.py \
  --output "${OUTPUT_ROOT}" \
  --no-output-tar \
  --split "${SPLIT}" \
  --full-list-only
