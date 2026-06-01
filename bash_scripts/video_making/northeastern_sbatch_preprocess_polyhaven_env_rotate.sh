#!/bin/bash
#SBATCH --job-name=preprocess_polyhaven_env_rotate
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Resolve project root from this script location:
# bash_scripts/video_making -> project root

python preprocess_scripts/preprocess_objaverse_env_variations.py \
    --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense_polyhaven \
    --output /projects/vig/Datasets/objaverse/hf-objaverse-v1/polyhaven_env_rotate \
    --split test \
    --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs \
    --n-variations 36 \
    --scene-list metadata/polyhaven_env_rotate.json
