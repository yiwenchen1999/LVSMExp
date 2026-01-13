#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess_test
#SBATCH --mem=32
#SBATCH --ntasks=16
#SBATCH --output=myjob.preprocess_test.out
#SBATCH --error=myjob.preprocess_test.err


python preprocess_scripts/preprocess_objaverse.py \
  --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense \
  --output /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps \
  --split test \
  --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs 
