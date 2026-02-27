#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess_test_short_list
#SBATCH --mem=32
#SBATCH --ntasks=16
#SBATCH --output=myjob.preprocess_test.out
#SBATCH --error=myjob.preprocess_test.err


python preprocess_scripts/preprocess_objaverse.py \
  --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_scenes \
  --output /scratch/chen.yiwe/temp_objaverse/lvsm_scenes \
  --output-tar /scratch/chen.yiwe/temp_objaverse/lvsm_scenes_tar \
  --split test \
  --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs 
