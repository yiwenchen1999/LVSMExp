#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess_train_short_list
#SBATCH --mem=32
#SBATCH --ntasks=16
#SBATCH --output=myjob.preprocess_train.out
#SBATCH --error=myjob.preprocess_train.err


python preprocess_scripts/preprocess_objaverse.py \
  --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense_lightPlus \
  --output /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse \
  --output-tar /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsmPlus_objaverse_tar \
  --split train \
  --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs 
