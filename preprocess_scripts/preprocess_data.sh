#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=preprocess_data
#SBATCH --mem=32G
#SBATCH --ntasks=32
#SBATCH --output=preprocess_data.out
#SBATCH --error=preprocess_data.err


python preprocess_scripts/preprocess_data.py --base_path re10k --output_dir re10k_processed --mode test