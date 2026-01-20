#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=rm_rayzer_format

# Remove the rayzer_format directory
cd /projects/vig/Datasets/objaverse/hf-objaverse-v1/
rm -rf rayzer_states_temp
rm -rf rayzer_states
rm -rf rayzer_format

echo "Removed rayzer_format directory"

