#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=download_polyhaven
#SBATCH --output=download_polyhaven.out
#SBATCH --error=download_polyhaven.err

python download_polyhaven.py