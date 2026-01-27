#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=rm_rayzer_format
#SBATCH --output=rm_rayzer_format.out
#SBATCH --error=rm_rayzer_format.err

# Remove the rayzer_format directory
target_dir="gaffer_render_postprocessed_obj"
find "$target_dir" -type f -delete
find "$target_dir" -type d -empty -delete
rmdir "$target_dir"
# rm -rf decoded_result

echo "Removed GSSPLAT intermediate results directory"


# dir="/projects/vig/Datasets/objaverse/hf-objaverse-v1/"
# find -maxdepth 1 -type d | sort | while read -r dir; do n=$(find "$dir" -type f | wc -l); printf "%4d : %s\n" $n "$dir"; done
