#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=rm_rayzer_format
#SBATCH --output=rm_rayzer_format.out
#SBATCH --error=rm_rayzer_format.err

# 10959059 : .
# 77755 : ./glbs
# 4025006 : ./lvsm_with_envmaps
# 68632 : ./lvsm_with_envmaps_rotating_env
# 18631 : ./lvsm_with_envmaps_test_split
# 2689870 : ./rendered
# 2557602 : ./rendered_dense
# 1496139 : ./rendered_dense_lightPlus
# 21408 : ./rendered_dense_v0
# 6210 : ./rendered_previews
# 18638 : ./rendered_test_split
# Remove the rayzer_format directory
target_dir="/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered"
# first create a zip file of the directory
zip -r "$target_dir.zip" "$target_dir"
find "$target_dir" -type f -delete
find "$target_dir" -type d -empty -delete
rmdir "$target_dir"
# rm -rf decoded_result

echo "Removed GSSPLAT intermediate results directory"


# dir="/projects/vig/Datasets/objaverse/hf-objaverse-v1/"
# find -maxdepth 1 -type d | sort | while read -r dir; do n=$(find "$dir" -type f | wc -l); printf "%4d : %s\n" $n "$dir"; done
