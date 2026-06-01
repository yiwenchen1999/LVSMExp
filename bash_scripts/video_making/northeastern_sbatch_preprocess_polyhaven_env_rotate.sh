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

PROJ="/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp"
INPUT_DIR="/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense_polyhaven"
OUTPUT_DIR="/projects/vig/Datasets/objaverse/hf-objaverse-v1/polyhaven_env_rotate"
SPLIT="test"
SCENE_LIST="$PROJ/metadata/polyhaven_env_rotate.json"

cd "$PROJ"

python preprocess_scripts/preprocess_objaverse_env_variations.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs \
    --n-variations 36 \
    --scene-list "$SCENE_LIST"

# Safety net: some historical runs did not leave full_list.txt; rebuild from metadata.
FULL_LIST_PATH="$OUTPUT_DIR/$SPLIT/full_list.txt"
if [ ! -s "$FULL_LIST_PATH" ]; then
  echo "full_list.txt missing or empty, rebuilding from metadata json files..."
  python - <<'PY'
import glob
import os

output_dir = "/projects/vig/Datasets/objaverse/hf-objaverse-v1/polyhaven_env_rotate"
split = "test"
metadata_dir = os.path.join(output_dir, split, "metadata")
full_list = os.path.join(output_dir, split, "full_list.txt")

if not os.path.isdir(metadata_dir):
    raise RuntimeError(f"metadata directory not found: {metadata_dir}")

json_files = sorted(glob.glob(os.path.join(metadata_dir, "*.json")))
if len(json_files) == 0:
    raise RuntimeError(f"no metadata json found under: {metadata_dir}")

with open(full_list, "w") as f:
    for p in json_files:
        f.write(os.path.abspath(p) + "\n")

print(f"Rebuilt {full_list} with {len(json_files)} scenes")
PY
fi

echo "Done. full_list: $FULL_LIST_PATH"
