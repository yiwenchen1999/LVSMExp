#!/bin/bash
set -euo pipefail

# Regenerate full_list.txt from existing metadata json files.
# Usage:
#   bash bash_scripts/video_making/northeastern_regenerate_polyhaven_env_rotate_full_list.sh
# Optional override:
#   OUTPUT_DIR=/path/to/dataset SPLIT=test bash ...

OUTPUT_DIR="${OUTPUT_DIR:-/projects/vig/Datasets/objaverse/hf-objaverse-v1/polyhaven_env_rotate}"
SPLIT="${SPLIT:-test}"
METADATA_DIR="$OUTPUT_DIR/$SPLIT/metadata"
FULL_LIST_PATH="$OUTPUT_DIR/$SPLIT/full_list.txt"

if [ ! -d "$METADATA_DIR" ]; then
  echo "ERROR: metadata directory not found: $METADATA_DIR"
  exit 1
fi

python - "$OUTPUT_DIR" "$SPLIT" <<'PY'
import glob
import os
import sys

output_dir = sys.argv[1]
split = sys.argv[2]
metadata_dir = os.path.join(output_dir, split, "metadata")
full_list_path = os.path.join(output_dir, split, "full_list.txt")

json_files = sorted(glob.glob(os.path.join(metadata_dir, "*.json")))
if len(json_files) == 0:
    raise RuntimeError(f"No json files found under {metadata_dir}")

with open(full_list_path, "w") as f:
    for p in json_files:
        f.write(os.path.abspath(p) + "\n")

print(f"Regenerated: {full_list_path}")
print(f"Total scenes: {len(json_files)}")
PY
