#!/bin/bash
# Interactive script to create evaluation index JSON files on Sony cluster
# Usage: source bash_scripts/Sony_clusters/interactive_create_evaluation_index.sh
#   or:  bash bash_scripts/Sony_clusters/interactive_create_evaluation_index.sh

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

# Default paths
export BASE_DATA_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse"
export DEFAULT_FULL_LIST="$BASE_DATA_DIR/polyhaven_lvsm/test/full_list.txt"
export DEFAULT_OUTPUT="$PROJ/data/evaluation_index_polyhaven_dense.json"

############################
# Logging
############################
echo "========================================"
echo "Create Evaluation Index (Sony Cluster)"
echo "========================================"
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "PY_SITE: $PY_SITE"
echo "----------------------------------------"

############################
# Interactive prompts
############################
echo ""
echo "Please configure the evaluation index parameters:"
echo ""

# Full list path
read -p "Full list path [$DEFAULT_FULL_LIST]: " FULL_LIST
FULL_LIST=${FULL_LIST:-$DEFAULT_FULL_LIST}

# Output path
read -p "Output JSON path [$DEFAULT_OUTPUT]: " OUTPUT_PATH
OUTPUT_PATH=${OUTPUT_PATH:-$DEFAULT_OUTPUT}

# Number of input frames
read -p "Number of input frames [4]: " N_INPUT
N_INPUT=${N_INPUT:-4}

# Number of target frames
read -p "Number of target frames [8]: " N_TARGET
N_TARGET=${N_TARGET:-8}

# Min frame distance
read -p "Min frame distance [25]: " MIN_FRAME_DIST
MIN_FRAME_DIST=${MIN_FRAME_DIST:-25}

# Max frame distance
read -p "Max frame distance [100]: " MAX_FRAME_DIST
MAX_FRAME_DIST=${MAX_FRAME_DIST:-100}

# Random seed
read -p "Random seed [42]: " SEED
SEED=${SEED:-42}

# Max scenes (optional)
read -p "Max scenes (press Enter for all): " MAX_SCENES
MAX_SCENES_ARG=""
if [ -n "$MAX_SCENES" ]; then
  MAX_SCENES_ARG="--max-scenes $MAX_SCENES"
fi

############################
# Confirmation
############################
echo ""
echo "========================================"
echo "Configuration Summary"
echo "========================================"
echo "Full list:        $FULL_LIST"
echo "Output path:      $OUTPUT_PATH"
echo "Input frames:     $N_INPUT"
echo "Target frames:    $N_TARGET"
echo "Min frame dist:   $MIN_FRAME_DIST"
echo "Max frame dist:   $MAX_FRAME_DIST"
echo "Random seed:      $SEED"
if [ -n "$MAX_SCENES" ]; then
  echo "Max scenes:       $MAX_SCENES"
else
  echo "Max scenes:       (all)"
fi
echo "----------------------------------------"
echo ""

read -p "Proceed with these settings? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

############################
# Run create_evaluation_index.py
############################
echo ""
echo "Creating evaluation index..."
singularity exec $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:\${PYTHONPATH:-}\"
  cd $PROJ

  python preprocess_scripts/create_evaluation_index.py \
    --full-list \"$FULL_LIST\" \
    --output \"$OUTPUT_PATH\" \
    --n-input $N_INPUT \
    --n-target $N_TARGET \
    --min-frame-dist $MIN_FRAME_DIST \
    --max-frame-dist $MAX_FRAME_DIST \
    --seed $SEED \
    $MAX_SCENES_ARG
"

############################
# Done
############################
echo ""
echo "========================================"
echo "Evaluation index created successfully!"
echo "========================================"
echo "Output: $OUTPUT_PATH"
echo ""
echo "You can now use this index in your inference scripts with:"
echo "  inference.view_idx_file_path = \"$OUTPUT_PATH\""
echo ""
