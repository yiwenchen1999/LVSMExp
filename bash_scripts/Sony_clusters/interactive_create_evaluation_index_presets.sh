#!/bin/bash
# Interactive script with presets to create evaluation index JSON files on Sony cluster
# Usage: source bash_scripts/Sony_clusters/interactive_create_evaluation_index_presets.sh
#   or:  bash bash_scripts/Sony_clusters/interactive_create_evaluation_index_presets.sh

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/data,/music-shared-disk"

# Base paths
export BASE_DATA_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse"

############################
# Logging
############################
echo "=============================================="
echo "Create Evaluation Index - Presets (Sony)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "PY_SITE: $PY_SITE"
echo "----------------------------------------------"
echo ""

############################
# Preset Menu
############################
echo "Select a preset configuration:"
echo ""
echo "1) Dense reconstruction (4 input + 8 target, 125 scenes)"
echo "   Full list: lvsm_scenes_dense/test/full_list.txt"
echo "   Output: data/evaluation_index_objaverse_dense.json"
echo ""
echo "2) Test split (4 input + 3 target, all scenes)"
echo "   Full list: lvsm_scenes_dense/test/full_list.txt"
echo "   Output: data/evaluation_index_objaverse_test_4i3o.json"
echo ""
echo "3) Small test (2 input + 6 target, 50 scenes)"
echo "   Full list: lvsm_scenes_dense/test/full_list.txt"
echo "   Output: data/evaluation_index_small_test.json"
echo ""
echo "4) Point light scenes (4 input + 8 target, 100 scenes)"
echo "   Full list: lvsmPlus_objaverse/test/full_list_point_light.txt"
echo "   Output: data/evaluation_index_pointlight.json"
echo ""
echo "5) Custom (enter all parameters manually)"
echo ""
echo "6) Exit"
echo "----------------------------------------------"
read -p "Choice [1-6]: " choice

############################
# Set parameters based on choice
############################
case "$choice" in
  1)
    FULL_LIST="$BASE_DATA_DIR/lvsm_scenes_dense/test/full_list.txt"
    OUTPUT_PATH="$PROJ/data/evaluation_index_objaverse_dense.json"
    N_INPUT=4
    N_TARGET=8
    MIN_FRAME_DIST=25
    MAX_FRAME_DIST=100
    SEED=42
    MAX_SCENES=125
    ;;
  2)
    FULL_LIST="$BASE_DATA_DIR/lvsm_scenes_dense/test/full_list.txt"
    OUTPUT_PATH="$PROJ/data/evaluation_index_objaverse_test_4i3o.json"
    N_INPUT=4
    N_TARGET=3
    MIN_FRAME_DIST=25
    MAX_FRAME_DIST=100
    SEED=42
    MAX_SCENES=""
    ;;
  3)
    FULL_LIST="$BASE_DATA_DIR/lvsm_scenes_dense/test/full_list.txt"
    OUTPUT_PATH="$PROJ/data/evaluation_index_small_test.json"
    N_INPUT=2
    N_TARGET=6
    MIN_FRAME_DIST=25
    MAX_FRAME_DIST=100
    SEED=42
    MAX_SCENES=50
    ;;
  4)
    FULL_LIST="$BASE_DATA_DIR/lvsmPlus_objaverse/test/full_list_point_light.txt"
    OUTPUT_PATH="$PROJ/data/evaluation_index_pointlight.json"
    N_INPUT=4
    N_TARGET=8
    MIN_FRAME_DIST=25
    MAX_FRAME_DIST=100
    SEED=42
    MAX_SCENES=100
    ;;
  5)
    # Custom mode
    echo ""
    echo "Custom configuration mode:"
    read -p "Full list path: " FULL_LIST
    read -p "Output JSON path: " OUTPUT_PATH
    read -p "Number of input frames [4]: " N_INPUT
    N_INPUT=${N_INPUT:-4}
    read -p "Number of target frames [8]: " N_TARGET
    N_TARGET=${N_TARGET:-8}
    read -p "Min frame distance [25]: " MIN_FRAME_DIST
    MIN_FRAME_DIST=${MIN_FRAME_DIST:-25}
    read -p "Max frame distance [100]: " MAX_FRAME_DIST
    MAX_FRAME_DIST=${MAX_FRAME_DIST:-100}
    read -p "Random seed [42]: " SEED
    SEED=${SEED:-42}
    read -p "Max scenes (press Enter for all): " MAX_SCENES
    ;;
  6)
    echo "Exiting."
    exit 0
    ;;
  *)
    echo "Invalid choice."
    exit 1
    ;;
esac

# Build max scenes argument
MAX_SCENES_ARG=""
if [ -n "$MAX_SCENES" ]; then
  MAX_SCENES_ARG="--max-scenes $MAX_SCENES"
fi

############################
# Confirmation
############################
echo ""
echo "=============================================="
echo "Configuration Summary"
echo "=============================================="
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
echo "----------------------------------------------"
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
echo "=============================================="
echo "✓ Evaluation index created successfully!"
echo "=============================================="
echo "Output: $OUTPUT_PATH"
echo ""
echo "You can now use this index in your inference scripts with:"
echo "  inference.view_idx_file_path = \"$OUTPUT_PATH\""
echo ""
