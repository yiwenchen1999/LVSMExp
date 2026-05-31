#!/bin/bash
#SBATCH --job-name=preprocess_polyhaven_env_rotate
#SBATCH --partition=ct_l40s
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=/group2/ct/yiwen/logs/%x.%N.%j.err

# Preprocess polyhaven scenes into env-rotation variations on the Sony cluster.
# Uses preprocess_scripts/preprocess_objaverse_env_variations.py with the tagged
# scene list metadata/polyhaven_env_rotate.json. Each listed env scene gets
# N_VARIATIONS rotated copies; white_env_0 scenes are processed without variations.

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

# Raw objaverse-format input (object_id/<split>/env_*/gt_*.png) and HDRIs.
export INPUT_DIR="${INPUT_DIR:-/music-shared-disk/group/ct/yiwen/data/objaverse/rendered_dense_polyhaven}"
export HDRI_DIR="${HDRI_DIR:-/music-shared-disk/group/ct/yiwen/data/objaverse/hdris}"

# New rotating-env output dataset (kept separate from polyhaven_lvsm).
export OUTPUT_DIR="${OUTPUT_DIR:-/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm_rotating_env}"

export SCENE_LIST="${SCENE_LIST:-$PROJ/metadata/polyhaven_env_rotate.json}"
export SPLIT="${SPLIT:-test}"
export N_VARIATIONS="${N_VARIATIONS:-36}"
# For limited inference/debugging: sample a shared consecutive frame chunk per object.
# This chunk is reused across scenes with same object id (e.g., Camera_01_env_0/env_1).
export CONSECUTIVE_FRAMES="${CONSECUTIVE_FRAMES:-20}"
export FRAME_CHUNK_SEED="${FRAME_CHUNK_SEED:-777}"

############################
# Logging
############################
echo "=============================================="
echo "SBATCH: preprocess polyhaven env rotation"
echo "=============================================="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-N/A}"
echo "PROJ: $PROJ"
echo "INPUT_DIR: $INPUT_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "HDRI_DIR: $HDRI_DIR"
echo "SCENE_LIST: $SCENE_LIST"
echo "SPLIT: $SPLIT"
echo "N_VARIATIONS: $N_VARIATIONS"
echo "CONSECUTIVE_FRAMES: $CONSECUTIVE_FRAMES"
echo "FRAME_CHUNK_SEED: $FRAME_CHUNK_SEED"
echo "----------------------------------------------"
echo ""

if [ ! -d "$INPUT_DIR" ]; then
  echo "ERROR: Input directory not found: $INPUT_DIR"
  exit 1
fi
if [ ! -f "$SCENE_LIST" ]; then
  echo "ERROR: Scene list not found: $SCENE_LIST"
  exit 1
fi

############################
# Run preprocessing
############################
export SINGULARITYENV_OPENCV_IO_ENABLE_OPENEXR=1
export SINGULARITYENV_QT_QPA_PLATFORM=offscreen
export SINGULARITYENV_PYOPENGL_PLATFORM=egl

echo "Checking/installing required packages..."
singularity exec --nv $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:\${PYTHONPATH:-}\"
  cd $PROJ

  pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
  pip install opencv-python-headless pyexr
"

echo ""
echo "Running env-variation preprocessing..."
singularity exec --nv $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:\${PYTHONPATH:-}\"
  cd $PROJ

  python preprocess_scripts/preprocess_objaverse_env_variations.py \
    --input \"$INPUT_DIR\" \
    --output \"$OUTPUT_DIR\" \
    --split \"$SPLIT\" \
    --hdri-dir \"$HDRI_DIR\" \
    --scene-list \"$SCENE_LIST\" \
    --n-variations \"$N_VARIATIONS\" \
    --consecutive-frames \"$CONSECUTIVE_FRAMES\" \
    --frame-chunk-seed \"$FRAME_CHUNK_SEED\"
"

echo ""
echo "=============================================="
echo "Preprocessing complete!"
echo "=============================================="
echo "Output: $OUTPUT_DIR/$SPLIT"
echo "full_list: $OUTPUT_DIR/$SPLIT/full_list.txt"
