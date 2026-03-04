#!/bin/bash
# Interactive degradation test for iterative editing (Sony cluster)
#
# Runs both image-space and token-space degradation experiments using the
# same checkpoint and envmap sequence for fair comparison.
#
# Usage:
#   bash bash_scripts/Sony_clusters/SonyAIClusterUtil/interactive_degrade_test.sh
#
# Optional env overrides (set before running):
#   SCENE_IDX=0          # which scene index to test (default: 0)
#   NUM_ITER=100         # number of editing iterations (default: 100)
#   NUM_INPUT_VIEWS=4    # context views (default: 4)

set -euo pipefail

############################
# Tunables
############################
SCENE_IDX="${SCENE_IDX:-0}"
NUM_ITER="${NUM_ITER:-100}"
NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-4}"

############################
# Paths & environment
# (inherited from interactive_inference_relight_general_dense_lr1e4_10k.sh)
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

# WANDB directories
export WANDB_DIR=/scratch2/$USER/wandb
export WANDB_ARTIFACT_DIR=/scratch2/$USER/wandb/artifacts
export WANDB_CACHE_DIR=/scratch2/$USER/wandb/cache
export WANDB_CONFIG_DIR=/scratch2/$USER/wandb/config

# Cache directories
export XDG_CACHE_HOME=/scratch2/$USER/.cache
export XDG_CONFIG_HOME=/scratch2/$USER/.config
export XDG_DATA_HOME=/scratch2/$USER/.local/share

# HuggingFace cache
export HF_HOME=/scratch2/$USER/.cache/huggingface
export HF_ACCELERATE_CONFIG_DIR=/scratch2/$USER/.cache/accelerate

# Data & checkpoint paths (same as dense_relight_env inference)
export DATA_LIST="/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm/test/full_list.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/dense_relight_env"
export LVSM_CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"

# Output directories
export OUT_IMAGE_SPACE="$PROJ/experiments/degrade_test_imageSpace"
export OUT_TOKEN_SPACE="$PROJ/experiments/degrade_test_tokenSpace"

############################
# Logging
############################
echo "=============================================="
echo "Iterative Editing Degradation Test (Sony)"
echo "=============================================="
echo "Host          : $(hostname)"
echo "Scene index   : $SCENE_IDX"
echo "Iterations    : $NUM_ITER"
echo "Input views   : $NUM_INPUT_VIEWS"
echo "CKPT_DIR      : $CKPT_DIR"
echo "LVSM_CKPT_DIR : $LVSM_CKPT_DIR"
echo "DATA_LIST     : $DATA_LIST"
echo "Out (image)   : $OUT_IMAGE_SPACE"
echo "Out (token)   : $OUT_TOKEN_SPACE"
echo "----------------------------------------------"
echo ""

############################
# Shared singularity env block
############################
SING_ENV="
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:\${PYTHONPATH:-}\"
  export WANDB_DIR=\"$WANDB_DIR\"
  export WANDB_ARTIFACT_DIR=\"$WANDB_ARTIFACT_DIR\"
  export WANDB_CACHE_DIR=\"$WANDB_CACHE_DIR\"
  export WANDB_CONFIG_DIR=\"$WANDB_CONFIG_DIR\"
  export XDG_CACHE_HOME=\"$XDG_CACHE_HOME\"
  export XDG_CONFIG_HOME=\"$XDG_CONFIG_HOME\"
  export XDG_DATA_HOME=\"$XDG_DATA_HOME\"
  export HF_HOME=\"$HF_HOME\"
  export HF_ACCELERATE_CONFIG_DIR=\"$HF_ACCELERATE_CONFIG_DIR\"
  cd $PROJ
"

# Shared torchrun + config overrides
SHARED_ARGS="\
    --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
    training.batch_size_per_gpu = 1 \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.dataset_path = \"$DATA_LIST\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.single_env_map = true \
    training.num_input_views = $NUM_INPUT_VIEWS \
    training.num_views = 12 \
    inference.if_inference = true \
    inference.degrade_test_scene_idx = $SCENE_IDX \
    inference.degrade_num_iterations = $NUM_ITER"

############################
# 1. Image-space experiment
############################
echo "[1/2] Running IMAGE-SPACE degradation test ..."
singularity exec --nv $BIND $SIF bash -lc "
  $SING_ENV

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29506 \
    inference_editor_degrade_test_imageSpace.py \
    $SHARED_ARGS \
    inference_out_dir = \"$OUT_IMAGE_SPACE\"
"
echo "[1/2] Image-space test complete."
echo ""

############################
# 2. Token-space experiment
############################
echo "[2/2] Running TOKEN-SPACE degradation test ..."
singularity exec --nv $BIND $SIF bash -lc "
  $SING_ENV

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29506 \
    inference_editor_degrade_test_tokenSpace.py \
    $SHARED_ARGS \
    inference_out_dir = \"$OUT_TOKEN_SPACE\"
"
echo "[2/2] Token-space test complete."

############################
# Done
############################
echo ""
echo "=============================================="
echo "Both degradation tests finished."
echo "=============================================="
echo "Image-space results: $OUT_IMAGE_SPACE"
echo "Token-space results: $OUT_TOKEN_SPACE"
echo ""
