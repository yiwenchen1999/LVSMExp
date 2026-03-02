#!/bin/bash
# Interactive inference with presets for Sony cluster
# Usage: bash bash_scripts/Sony_clusters/interactive_inference_presets.sh

set -euo pipefail

############################
# Paths & environment
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

# Base paths
export BASE_DATA_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse"
export LVSM_CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"

############################
# Logging
############################
echo "=============================================="
echo "Inference with Presets (Sony Cluster)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "----------------------------------------------"
echo ""

############################
# Preset Menu
############################
echo "Select inference preset:"
echo ""
echo "1) Polyhaven dataset (dense_relight_env model)"
echo "   Data: polyhaven_lvsm/test"
echo "   Model: ckpt/dense_relight_env"
echo "   Eval index: evaluation_index_polyhaven_dense.json"
echo ""
echo "2) LVSM dense (relight_general_dense_lr1e4 model)"
echo "   Data: lvsm_scenes_dense/test"
echo "   Model: ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4"
echo "   Eval index: evaluation_index_objaverse_dense.json"
echo ""
echo "3) LVSM dense test split"
echo "   Data: lvsm_with_envmaps_test_split/test"
echo "   Model: ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4"
echo "   Eval index: evaluation_index_objaverse_dense_test_split.json"
echo ""
echo "4) Custom (enter paths manually)"
echo ""
echo "5) Exit"
echo "----------------------------------------------"
read -p "Choice [1-5]: " choice

############################
# Set parameters based on choice
############################
case "$choice" in
  1)
    DATA_LIST="$BASE_DATA_DIR/polyhaven_lvsm/test/full_list.txt"
    CKPT_DIR="$PROJ/ckpt/dense_relight_env"
    EVAL_INDEX="$PROJ/data/evaluation_index_polyhaven_dense.json"
    OUTPUT_DIR="$PROJ/experiments/evaluation/polyhaven_dense_inference"
    CONFIG="configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml"
    ;;
  2)
    DATA_LIST="$BASE_DATA_DIR/lvsm_scenes_dense/test/full_list.txt"
    CKPT_DIR="$PROJ/ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4"
    EVAL_INDEX="$PROJ/data/evaluation_index_objaverse_dense.json"
    OUTPUT_DIR="$PROJ/experiments/evaluation/lvsm_dense_inference"
    CONFIG="configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml"
    ;;
  3)
    DATA_LIST="$BASE_DATA_DIR/lvsm_with_envmaps_test_split/test/full_list.txt"
    CKPT_DIR="$PROJ/ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4"
    EVAL_INDEX="$PROJ/data/evaluation_index_objaverse_dense_test_split.json"
    OUTPUT_DIR="$PROJ/experiments/evaluation/test_split_inference"
    CONFIG="configs/LVSM_scene_encoder_decoder_wEditor.yaml"
    ;;
  4)
    # Custom mode
    echo ""
    echo "Custom configuration mode:"
    read -p "Data list path: " DATA_LIST
    read -p "Checkpoint directory: " CKPT_DIR
    read -p "Evaluation index JSON: " EVAL_INDEX
    read -p "Output directory: " OUTPUT_DIR
    read -p "Config file [configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml]: " CONFIG
    CONFIG=${CONFIG:-configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml}
    ;;
  5)
    echo "Exiting."
    exit 0
    ;;
  *)
    echo "Invalid choice."
    exit 1
    ;;
esac

############################
# Additional parameters
############################
echo ""
read -p "Batch size per GPU [4]: " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-4}

read -p "Number of input views [4]: " N_INPUT
N_INPUT=${N_INPUT:-4}

read -p "Number of target views [8]: " N_TARGET
N_TARGET=${N_TARGET:-8}

read -p "Compute metrics? [y/N]: " COMPUTE_METRICS
if [[ "$COMPUTE_METRICS" =~ ^[Yy]$ ]]; then
  COMPUTE_METRICS_FLAG="true"
else
  COMPUTE_METRICS_FLAG="false"
fi

read -p "Render video? [y/N]: " RENDER_VIDEO
if [[ "$RENDER_VIDEO" =~ ^[Yy]$ ]]; then
  RENDER_VIDEO_FLAG="true"
else
  RENDER_VIDEO_FLAG="false"
fi

############################
# Confirmation
############################
echo ""
echo "=============================================="
echo "Configuration Summary"
echo "=============================================="
echo "Data list:        $DATA_LIST"
echo "Checkpoint:       $CKPT_DIR"
echo "LVSM checkpoint:  $LVSM_CKPT_DIR"
echo "Config:           $CONFIG"
echo "Eval index:       $EVAL_INDEX"
echo "Output dir:       $OUTPUT_DIR"
echo "Batch size:       $BATCH_SIZE"
echo "Input views:      $N_INPUT"
echo "Target views:     $N_TARGET"
echo "Compute metrics:  $COMPUTE_METRICS_FLAG"
echo "Render video:     $RENDER_VIDEO_FLAG"
echo "----------------------------------------------"
echo ""

# Check if files exist
if [ ! -f "$EVAL_INDEX" ]; then
  echo "WARNING: Evaluation index not found: $EVAL_INDEX"
  echo "You may need to create it first."
  echo ""
fi

read -p "Proceed with inference? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

############################
# Run inference
############################
echo ""
echo "Starting inference..."
singularity exec --nv $BIND $SIF bash -lc "
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

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29506 \
    inference_editor.py --config \"$CONFIG\" \
    training.dataset_path = \"$DATA_LIST\" \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.batch_size_per_gpu = $BATCH_SIZE \
    training.target_has_input = false \
    training.num_views = \$(($N_INPUT + $N_TARGET)) \
    training.square_crop = true \
    training.num_input_views = $N_INPUT \
    training.num_target_views = $N_TARGET \
    inference.if_inference = true \
    inference.compute_metrics = $COMPUTE_METRICS_FLAG \
    inference.render_video = $RENDER_VIDEO_FLAG \
    inference.view_idx_file_path = \"$EVAL_INDEX\" \
    inference_out_dir = \"$OUTPUT_DIR\"
"

############################
# Done
############################
echo ""
echo "=============================================="
echo "✓ Inference complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
