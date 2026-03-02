#!/bin/bash
# Interactive inference for polyhaven dataset on Sony cluster
# Usage: bash bash_scripts/Sony_clusters/interactive_inference_polyhaven_dense.sh

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

# WANDB directories (Sony cluster paths)
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

# Inference paths (Sony cluster)
export DATA_LIST="/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm/test/full_list.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/dense_relight_env"
export LVSM_CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"
export EVAL_INDEX="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/data/evaluation_index_polyhaven_dense.json"
export OUTPUT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/experiments/evaluation/polyhaven_dense_inference"

############################
# Logging
############################
echo "=============================================="
echo "Inference for Polyhaven Dataset (Sony)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "EVAL_INDEX: $EVAL_INDEX"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PY_SITE: $PY_SITE"
echo "WANDB_DIR: $WANDB_DIR"
echo "----------------------------------------------"
echo ""

############################
# Check if evaluation index exists
############################
if [ ! -f "$EVAL_INDEX" ]; then
  echo "ERROR: Evaluation index not found: $EVAL_INDEX"
  echo ""
  echo "Please create it first using:"
  echo "  bash bash_scripts/Sony_clusters/interactive_create_evaluation_index_presets.sh"
  echo ""
  exit 1
fi

############################
# Confirmation
############################
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
    inference_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
    training.dataset_path = \"$DATA_LIST\" \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.batch_size_per_gpu = 4 \
    training.target_has_input = false \
    training.num_views = 12 \
    training.square_crop = true \
    training.num_input_views = 4 \
    training.num_target_views = 8 \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
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
