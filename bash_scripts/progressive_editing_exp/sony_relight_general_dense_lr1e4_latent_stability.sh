#!/bin/bash
# Train relight editor latent stability experiment on Sony cluster.
# Usage:
#   bash bash_scripts/progressive_editing_exp/sony_relight_general_dense_lr1e4_latent_stability.sh

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

# Train paths (allow overrides with env vars)
export DATA_LIST="${DATA_LIST:-/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm/test/full_list.txt}"
export CKPT_DIR="${CKPT_DIR:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/progressive_results_new}"
export LVSM_CKPT_DIR="${LVSM_CKPT_DIR:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-LVSM_edit_dense_general_lr1e4}"

############################
# Logging
############################
echo "=============================================="
echo "Train Relight Latent Stability (Sony)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "WANDB_EXP_NAME: $WANDB_EXP_NAME"
echo "PY_SITE: $PY_SITE"
echo "WANDB_DIR: $WANDB_DIR"
echo "----------------------------------------------"
echo ""

############################
# Checks
############################
if [ ! -f "$DATA_LIST" ]; then
  echo "ERROR: Dataset list not found: $DATA_LIST"
  exit 1
fi

############################
# Confirmation
############################
read -p "Proceed with training? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

############################
# Run training
############################
echo ""
echo "Starting training..."
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
    --rdzv_endpoint localhost:29503 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
    training.batch_size_per_gpu = 1 \
    training.dataset_path = \"$DATA_LIST\" \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.wandb_exp_name = \"$WANDB_EXP_NAME\" \
    training.relight_signals = \"[envmap]\" \
    training.multi_edit.enable = true \
    training.multi_edit.max_steps = 10 \
    training.multi_edit.sample_mode = uniform \
    training.multi_edit.force_all_steps = true \
    training.multi_edit.final_only = true \
    training.multi_edit.insufficient_chain_policy = resample \
    training.warmup = 3000 \
    training.vis_every = 1 \
    training.lr = 0.0000
"

############################
# Done
############################
echo ""
echo "=============================================="
echo "✓ Training complete!"
echo "=============================================="
echo "Checkpoints saved to: $CKPT_DIR"
echo ""
