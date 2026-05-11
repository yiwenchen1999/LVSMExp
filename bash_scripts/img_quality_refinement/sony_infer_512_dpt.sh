#!/usr/bin/env bash
# Same as bash_scripts/img_quality_refinement/sony_infer_512.sh but uses the DPT decoder-head
# config (trained with e.g. nebius_train_dense_512_dpt.sh).
#
# Usage:
#   bash bash_scripts/img_quality_refinement/sony_infer_512_dpt.sh
#   DATASET_PATH=/path/to/full_list.txt CKPT_DIR=... bash ...

set -euo pipefail

############################
# Paths & environment (Sony)
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

export WANDB_DIR=/scratch2/$USER/wandb
export WANDB_ARTIFACT_DIR=/scratch2/$USER/wandb/artifacts
export WANDB_CACHE_DIR=/scratch2/$USER/wandb/cache
export WANDB_CONFIG_DIR=/scratch2/$USER/wandb/config

export XDG_CACHE_HOME=/scratch2/$USER/.cache
export XDG_CONFIG_HOME=/scratch2/$USER/.config
export XDG_DATA_HOME=/scratch2/$USER/.local/share

export HF_HOME=/scratch2/$USER/.cache/huggingface
export HF_ACCELERATE_CONFIG_DIR=/scratch2/$USER/.cache/accelerate

export DATASET_PATH="${DATASET_PATH:-/music-shared-disk/group/ct/yiwen/data/objaverse/lvsm_with_envmaps/test/full_list.txt}"
export CKPT_DIR="${CKPT_DIR:-$PROJ/ckpt/infer_512_dpt_objaverse}"
export LVSM_CKPT_DIR="${LVSM_CKPT_DIR:-$PROJ/ckpt/LVSM_object_encoder_decoder_512}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-infer_512_dpt_lvsm_with_envmaps}"
RESUME_CKPT="${RESUME_CKPT:-$PROJ/ckpt/dpt_decoder_512_1e5/ckpt_0000000000012000.pt}"
echo "RESUME_CKPT: $RESUME_CKPT"

############################
# Logging
############################
echo "=============================================="
echo "Relight dense 512 + DPT head (Sony, vis-only)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATASET_PATH: $DATASET_PATH"
echo "CKPT_DIR: $CKPT_DIR"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "WANDB_EXP_NAME: $WANDB_EXP_NAME"
echo "----------------------------------------------"
echo ""

if [ ! -d "$DATASET_PATH" ] && [ ! -f "$DATASET_PATH" ]; then
  echo "ERROR: Dataset path not found: $DATASET_PATH"
  exit 1
fi

############################
# Run (torchrun inside Singularity)
############################
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
    --rdzv_endpoint localhost:29501 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_res_singleMap_dpt_transfer.yaml \
    training.batch_size_per_gpu = 1 \
    training.dataset_path = \"$DATASET_PATH\" \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.resume_ckpt = \"$RESUME_CKPT\" \
    training.reset_training_state = true \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.wandb_exp_name = \"$WANDB_EXP_NAME\" \
    training.relight_signals = \"[envmap]\" \
    training.warmup = 3000 \
    training.vis_every = 1 \
    training.lr = 0.0000 \
"

echo ""
echo "=============================================="
echo "Done. Checkpoints: $CKPT_DIR"
echo "=============================================="
