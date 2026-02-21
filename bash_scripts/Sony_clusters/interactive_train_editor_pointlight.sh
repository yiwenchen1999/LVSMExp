#!/bin/bash
# Run inside an interactive Sony cluster session.
# Usage:
#   source bash_scripts/Sony_clusters/interactive_train_editor_pointlight.sh

set -euo pipefail

export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/data,/music-shared-disk"

export WANDB_DIR=/scratch2/$USER/wandb
export WANDB_ARTIFACT_DIR=/scratch2/$USER/wandb/artifacts
export WANDB_CACHE_DIR=/scratch2/$USER/wandb/cache
export WANDB_CONFIG_DIR=/scratch2/$USER/wandb/config
export XDG_CACHE_HOME=/scratch2/$USER/.cache
export XDG_CONFIG_HOME=/scratch2/$USER/.config
export XDG_DATA_HOME=/scratch2/$USER/.local/share
export HF_HOME=/scratch2/$USER/.cache/huggingface
export HF_ACCELERATE_CONFIG_DIR=/scratch2/$USER/.cache/accelerate

export DATA_LIST="/music-shared-disk/group/ct/yiwen/data/objaverse/lvsmPlus_objaverse/test/full_list_point_light.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder_wEditor_pointlight"
export LVSM_CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_object_encoder_decoder_dense"

echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "----------------------------------"

singularity exec --nv $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
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

  torchrun --nproc_per_node 2 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29517 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_pointlight.yaml \
    training.dataset_path = \"$DATA_LIST\" \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.lr = 0.0001 \
    training.vis_every = 1000 \
    training.warmup = 3000 \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.wandb_exp_name = LVSM_editor_pointlight
"
