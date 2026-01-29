#!/bin/bash
#SBATCH --job-name=lvsm_flowmatch_editor_reconLoss
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=168:00:00
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=/group2/ct/yiwen/logs/%x.%N.%j.err

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/data,/music-shared-disk"

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

# Training paths (Sony cluster)
export DATA_LIST="/music-shared-disk/group/ct/yiwen/data/objaverse/lvsm_with_envmaps/test/full_list.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_flow_match_editor_dense_lr1e4_singleMap_wReconLoss"
export LVSM_CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"

# Make sure to initialize from the single step editor checkpoint if available
# This path should point to the checkpoint of the model trained with relight_general_dense_lr1e4_singleMap.sh
SINGLE_STEP_CKPT="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4_singleMap"

############################
# Logging (optional but useful)
############################
echo "Host: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "PY_SITE: $PY_SITE"
echo "WANDB_DIR: $WANDB_DIR"
echo "----------------------------------"

############################
# Run training
############################
srun singularity exec --nv $BIND $SIF bash -lc "
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

  torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id \$SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29507 \
    train_flowmatch_editor.py --config configs/LVSM_flow_match_editor.yaml \
    training.batch_size_per_gpu = 16 \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.wandb_exp_name = LVSM_flowmatch_dense_lr1e4_singleMap_wReconLoss \
    training.warmup = 3000 \
    training.vis_every = 1000 \
    training.lr = 0.0001 \
    training.single_env_map = true \
    training.skip_renderer = false \
    training.flow_match.noise_scale = 0.0 \
    training.dataset_path = \"$DATA_LIST\"
"


