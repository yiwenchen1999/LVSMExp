#!/bin/bash
#SBATCH --job-name=recon_stanford_512
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=/group2/ct/yiwen/logs/%x.%N.%j.err

# Train Stanford ORB recon-only at 512x512 from 256x256 checkpoint.
# (Sony cluster, Singularity.)

set -euo pipefail

############################
# Paths & environment
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

############################
# Training controls
############################
export DATASET_ROOT="${DATASET_ROOT:-/music-shared-disk/group/ct/yiwen/data/lvsm_stanford_orb}"
export DATASET_PATH="${DATASET_PATH:-$DATASET_ROOT/test/full_list.txt}"
export OG_DATASET_BASE="${OG_DATASET_BASE:-/projects/vig/Datasets/stanfordORB}"
export LOCAL_DATASET_BASE="${LOCAL_DATASET_BASE:-/music-shared-disk/group/ct/yiwen/data}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJ/ckpt/recon_stanford_512_sony}"
export RESUME_CKPT="${RESUME_CKPT:-$PROJ/ckpt/stanfordORBrecon256.pt}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-recon_stanford_512_sony}"

export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
export NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-16}"
export NUM_TARGET_VIEWS="${NUM_TARGET_VIEWS:-8}"
export NUM_VIEWS="${NUM_VIEWS:-24}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-3000}"
export VIS_EVERY="${VIS_EVERY:-1000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export DATALOADER_SEED="${DATALOADER_SEED:-779}"
export MASTER_PORT="${MASTER_PORT:-29533}"

############################
# Logging
############################
echo "=============================================="
echo "SBATCH: Stanford ORB recon-only @ 512"
echo "=============================================="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-N/A}"
echo "PROJ: $PROJ"
echo "DATASET_PATH: $DATASET_PATH"
echo "OG_DATASET_BASE: $OG_DATASET_BASE"
echo "LOCAL_DATASET_BASE: $LOCAL_DATASET_BASE"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "RESUME_CKPT: $RESUME_CKPT"
echo "WANDB_EXP_NAME: $WANDB_EXP_NAME"
echo "LR: $LEARNING_RATE"
echo "recon_only: true (no relit supervision)"
echo "----------------------------------------------"
echo ""

if [ ! -f "$DATASET_PATH" ]; then
  echo "ERROR: Dataset list not found: $DATASET_PATH"
  exit 1
fi

if [ ! -f "$RESUME_CKPT" ] && [ ! -d "$RESUME_CKPT" ]; then
  echo "ERROR: Resume checkpoint not found: $RESUME_CKPT"
  exit 1
fi

############################
# Run training (directly in batch allocation; no srun step)
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
  cd \"$PROJ\"

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:$MASTER_PORT \
    train.py --config configs/LVSM_scene_encoder_decoder_512.yaml \
    training.dataset_name = data.dataset_scene_stanfordORB.Dataset \
    training.dataset_path = \"$DATASET_PATH\" \
    training.og_dataset_base = \"$OG_DATASET_BASE\" \
    training.local_dataset_base = \"$LOCAL_DATASET_BASE\" \
    training.recon_only = true \
    training.use_relit_images = false \
    training.resume_ckpt = \"$RESUME_CKPT\" \
    training.reset_training_state = true \
    training.checkpoint_dir = \"$CHECKPOINT_DIR\" \
    training.wandb_exp_name = \"$WANDB_EXP_NAME\" \
    training.batch_size_per_gpu = ${BATCH_SIZE_PER_GPU} \
    training.num_input_views = ${NUM_INPUT_VIEWS} \
    training.num_target_views = ${NUM_TARGET_VIEWS} \
    training.num_views = ${NUM_VIEWS} \
    training.lr = ${LEARNING_RATE} \
    training.warmup = ${WARMUP_STEPS} \
    training.seed = ${DATALOADER_SEED} \
    training.vis_every = ${VIS_EVERY} \
    training.save_every = ${SAVE_EVERY}
"

echo ""
echo "=============================================="
echo "SBATCH training complete."
echo "=============================================="
echo "Checkpoints saved to: $CHECKPOINT_DIR"

