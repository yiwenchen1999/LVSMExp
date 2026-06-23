#!/bin/bash
#SBATCH --job-name=video_train_load_all_recon
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=/group2/ct/yiwen/logs/%x.%N.%j.err

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
export DATASET_PATH="${DATASET_PATH:-/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm/test/full_list_plant.txt}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJ/ckpt_dpt/video_making_polyhaven_plant_all}"
export RESUME_CKPT="${RESUME_CKPT:-$PROJ/ckpt_dpt/video_making_train_load_all_recon_only}"
export LVSM_CKPT_DIR="${LVSM_CKPT_DIR:-$PROJ/ckpt/LVSM_object_encoder_decoder_512}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-LVSM_video_making_train_load_all_recon_only}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export DATALOADER_SEED="${DATALOADER_SEED:-779}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"

export LOAD_ALL_FRAMES="${LOAD_ALL_FRAMES:-true}"
export NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-10}"
export LOAD_ALL_MAX_TARGET_VIEWS="${LOAD_ALL_MAX_TARGET_VIEWS:-32}"
export EXCLUDE_WHITE_ENV0_FROM_RELIT="${EXCLUDE_WHITE_ENV0_FROM_RELIT:-true}"

############################
# Logging
############################
echo "=============================================="
echo "SBATCH: train_editor load-all recon-only (polyhaven_lvsm)"
echo "=============================================="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-N/A}"
echo "DATASET_PATH: $DATASET_PATH"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "RESUME_CKPT: $RESUME_CKPT"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "WANDB_EXP_NAME: $WANDB_EXP_NAME"
echo "BATCH_SIZE_PER_GPU: $BATCH_SIZE_PER_GPU"
echo "LOAD_ALL_FRAMES: $LOAD_ALL_FRAMES"
echo "NUM_INPUT_VIEWS: $NUM_INPUT_VIEWS"
echo "LOAD_ALL_MAX_TARGET_VIEWS: $LOAD_ALL_MAX_TARGET_VIEWS"
echo "EXCLUDE_WHITE_ENV0_FROM_RELIT: $EXCLUDE_WHITE_ENV0_FROM_RELIT"
echo "recon_only: true (editor pass skipped, supervise on target.image)"
echo "----------------------------------------------"
echo ""

if [ ! -f "$DATASET_PATH" ]; then
  echo "ERROR: Dataset list not found: $DATASET_PATH"
  exit 1
fi

############################
# Run training
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
    --rdzv_endpoint localhost:29542 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_res_singleMap_dpt_transfer_recon_only.yaml \
    training.dataset_path = \"$DATASET_PATH\" \
    training.checkpoint_dir = \"$CHECKPOINT_DIR\" \
    training.resume_ckpt = \"$RESUME_CKPT\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.wandb_exp_name = \"$WANDB_EXP_NAME\" \
    training.batch_size_per_gpu = ${BATCH_SIZE_PER_GPU} \
    training.seed = ${DATALOADER_SEED} \
    training.recon_only = true \
    training.use_relit_images = false \
    training.num_input_views = ${NUM_INPUT_VIEWS} \
    training.load_all_frames = ${LOAD_ALL_FRAMES} \
    training.load_all_max_target_views = ${LOAD_ALL_MAX_TARGET_VIEWS} \
    training.exclude_white_env0_from_relit_sampling = ${EXCLUDE_WHITE_ENV0_FROM_RELIT} \
    training.lr = ${LEARNING_RATE} \
    training.warmup = ${WARMUP_STEPS} \
    training.reset_training_state = true \
    training.dpt_transfer.enabled = true \
    training.dpt_transfer.train_stage = stage2 \
    training.dpt_transfer.stage2_unfreeze = all \
    training.dpt_transfer.backbone_lr_scale = 1.0 \
    training.checkpoint_every = 1000 \
    training.vis_every = 1
"
