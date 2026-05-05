#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_dense_512_dpt_transfer
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_dense_512_dpt_transfer.out
#SBATCH --error=myjob.relight_dense_512_dpt_transfer.err

# DPT transfer training at 512x512 with single env-map sampling.
# Two-stage behavior is controlled by training.dpt_transfer.* overrides below.

export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

export WANDB_DIR=/scratch/chen.yiwe/wandb
export WANDB_ARTIFACT_DIR=/scratch/chen.yiwe/wandb/artifacts
export WANDB_CACHE_DIR=/scratch/chen.yiwe/wandb/cache
export WANDB_CONFIG_DIR=/scratch/chen.yiwe/wandb/config

export XDG_CACHE_HOME=/scratch/chen.yiwe/.cache
export XDG_CONFIG_HOME=/scratch/chen.yiwe/.config
export XDG_DATA_HOME=/scratch/chen.yiwe/.local/share

# Stage controls (override from shell if needed).
TRAIN_STAGE=${TRAIN_STAGE:-auto}                  # stage1 | stage2 | auto
STAGE1_STEPS=${STAGE1_STEPS:-5000}
DISTILL_WEIGHT=${DISTILL_WEIGHT:-0.0}
BACKBONE_LR_SCALE=${BACKBONE_LR_SCALE:-0.1}

torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id 28635 --rdzv_backend c10d --rdzv_endpoint localhost:29501 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_singleMap_dpt_transfer.yaml \
    training.batch_size_per_gpu = 8 \
    training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_512_singleMap_dpt_transfer \
    training.LVSM_checkpoint_dir = ckpt/LVSM_object_encoder_decoder_512 \
    training.wandb_exp_name = LVSM_edit_dense_general_512_dptTransfer_singleMap \
    training.warmup = 3000 \
    training.vis_every = 1000 \
    training.lr = 0.0001 \
    training.single_env_map = true \
    training.dpt_transfer.train_stage = ${TRAIN_STAGE} \
    training.dpt_transfer.stage1_steps = ${STAGE1_STEPS} \
    training.dpt_transfer.distill_weight = ${DISTILL_WEIGHT} \
    training.dpt_transfer.backbone_lr_scale = ${BACKBONE_LR_SCALE}
