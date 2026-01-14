#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a5000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=overfit_objaverse
#SBATCH --mem=32
#SBATCH --ntasks=16
#SBATCH --output=myjob.overfit_objaverse.out
#SBATCH --error=myjob.overfit_objaverse.err
export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

export WANDB_DIR=/scratch/chen.yiwe/wandb
export WANDB_ARTIFACT_DIR=/scratch/chen.yiwe/wandb/artifacts
export WANDB_CACHE_DIR=/scratch/chen.yiwe/wandb/cache
export WANDB_CONFIG_DIR=/scratch/chen.yiwe/wandb/config

export XDG_CACHE_HOME=/scratch/chen.yiwe/.cache
export XDG_CONFIG_HOME=/scratch/chen.yiwe/.config
export XDG_DATA_HOME=/scratch/chen.yiwe/.local/share

torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_overfit.yaml \
    training.batch_size_per_gpu = 4 \
    training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_overfit \
    training.LVSM_checkpoint_dir = ckpt/LVSM_object_encoder_decoder
