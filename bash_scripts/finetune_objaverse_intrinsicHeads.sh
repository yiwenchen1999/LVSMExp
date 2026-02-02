#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=168:00:00
#SBATCH --job-name=finetune_objaverse_intrinsicHeads
#SBATCH --mem=32
#SBATCH --ntasks=16
#SBATCH --output=myjob.finetune_objaverse_intrinsicHeads.out
#SBATCH --error=myjob.finetune_objaverse_intrinsicHeads.err
export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

export WANDB_DIR=/scratch/chen.yiwe/wandb
export WANDB_ARTIFACT_DIR=/scratch/chen.yiwe/wandb/artifacts
export WANDB_CACHE_DIR=/scratch/chen.yiwe/wandb/cache
export WANDB_CONFIG_DIR=/scratch/chen.yiwe/wandb/config

export XDG_CACHE_HOME=/scratch/chen.yiwe/.cache
export XDG_CONFIG_HOME=/scratch/chen.yiwe/.config
export XDG_DATA_HOME=/scratch/chen.yiwe/.local/share

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    train.py --config configs/LVSM_scene_encoder_decoder_wIntrinsicDecoder.yaml \
    training.batch_size_per_gpu = 4 \
    training.grad_accum_steps = 1

