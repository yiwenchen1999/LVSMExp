#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_general_dense
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_general_dense.out
#SBATCH --error=myjob.relight_general_dense.err
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
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29501 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
    training.batch_size_per_gpu = 4 \
    training.checkpoint_dir = ckpt/LVSM_editor_general_dense_lr1e4_IntrinsicReg1a \
    training.LVSM_checkpoint_dir = ckpt/LVSM_object_encoder_decoder_intrinsicHead \
    training.wandb_exp_name = LVSM_edit_dense_general_lr1e4_IntrinsicReg1a \
    training.warmup = 3000 \
    training.vis_every = 1000 \
    training.lr = 0.0001 \
    model.transformer.use_albedo_decoder = true \
    training.use_albedos = true \
    training.white_env_as_albedo = false \
    training.use_relit_images = true \
    training.single_env_map = true
