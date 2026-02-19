#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=train_editor_pointlight
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.train_editor_pointlight.out
#SBATCH --error=myjob.train_editor_pointlight.err
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
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_pointlight.yaml \
    training.batch_size_per_gpu = 8 \
    training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_pointlight \
    training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_dense \
    training.wandb_exp_name = LVSM_edit_pointlight \
    training.dataset_path = /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse/test/full_list.txt \
    training.whiteEnvInput = true \
    training.vis_every = 2 \
    training.lr = 0.0001
