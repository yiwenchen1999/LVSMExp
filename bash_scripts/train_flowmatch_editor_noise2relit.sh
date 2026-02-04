#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_flowmatch_noise2relit
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_flowmatch_noise2relit.out
#SBATCH --error=myjob.relight_flowmatch_noise2relit.err
export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

export WANDB_DIR=/scratch/chen.yiwe/wandb
export WANDB_ARTIFACT_DIR=/scratch/chen.yiwe/wandb/artifacts
export WANDB_CACHE_DIR=/scratch/chen.yiwe/wandb/cache
export WANDB_CONFIG_DIR=/scratch/chen.yiwe/wandb/config

export XDG_CACHE_HOME=/scratch/chen.yiwe/.cache
export XDG_CONFIG_HOME=/scratch/chen.yiwe/.config
export XDG_DATA_HOME=/scratch/chen.yiwe/.local/share

# Make sure to initialize from the single step editor checkpoint if available
# This path should point to the checkpoint of the model trained with relight_general_dense_lr1e4_singleMap.sh
SINGLE_STEP_CKPT="ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4_singleMap"

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18637 --rdzv_backend c10d --rdzv_endpoint localhost:29503 \
    train_flowmatch_editor.py --config configs/LVSM_flow_match_editor_noise2relit.yaml \
    training.batch_size_per_gpu = 4 \
    training.checkpoint_dir = ckpt/LVSM_flow_match_editor_noise2relit_dense_lr1e4_singleMap \
    training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
    training.wandb_exp_name = LVSM_flowmatch_noise2relit_dense_lr1e4_singleMap \
    training.warmup = 3000 \
    training.vis_every = 1000 \
    training.lr = 0.0001 \
    training.single_env_map = true \
    training.skip_renderer = true \
    training.flow_match.noise_scale = 0.0

