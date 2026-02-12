#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_x_prediction_overfit_multiTime_render_only
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_x_prediction_overfit_multiTime_render_only.out
#SBATCH --error=myjob.relight_x_prediction_overfit_multiTime_render_only.err
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
    --rdzv_id 18642 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    train_flowmatch_editor.py --config configs/LVSM_flow_match_editor_noise2relit.yaml \
    model.class_name = model.LVSM_x_prediction_editor_noise2relit.XPredictionEditor \
    training.batch_size_per_gpu = 4 \
    training.checkpoint_dir = ckpt/LVSM_x_prediction_editor_noise2relit_overfit_multiTime_render_only \
    training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
    training.wandb_exp_name = LVSM_x_prediction_overfit_multiTime_render_only \
    training.dataset_path = data_samples/objaverse_processed_with_envmaps/test/full_list.txt \
    training.warmup = 3000 \
    training.vis_every = 1000 \
    training.lr = 0.00001 \
    training.single_env_map = true \
    training.skip_renderer = false \
    training.training_mode = render_only \
    training.fixed_t = 0.0 \
    training.noise_seed = 42 \
    training.single_step_inference = true \
    training.compute_hungarian_loss = false \
