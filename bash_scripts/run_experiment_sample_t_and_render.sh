#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=experiment_sample_t_render
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.experiment_sample_t_render.out
#SBATCH --error=myjob.experiment_sample_t_render.err
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
    --rdzv_id 18650 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    experiment_sample_t_and_render.py --config configs/LVSM_flow_match_editor_noise2relit.yaml \
    model.class_name = model.LVSM_x_prediction_editor_noise2relit_overfit_chamfer.XPredictionEditor \
    training.batch_size_per_gpu = 1 \
    training.checkpoint_dir = ckpt/LVSM_x_prediction_editor_noise2relit_overfit_chamfer_flow_only \
    training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
    training.dataset_path = data_samples/objaverse_processed_with_envmaps/test/full_list.txt \
    training.single_env_map = true \
    training.num_samples_to_process = 1 \
    training.experiment_output_dir = experiment_results/sample_t_and_render \
    training.compute_hungarian_loss = true \
    training.noise_seed = 42

