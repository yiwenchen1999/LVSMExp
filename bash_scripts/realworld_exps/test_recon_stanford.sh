#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_stanford
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_stanford.out
#SBATCH --error=myjob.relight_stanford.err

export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

# Run Weights & Biases offline (no API/network during training; sync later with `wandb sync <run-dir>`).
torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29503 \
    train.py --config configs/LVSM_scene_encoder_decoder.yaml \
    training.batch_size_per_gpu = 8 \
    training.dataset_path = /projects/vig/Datasets/stanfordORB/lvsm_stanford_orb/train/full_list.txt \
    training.checkpoint_dir = ckpt/realworld_exps \
    training.wandb_exp_name = realworld_exps \
    training.num_input_views = 16 \
    training.num_target_views = 1 \
    training.num_views = 17 \
    training.warmup = 3000 \
    training.vis_every = 1 \
    training.lr = 0.0001
