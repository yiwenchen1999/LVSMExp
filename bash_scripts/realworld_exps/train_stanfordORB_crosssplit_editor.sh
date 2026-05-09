#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=72:00:00
#SBATCH --job-name=stanfordORB_crosssplit_editor
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.stanfordORB_crosssplit_editor.out
#SBATCH --error=myjob.stanfordORB_crosssplit_editor.err

export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29521 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_stanfordORB_crosssplit.yaml \
    training.dataset_path = data_samples/processed_stanford_ORB_objaverse_like/test/full_list.txt \
    training.context_dataset_path = data_samples/processed_stanford_ORB_objaverse_like/train/full_list.txt \
    training.LVSM_checkpoint_dir = ckpt/realworld_exps \
    training.checkpoint_dir = ckpt/realworld_exps_relight_stanfordORB \
    training.wandb_exp_name = LVSM_edit_stanfordORB_crosssplit \
    training.warmup = 1500 \
    training.vis_every = 1000 \
    training.lr = 0.0001
