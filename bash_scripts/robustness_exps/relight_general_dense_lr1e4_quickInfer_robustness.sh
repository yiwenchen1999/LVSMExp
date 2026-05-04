#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_dense_robustness
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_dense_robustness.out
#SBATCH --error=myjob.relight_dense_robustness.err

export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29503 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
    training.batch_size_per_gpu = 1 \
    training.dataset_path = /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsmPlus_objaverse_robustTest_merged/test/full_list.txt \
    training.checkpoint_dir = ckpt/progressive_results_formatchange \
    training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
    training.wandb_exp_name = LVSM_edit_dense_general_lr1e4_robustness_quickInfer \
    training.relight_signals = "[envmap]" \
    training.multi_edit.enable = false \
    training.multi_edit.max_steps = 1 \
    training.warmup = 3000 \
    training.vis_every = 1 \
    training.lr = 0.0000
