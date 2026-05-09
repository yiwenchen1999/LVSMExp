#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=72:00:00
#SBATCH --job-name=obj-light_crosssplit_editor
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.obj-light_crosssplit_editor.out
#SBATCH --error=myjob.obj-light_crosssplit_editor.err

export HF_HOME=/projects/vig/yiwenc/caches
export HF_ACCELERATE_CONFIG_DIR=/projects/vig/yiwenc/caches/accelerate

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29522 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_obj-with-light_crosssplit.yaml \
    training.dataset_path = /projects/vig/Datasets/obj-with-light/lvsm_format/test/full_list.txt \
    training.context_dataset_path = /projects/vig/Datasets/obj-with-light/lvsm_format/train/full_list.txt \
    training.cross_split_relight = true \
    training.editor_condition_source = target \
    training.LVSM_checkpoint_dir = ckpt/realworld_exps_obj-with-light \
    training.checkpoint_dir = ckpt/test_relight_obj-with-light \
    training.wandb_exp_name = test_relight_obj-with-light \
    training.num_input_views = 16 \
    training.num_target_views = 1 \
    training.num_views = 17 \
    training.warmup = 1500 \
    training.vis_every = 1 \
    training.lr = 0.0000
