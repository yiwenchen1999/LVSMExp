#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_dense_256_dpt_transfer
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_dense_256_dpt_transfer.out
#SBATCH --error=myjob.relight_dense_256_dpt_transfer.err

# DPT transfer training at 256x256 with single env-map sampling.
# Two-stage behavior is controlled by training.dpt_transfer.* overrides below.

# Good default initialization for MLP -> DPT transfer:
# - stage1 warmup with frozen backbone
# - stage2 joint finetune with smaller backbone LR
# - gates: l12 on first, l9/l6/l3 progressively enabled
TRAIN_STAGE=${TRAIN_STAGE:-auto}
STAGE1_STEPS=${STAGE1_STEPS:-5000}
DISTILL_WEIGHT=${DISTILL_WEIGHT:-0.0}
BACKBONE_LR_SCALE=${BACKBONE_LR_SCALE:-0.1}

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 28635 --rdzv_backend c10d --rdzv_endpoint localhost:29501 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_singleMap_dpt_transfer.yaml \
    training.batch_size_per_gpu = 8 \
    training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_512_singleMap_dpt_transfer \
    training.dataset_path = /mnt/filesystem-z1/lvsmPlus_objaverse/test/full_list.txt \
    training.LVSM_checkpoint_dir = ckpt/dpt_decoder_256 \
    training.wandb_exp_name = LVSM_edit_dense_general_256_dptTransfer \
    training.warmup = 3000 \
    training.vis_every = 1000 \
    training.lr = 0.0001 \
    training.single_env_map = true \
    training.dpt_transfer.enabled = true \
    training.dpt_transfer.train_stage = ${TRAIN_STAGE} \
    training.dpt_transfer.freeze_backbone_in_stage1 = true \
    training.dpt_transfer.stage1_steps = ${STAGE1_STEPS} \
    training.dpt_transfer.stage2_unfreeze = all \
    training.dpt_transfer.distill_weight = ${DISTILL_WEIGHT} \
    training.dpt_transfer.backbone_lr_scale = ${BACKBONE_LR_SCALE} \
    training.dpt_transfer.gate_init.l12 = 1.0 \
    training.dpt_transfer.gate_init.l9 = 0.0 \
    training.dpt_transfer.gate_init.l6 = 0.0 \
    training.dpt_transfer.gate_init.l3 = 0.0 \
    training.dpt_transfer.gate_ramp_steps.l9_start = 0 \
    training.dpt_transfer.gate_ramp_steps.l6_start = 2000 \
    training.dpt_transfer.gate_ramp_steps.l3_start = 4000 \
    training.dpt_transfer.gate_ramp_steps.ramp_len = 2000
