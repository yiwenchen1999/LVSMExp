#!/bin/bash
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=72:00:00
#SBATCH --job-name=relight_dense_512_dpt_from256
#SBATCH --mem=128
#SBATCH --ntasks=32
#SBATCH --output=myjob.relight_dense_512_dpt_from256.out
#SBATCH --error=myjob.relight_dense_512_dpt_from256.err

# Finetune DPT transfer at 512x512, initializing compatible weights from the 256-res run
# (ckpt/dpt_decoder_256). LVSM base uses 512 object ckpt for architecture-aligned layers.
#
# Flow: init_from_LVSM(512 base) -> load_ckpt (resume_ckpt if CHECKPOINT_DIR has no .pt yet) strict=False.
# Once CHECKPOINT_DIR contains checkpoints, later runs continue from there (resume_ckpt is ignored).
#
# Optional env overrides:
#   CUDA_VISIBLE_DEVICES  default: 7
#   NPROC_PER_NODE       default: 1 (must match visible GPU count)
#   RESUME_CKPT=ckpt/dpt_decoder_256   # default; directory or path to a specific .pt
#   CHECKPOINT_DIR=ckpt/dpt_decoder_512
#   TRAIN_STAGE, STAGE1_STEPS, etc. same as 256 script

TRAIN_STAGE=${TRAIN_STAGE:-stage2}
STAGE1_STEPS=${STAGE1_STEPS:-0}
DISTILL_WEIGHT=${DISTILL_WEIGHT:-0.0}
BACKBONE_LR_SCALE=${BACKBONE_LR_SCALE:-1}
OG_DATASET_BASE=${OG_DATASET_BASE:-/scratch/chen.yiwe/temp_objaverse}
LOCAL_DATASET_BASE=${LOCAL_DATASET_BASE:-/mnt/data-disk2}
DATALOADER_SEED=${DATALOADER_SEED:-779}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

RESUME_CKPT=${RESUME_CKPT:-ckpt/dpt_decoder_512_1e5/ckpt_0000000000012000.pt}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-ckpt/infer_512_dpt}

torchrun --nproc_per_node "${NPROC_PER_NODE}" --nnodes 1 \
    --rdzv_id 28636 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_res_singleMap_dpt_transfer.yaml \
    training.batch_size_per_gpu = 8 \
    training.checkpoint_dir = ${CHECKPOINT_DIR} \
    training.resume_ckpt = ${RESUME_CKPT} \
    training.dataset_path = /mnt/data-disk2/lvsmPlus_objaverse/test/full_list.txt \
    training.LVSM_checkpoint_dir = ckpt/LVSM_object_encoder_decoder_512 \
    training.wandb_exp_name = LVSM_edit_dense_general_512_dptTransfer_from256 \
    training.warmup = 1000 \
    training.vis_every = 1 \
    training.lr = 0.00000 \
    training.batch_size_per_gpu = 1 \
    training.og_dataset_base = ${OG_DATASET_BASE} \
    training.local_dataset_base = ${LOCAL_DATASET_BASE} \
    training.seed = ${DATALOADER_SEED} \
    training.reset_training_state = true \
    training.dpt_transfer.enabled = true \
    training.dpt_transfer.train_stage = ${TRAIN_STAGE} \
    training.dpt_transfer.freeze_backbone_in_stage1 = false \
    training.dpt_transfer.stage1_steps = ${STAGE1_STEPS} \
    training.dpt_transfer.stage2_unfreeze = all \
    training.dpt_transfer.distill_weight = ${DISTILL_WEIGHT} \
    training.dpt_transfer.backbone_lr_scale = ${BACKBONE_LR_SCALE} \
    training.dpt_transfer.gate_init.l12 = 1.0 \
    training.dpt_transfer.gate_init.l9 = 0.0 \
    training.dpt_transfer.gate_init.l6 = 0.0 \
    training.dpt_transfer.gate_init.l3 = 0.0 \
    training.dpt_transfer.gate_ramp_steps.l9_start = 0 \
    training.dpt_transfer.gate_ramp_steps.l6_start =  0 \
    training.dpt_transfer.gate_ramp_steps.l3_start = 0 \
    training.dpt_transfer.gate_ramp_steps.ramp_len = 0
