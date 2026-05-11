#!/bin/bash
# Nebius VM (non-Slurm): Stanford ORB 512 cross-split editor finetune.
# Run from the LVSMExp repo root:  bash bash_scripts/realworld_exps/train_stanfordORB_512_editor.sh
#
# - init_from_LvSM: 512 Images2LatentScene (default ckpt/LVSM_scene_encoder_decoder_512 under repo).
# - resume_ckpt: 256-res Stanford ORB editor weights; strict=False after LVSM init.
# - After CHECKPOINT_DIR contains .pt files, later runs resume from there (resume_ckpt ignored).
#
# Optional env (same idea as bash_scripts/img_quality_refinement/nebius_train_dense_512_dpt.sh):
#   DATASET_ROOT   default: $HOME/Datasets/stanfordORB/lvsm_stanford_orb
#   OG_DATASET_BASE / LOCAL_DATASET_BASE: remap paths inside full_list (e.g. xfer /work/vig/... -> this VM)
#   NPROC_PER_NODE default: 4 (override for your GPU count)
#   LVSM_CKPT_512  RESUME_CKPT  CHECKPOINT_DIR
#   HF_HOME  HF_ACCELERATE_CONFIG_DIR

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${HOME}/Datasets/stanfordORB/lvsm_stanford_orb}"
OG_DATASET_BASE="${OG_DATASET_BASE:-/projects/vig/Datasets/stanfordORB}"
LOCAL_DATASET_BASE="${LOCAL_DATASET_BASE:-${HOME}/Datasets/stanfordORB}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_ACCELERATE_CONFIG_DIR="${HF_ACCELERATE_CONFIG_DIR:-${HOME}/.cache/huggingface/accelerate}"

LVSM_CKPT_512="${LVSM_CKPT_512:-ckpt/LVSM_scene_encoder_decoder_512}"
RESUME_CKPT="${RESUME_CKPT:-ckpt/realworld_exps_relight_stanfordORB}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-ckpt/realworld_exps_relight_stanfordORB_512}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" --nnodes 1 \
    --rdzv_id 28637 --rdzv_backend c10d --rdzv_endpoint localhost:29524 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_stanfordORB_crosssplit_512.yaml \
    training.dataset_path = "${DATASET_ROOT}/test/full_list.txt" \
    training.context_dataset_path = "${DATASET_ROOT}/train/full_list.txt" \
    training.cross_split_relight = true \
    training.batch_size_per_gpu = 4 \
    training.og_dataset_base = "${OG_DATASET_BASE}" \
    training.local_dataset_base = "${LOCAL_DATASET_BASE}" \
    training.editor_condition_source = target \
    training.LVSM_checkpoint_dir = "${LVSM_CKPT_512}" \
    training.resume_ckpt = "${RESUME_CKPT}" \
    training.checkpoint_dir = "${CHECKPOINT_DIR}" \
    training.wandb_exp_name = LVSM_edit_stanfordORB_crosssplit_512_nebius \
    training.warmup = 1500 \
    training.vis_every = 1000 \
    training.save_every = 1000 \
    training.lr = 0.0001
