#!/bin/bash
set -euo pipefail

# Local single-node launch
# Usage:
#   bash bash_scripts/train_editor_pointlight.sh

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

CONFIG_PATH="configs/LVSM_scene_encoder_decoder_wEditor_pointlight.yaml"

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_pointlight.yaml \
    training.batch_size_per_gpu = 8 \
    training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_pointlight \
    training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_dense \
    training.wandb_exp_name = LVSM_edit_pointlight \
    training.dataset_path = /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse/test/full_list.txt \
    training.whiteEnvInput = true \
    training.lr = 0.0002
