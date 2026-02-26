#!/bin/bash
# Finetune on lvsmPlus_objaverse (VoltagePark - dedicated GPU, no sbatch).
# Usage: bash bash_scripts/VoltagePark/finetune_residual.sh
#   or:  source bash_scripts/VoltagePark/finetune_residual.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ############################
# # VoltagePark paths
# ############################
export PROJ="${PROJ:-$REPO_ROOT}"
export DATA_LIST="${DATA_LIST:-/data/lvsmPlus_objaverse/train/full_list.txt}"
export CKPT_DIR="${CKPT_DIR:-$PROJ/ckpt/LVSM_scene_encoder_decoder_wEditor_residual}"

# Detect GPU count (override with NPROC env var)
if [[ -n "${NPROC:-}" ]]; then
    NPROC_PER_NODE="$NPROC"
else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l) || NPROC_PER_NODE=1
fi
NNODES="${NNODES:-1}"

############################
# Logging
############################
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "nnodes: $NNODES"
echo "----------------------------------"

############################
# Run training
############################

torchrun --nproc_per_node "$NPROC_PER_NODE" --nnodes "$NNODES" \
    --rdzv_id "$(date +%s)" --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    train.py --config configs/LVSM_scene_encoder_decoder_wEditor_residual.yaml \
    training.batch_size_per_gpu = 8 \
    training.dataset_path = "$DATA_LIST" \
    training.checkpoint_dir = "$CKPT_DIR" \
    training.grad_accum_steps = 1
