#!/bin/bash
# Finetune on lvsmPlus_objaverse (VoltagePark - dedicated GPU, no sbatch).
# Usage: bash bash_scripts/VoltagePark/finetune_objaverse.sh
#   or:  source bash_scripts/VoltagePark/finetune_objaverse.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ############################
# # VoltagePark paths
# ############################
export PROJ="${PROJ:-$REPO_ROOT}"
export DATA_LIST="${DATA_LIST:-/data/lvsmPlus_objaverse/train/full_list.txt}"
export CKPT_DIR="${CKPT_DIR:-$PROJ/ckpt/LVSM_object_encoder_decoder_sparse}"

# # Caches (use $HOME on clean machine)
# export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_ACCELERATE_CONFIG_DIR="${HF_ACCELERATE_CONFIG_DIR:-$HOME/.cache/accelerate}"
# export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"
# export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$HOME/wandb/artifacts}"
# export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/wandb/cache}"
# export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$HOME/wandb/config}"
# export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
# export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
# export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"

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
    train.py --config configs/LVSM_scene_encoder_decoder_sparse.yaml \
    training.batch_size_per_gpu = 32 \
    training.dataset_path = "$DATA_LIST" \
    training.checkpoint_dir = "$CKPT_DIR" \
    training.grad_accum_steps = 1
