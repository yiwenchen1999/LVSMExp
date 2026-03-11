#!/bin/bash
# Inference for model trained by relight_general_dense_lr1e4_combined_dense.sh
# with condition_reverse mode (input from relit scene, relit/lighting from current scene).
# Uses evaluation_index_scenes_comb.json for view indices.
#
# Usage: bash bash_scripts/VoltagePark/inference_combined_dense_condition_reverse.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

############################
# VoltagePark paths
############################
export PROJ="${PROJ:-$REPO_ROOT}"
export DATA_LIST="${DATA_LIST:-/data/lvsm_scenes_dense/test/full_list_scenes.txt}"
export CKPT_DIR="${CKPT_DIR:-$PROJ/ckpt/relight_combined_scenes}"
export LVSM_CKPT_DIR="${LVSM_CKPT_DIR:-$PROJ/ckpt/LVSM_scene_encoder_decoder_dense}"
export EVAL_INDEX="${EVAL_INDEX:-$PROJ/data/demo_scene_rotate.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJ/experiments/evaluation/demo_scene_rotate}"

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
echo "=============================================="
echo "Inference - condition_reverse (VoltagePark)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "EVAL_INDEX: $EVAL_INDEX"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "nnodes: $NNODES"
echo "MODE: condition_reverse"
echo "----------------------------------------------"
echo ""

############################
# Check evaluation index
############################
if [ ! -f "$EVAL_INDEX" ]; then
  echo "ERROR: Evaluation index not found: $EVAL_INDEX"
  exit 1
fi

############################
# Run inference
############################
echo "Starting condition_reverse inference..."

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id "$(date +%s)" --rdzv_backend c10d --rdzv_endpoint localhost:29501 \
    inference_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_envmap_pointlight.yaml \
    training.dataset_path = "$DATA_LIST" \
    training.checkpoint_dir = "$CKPT_DIR" \
    training.LVSM_checkpoint_dir = "$LVSM_CKPT_DIR" \
    training.batch_size_per_gpu = 4 \
    training.target_has_input = false \
    training.num_views = 12 \
    training.square_crop = true \
    training.num_input_views = 4 \
    training.num_target_views = 8 \
    training.condition_reverse = true \
    training.single_env_map = true \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    inference.view_idx_file_path = "$EVAL_INDEX" \
    inference_out_dir = "$OUTPUT_DIR"

############################
# Done
############################
echo ""
echo "=============================================="
echo "Inference complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
