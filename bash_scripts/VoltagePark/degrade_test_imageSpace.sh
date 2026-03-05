#!/bin/bash
# Degradation test — IMAGE SPACE
# Iteratively: reconstruct -> edit(envmap) -> render -> use rendered as new input.
#
# Usage: bash bash_scripts/VoltagePark/degrade_test_imageSpace.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

############################
# VoltagePark paths
############################
export PROJ="${PROJ:-$REPO_ROOT}"
export DATA_LIST="${DATA_LIST:-/data/lvsm_scenes_dense/test/full_list.txt}"
export CKPT_DIR="${CKPT_DIR:-$PROJ/ckpt/relight_combined_dense}"
export LVSM_CKPT_DIR="${LVSM_CKPT_DIR:-$PROJ/ckpt/LVSM_scene_encoder_decoder_dense}"
export EVAL_INDEX="${EVAL_INDEX:-$PROJ/data/evaluation_index_scenes_comb.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJ/experiments/degradation_exp/image_space}"

if [[ -n "${NPROC:-}" ]]; then
    NPROC_PER_NODE="$NPROC"
else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l) || NPROC_PER_NODE=1
fi
NNODES="${NNODES:-1}"

echo "=============================================="
echo "Degradation Test — IMAGE SPACE (VoltagePark)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "EVAL_INDEX: $EVAL_INDEX"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "----------------------------------------------"

if [ ! -f "$EVAL_INDEX" ]; then
  echo "ERROR: Evaluation index not found: $EVAL_INDEX"
  exit 1
fi

torchrun --nproc_per_node "$NPROC_PER_NODE" --nnodes "$NNODES" \
    --rdzv_id "$(date +%s)" --rdzv_backend c10d --rdzv_endpoint localhost:29501 \
    inference_editor_degrade_test_imageSpace.py \
    --config configs/LVSM_scene_encoder_decoder_wEditor_envmap_pointlight.yaml \
  training.dataset_path = /data/polyhaven_lvsm/test/full_list.txt \
  training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4_singleMap \
  training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
  training.batch_size_per_gpu = 4 \
  training.target_has_input = false \
  training.num_views = 12 \
  training.square_crop = true \
  training.num_input_views = 4 \
  training.num_target_views = 8 \
  inference.if_inference = true \
  inference.compute_metrics = true \
  inference.render_video = false \
  inference.same_pose = True \
  inference.view_idx_file_path = data/evaluation_index_polyhaven.json \
    inference_out_dir = "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Degradation test (image space) complete!"
echo "Results: $OUTPUT_DIR"
echo "=============================================="
