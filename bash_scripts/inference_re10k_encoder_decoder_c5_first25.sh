#!/usr/bin/env bash
# Encoder–decoder LVSM eval on the 25-scene RE10K split (data/re10k_c5_64_first25.json).
# Writes the usual per-scene folders (metadata, metrics, input.png, gt_vs_pred.png) plus:
#   ${OUT_DIR}/context/<scene>/<frame_idx>.png
#   ${OUT_DIR}/gt/<scene>/<frame_idx>.png
#   ${OUT_DIR}/predicted/<scene>/<frame_idx>.png
#   ${OUT_DIR}/test.log   (stdout/stderr from this run)
#
# Usage on a GPU node:
#   export CHECKPOINT=/path/to/scene_encoder_decoder_256.pt
#   export DATASET_PATH=/path/to/preprocessed_data/re10k_c5_64_first25/test/full_list.txt
#   bash bash_scripts/inference_re10k_encoder_decoder_c5_first25.sh
#
# Slurm: copy the #SBATCH block below to the top of this file (before set -e), set partition/account,
#        then: sbatch bash_scripts/inference_re10k_encoder_decoder_c5_first25.sh
#
# Checkpoint (paper / README): https://huggingface.co/coast01/LVSM/resolve/main/scene_encoder_decoder_256.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to scene_encoder_decoder_256.pt (or your .pt path)}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to preprocessed test/full_list.txt for the 25 scenes}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/evaluation/re10k_c5_first25_encoder_decoder}"
VIEW_IDX="${VIEW_IDX:-${REPO_ROOT}/data/re10k_c5_64_first25.json}"

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29506}"

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/test.log"

{
  echo "========== $(date -Is) =========="
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "CHECKPOINT=${CHECKPOINT}"
  echo "DATASET_PATH=${DATASET_PATH}"
  echo "VIEW_IDX=${VIEW_IDX}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "NPROC=${NPROC}"
  echo "=============================================="

  torchrun --nproc_per_node="${NPROC}" --nnodes=1 \
    --rdzv_id=18635 --rdzv_backend=c10d --rdzv_endpoint="localhost:${MASTER_PORT}" \
    inference.py \
    --config configs/LVSM_scene_encoder_decoder.yaml \
    training.checkpoint_dir="${CHECKPOINT}" \
    training.dataset_path="${DATASET_PATH}" \
    training.dataset_name=data.dataset_scene_og.Dataset \
    training.batch_size_per_gpu=1 \
    training.num_workers=2 \
    training.num_threads=4 \
    training.num_views=69 \
    training.num_input_views=5 \
    training.num_target_views=64 \
    training.square_crop=true \
    training.target_has_input=false \
    inference.if_inference=true \
    inference.compute_metrics=true \
    inference.export_split_layout=true \
    inference.view_idx_file_path="${VIEW_IDX}" \
    inference.generate_website=false \
    inference.render_video=false \
    inference_out_dir="${OUT_DIR}"
} 2>&1 | tee "${LOG_FILE}"

echo "Finished. Logs: ${LOG_FILE}"
