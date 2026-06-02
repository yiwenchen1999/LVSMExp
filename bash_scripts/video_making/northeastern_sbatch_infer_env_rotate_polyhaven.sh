#!/bin/bash
#SBATCH --job-name=infer_polyhaven_env_rotate
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJ="/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp"
DATASET_FULL_LIST="/projects/vig/Datasets/objaverse/hf-objaverse-v1/polyhaven_env_rotate/test/full_list.txt"
CHECKPOINT_DIR="/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt_dpt/dpt_decoder_512_1e5"
LVSM_CKPT_DIR="$PROJ/ckpt/LVSM_object_encoder_decoder_512"
EVAL_INDEX="$PROJ/data/evaluation_index_polyhaven_env_rotate_4in4out.json"
OUT_DIR="$PROJ/experiments/evaluation/polyhaven_env_rotate"

NUM_INPUT_VIEWS=4
NUM_TARGET_VIEWS=4
NUM_VIEWS=$((NUM_INPUT_VIEWS + NUM_TARGET_VIEWS))

cd "$PROJ"

# Optional: activate the same environment used in shortcut.sh when conda is available.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate /projects/vig/yiwenc/all_env/rayzer
fi

if [ ! -f "$DATASET_FULL_LIST" ]; then
  echo "ERROR: dataset full_list not found: $DATASET_FULL_LIST"
  exit 1
fi

if [ ! -f "$EVAL_INDEX" ] || [ "${REBUILD_EVAL_INDEX:-0}" = "1" ]; then
  echo "Building evaluation index: $EVAL_INDEX"
  python preprocess_scripts/create_evaluation_index.py \
    --full-list "$DATASET_FULL_LIST" \
    --output "$EVAL_INDEX" \
    --n-input "$NUM_INPUT_VIEWS" \
    --n-target "$NUM_TARGET_VIEWS" \
    --seed 42
fi

torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id "${SLURM_JOB_ID:-18635}" --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
exp_rotate_env.py --config "configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_res_singleMap_dpt_transfer.yaml" \
training.dataset_path = "$DATASET_FULL_LIST" \
training.checkpoint_dir = "$CHECKPOINT_DIR" \
training.LVSM_checkpoint_dir = "$LVSM_CKPT_DIR" \
training.batch_size_per_gpu = 1 \
training.target_has_input = false \
training.num_views = "$NUM_VIEWS" \
training.square_crop = true \
training.num_input_views = "$NUM_INPUT_VIEWS" \
training.num_target_views = "$NUM_TARGET_VIEWS" \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = false \
inference.view_idx_file_path = "$EVAL_INDEX" \
inference_out_dir = "$OUT_DIR"
