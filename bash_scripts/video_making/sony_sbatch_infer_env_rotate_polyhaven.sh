#!/bin/bash
#SBATCH --job-name=video_infer_env_rotate
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=/group2/ct/yiwen/logs/%x.%N.%j.err

# Env-rotation inference on the Sony cluster.
# Runs exp_rotate_env.py over the preprocessed rotating-env dataset, using random
# contiguous-chunk sampling (evenly placed context inside the chunk) instead of a
# fixed view-index json. Context frame ids come from the batch (input.index).

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

export WANDB_DIR=/scratch2/$USER/wandb
export WANDB_ARTIFACT_DIR=/scratch2/$USER/wandb/artifacts
export WANDB_CACHE_DIR=/scratch2/$USER/wandb/cache
export WANDB_CONFIG_DIR=/scratch2/$USER/wandb/config

export XDG_CACHE_HOME=/scratch2/$USER/.cache
export XDG_CONFIG_HOME=/scratch2/$USER/.config
export XDG_DATA_HOME=/scratch2/$USER/.local/share

export HF_HOME=/scratch2/$USER/.cache/huggingface
export HF_ACCELERATE_CONFIG_DIR=/scratch2/$USER/.cache/accelerate

############################
# Inference controls
############################
export DATASET_PATH="${DATASET_PATH:-/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm_rotating_env/test/full_list.txt}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJ/ckpt_dpt/dpt_decoder_512_1e5}"
export LVSM_CKPT_DIR="${LVSM_CKPT_DIR:-$PROJ/ckpt/LVSM_object_encoder_decoder_512}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJ/experiments/evaluation/video_making_env_rotate}"

# Random contiguous-chunk sampling: chunk_len = num_input_views + num_target_views.
export NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-4}"
export NUM_TARGET_VIEWS="${NUM_TARGET_VIEWS:-8}"
export RANDOM_CHUNK_SAMPLING="${RANDOM_CHUNK_SAMPLING:-true}"
export RANDOM_CHUNK_SEED="${RANDOM_CHUNK_SEED:-}"
export RENDER_VIDEO="${RENDER_VIDEO:-false}"
export COMPUTE_METRICS="${COMPUTE_METRICS:-true}"

NUM_VIEWS=$(( NUM_INPUT_VIEWS + NUM_TARGET_VIEWS ))

############################
# Logging
############################
echo "=============================================="
echo "SBATCH: exp_rotate_env (polyhaven rotating-env)"
echo "=============================================="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-N/A}"
echo "DATASET_PATH: $DATASET_PATH"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NUM_INPUT_VIEWS: $NUM_INPUT_VIEWS"
echo "NUM_TARGET_VIEWS: $NUM_TARGET_VIEWS"
echo "NUM_VIEWS (chunk_len): $NUM_VIEWS"
echo "RANDOM_CHUNK_SAMPLING: $RANDOM_CHUNK_SAMPLING"
echo "RANDOM_CHUNK_SEED: ${RANDOM_CHUNK_SEED:-<unset>}"
echo "RENDER_VIDEO: $RENDER_VIDEO"
echo "COMPUTE_METRICS: $COMPUTE_METRICS"
echo "----------------------------------------------"
echo ""

if [ ! -f "$DATASET_PATH" ]; then
  echo "ERROR: Dataset list not found: $DATASET_PATH"
  exit 1
fi

# Optional random_chunk_seed: only pass the override if the user set it.
SEED_ARG=""
if [ -n "${RANDOM_CHUNK_SEED}" ]; then
  SEED_ARG="inference.random_chunk_seed = ${RANDOM_CHUNK_SEED}"
fi
export SEED_ARG

############################
# Run inference
############################
singularity exec --nv $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:\${PYTHONPATH:-}\"
  export WANDB_DIR=\"$WANDB_DIR\"
  export WANDB_ARTIFACT_DIR=\"$WANDB_ARTIFACT_DIR\"
  export WANDB_CACHE_DIR=\"$WANDB_CACHE_DIR\"
  export WANDB_CONFIG_DIR=\"$WANDB_CONFIG_DIR\"
  export XDG_CACHE_HOME=\"$XDG_CACHE_HOME\"
  export XDG_CONFIG_HOME=\"$XDG_CONFIG_HOME\"
  export XDG_DATA_HOME=\"$XDG_DATA_HOME\"
  export HF_HOME=\"$HF_HOME\"
  export HF_ACCELERATE_CONFIG_DIR=\"$HF_ACCELERATE_CONFIG_DIR\"
  cd \"$PROJ\"

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29547 \
    exp_rotate_env.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_res_singleMap_dpt_transfer.yaml \
    training.dataset_path = \"$DATASET_PATH\" \
    training.checkpoint_dir = \"$CHECKPOINT_DIR\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.batch_size_per_gpu = 1 \
    training.target_has_input = false \
    training.num_views = ${NUM_VIEWS} \
    training.num_input_views = ${NUM_INPUT_VIEWS} \
    training.num_target_views = ${NUM_TARGET_VIEWS} \
    inference.if_inference = true \
    inference.compute_metrics = ${COMPUTE_METRICS} \
    inference.render_video = ${RENDER_VIDEO} \
    inference.random_chunk_sampling = ${RANDOM_CHUNK_SAMPLING} \
    \${SEED_ARG} \
    inference_out_dir = \"$OUTPUT_DIR\"
"
