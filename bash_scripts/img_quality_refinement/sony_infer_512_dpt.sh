#!/usr/bin/env bash
# Inference for 512x512 DPT-transfer editor ckpt (trained with nebius_train_dense_512_dpt.sh).
# Mirrors bash_scripts/img_quality_refinement/sony_infer_512.sh (paths, Singularity) but runs
# inference_editor.py + configs/.../general_dense_512_res_singleMap_dpt_transfer.yaml.
#
# Frame / view sampling parity with sony_infer_512.sh:
#   - training.seed=777 (default; sony_infer_512 does not override seed → 777 in train_editor)
#   - training.view_selector min_frame_dist/max_frame_dist = 15 / 60 (same as singleMap + DPT yamls)
#   - training.single_env_map=false (matches general_dense_512_singleMap.yaml used by sony_infer_512)
#   - inference.view_idx_file_path=null in base yaml so dataset uses random view_selector (not eval JSON)
#
# Usage:
#   bash bash_scripts/img_quality_refinement/sony_infer_512_dpt.sh
#   CKPT_DIR=... OUTPUT_DIR=... DATASET_PATH=... bash ...

set -euo pipefail

############################
# Paths & environment (Sony)
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

export DATASET_PATH="${DATASET_PATH:-/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm/test/full_list.txt}"
export CKPT_DIR="${CKPT_DIR:-$PROJ/ckpt/dpt_decoder_512_1e5}"
export LVSM_CKPT_DIR="${LVSM_CKPT_DIR:-$PROJ/ckpt/LVSM_object_encoder_decoder_512}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJ/experiments/evaluation/infer_512_dpt_transfer}"

export TRAINING_SEED="${TRAINING_SEED:-777}"

############################
# Logging
############################
echo "=============================================="
echo "Inference: 512 DPT-transfer editor (Sony)"
echo "=============================================="
echo "Host: $(hostname)"
echo "PROJ: $PROJ"
echo "DATASET_PATH: $DATASET_PATH"
echo "CKPT_DIR: $CKPT_DIR"
echo "LVSM_CKPT_DIR: $LVSM_CKPT_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TRAINING_SEED (view sampling): $TRAINING_SEED"
echo "----------------------------------------------"
echo ""

if [ ! -d "$DATASET_PATH" ] && [ ! -f "$DATASET_PATH" ]; then
  echo "ERROR: Dataset path not found: $DATASET_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

############################
# Run (torchrun inside Singularity)
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
  cd $PROJ

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29511 \
    inference_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_res_singleMap_dpt_transfer.yaml \
    training.batch_size_per_gpu = 1 \
    training.dataset_path = \"$DATASET_PATH\" \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.seed = $TRAINING_SEED \
    training.single_env_map = false \
    training.view_selector.min_frame_dist = 15 \
    training.view_selector.max_frame_dist = 60 \
    training.num_input_views = 4 \
    training.num_target_views = 8 \
    training.num_views = 12 \
    training.target_has_input = true \
    training.relight_signals = \"[envmap]\" \
    training.warmup = 3000 \
    training.vis_every = 1 \
    training.lr = 0.0000 \
"

echo ""
echo "=============================================="
echo "Done. Results: $OUTPUT_DIR"
echo "=============================================="
