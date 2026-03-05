#!/bin/bash
# Interactive degradation test for iterative editing (Sony cluster)
#
# Runs both image-space and token-space degradation experiments using the
# same checkpoint and envmap sequence for fair comparison.
#
# Usage:
#   bash bash_scripts/Sony_clusters/SonyAIClusterUtil/interactive_degrade_test.sh
#
# Optional env overrides (set before running):
#   NUM_SCENES=50        # number of scenes to process (default: 50)
#   NUM_ITER=50          # number of editing iterations per scene (default: 50)
#   NUM_INPUT_VIEWS=4    # context views (default: 4)
#   SAVE_IMAGES=false    # save per-step images (default: false for multi-scene)

set -euo pipefail

############################
# Tunables
############################
NUM_SCENES="${NUM_SCENES:-50}"
NUM_ITER="${NUM_ITER:-20}"
NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-4}"
SAVE_IMAGES="${SAVE_IMAGES:-false}"

############################
# Paths & environment
# (inherited from interactive_inference_relight_general_dense_lr1e4_10k.sh)
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

# WANDB directories
export WANDB_DIR=/scratch2/$USER/wandb
export WANDB_ARTIFACT_DIR=/scratch2/$USER/wandb/artifacts
export WANDB_CACHE_DIR=/scratch2/$USER/wandb/cache
export WANDB_CONFIG_DIR=/scratch2/$USER/wandb/config

# Cache directories
export XDG_CACHE_HOME=/scratch2/$USER/.cache
export XDG_CONFIG_HOME=/scratch2/$USER/.config
export XDG_DATA_HOME=/scratch2/$USER/.local/share

# HuggingFace cache
export HF_HOME=/scratch2/$USER/.cache/huggingface
export HF_ACCELERATE_CONFIG_DIR=/scratch2/$USER/.cache/accelerate

# Data & checkpoint paths (same as dense_relight_env inference)
export DATA_LIST="/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm/test/full_list.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/dense_relight_env"
export LVSM_CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"

# Output directories
export OUT_IMAGE_SPACE="$PROJ/experiments/degrade_test_imageSpace"
export OUT_TOKEN_SPACE="$PROJ/experiments/degrade_test_tokenSpace"

############################
# Logging
############################
echo "=============================================="
echo "Iterative Editing Degradation Test (Sony)"
echo "=============================================="
echo "Host          : $(hostname)"
echo "Num scenes    : $NUM_SCENES"
echo "Iterations    : $NUM_ITER"
echo "Input views   : $NUM_INPUT_VIEWS"
echo "Save images   : $SAVE_IMAGES"
echo "CKPT_DIR      : $CKPT_DIR"
echo "LVSM_CKPT_DIR : $LVSM_CKPT_DIR"
echo "DATA_LIST     : $DATA_LIST"
echo "Out (image)   : $OUT_IMAGE_SPACE"
echo "Out (token)   : $OUT_TOKEN_SPACE"
echo "----------------------------------------------"
echo ""

############################
# Shared singularity env block
############################
SING_ENV="
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
"

# Shared torchrun + config overrides
SHARED_ARGS="\
    --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
    training.batch_size_per_gpu = 1 \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    training.dataset_path = \"$DATA_LIST\" \
    training.LVSM_checkpoint_dir = \"$LVSM_CKPT_DIR\" \
    training.single_env_map = true \
    training.num_input_views = $NUM_INPUT_VIEWS \
    training.num_views = 12 \
    inference.if_inference = true \
    inference.view_idx_file_path = \"$PROJ/data/evaluation_index_polyhaven_dense.json\" \
    inference.degrade_num_scenes = $NUM_SCENES \
    inference.degrade_num_iterations = $NUM_ITER \
    inference.degrade_save_images = $SAVE_IMAGES"

############################
# 1. Image-space experiment
############################
echo "[1/2] Running IMAGE-SPACE degradation test ..."
singularity exec --nv $BIND $SIF bash -lc "
  $SING_ENV

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29506 \
    inference_editor_degrade_test_imageSpace.py \
    $SHARED_ARGS \
    inference_out_dir = \"$OUT_IMAGE_SPACE\"
"
echo "[1/2] Image-space test complete."
echo ""

############################
# 2. Token-space experiment
############################
echo "[2/2] Running TOKEN-SPACE degradation test ..."
singularity exec --nv $BIND $SIF bash -lc "
  $SING_ENV

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id \$(date +%s) \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29506 \
    inference_editor_degrade_test_tokenSpace.py \
    $SHARED_ARGS \
    inference_out_dir = \"$OUT_TOKEN_SPACE\"
"
echo "[2/2] Token-space test complete."

############################
# 3. Plot comparison
############################
echo ""
echo "[3/3] Plotting average degradation curves ..."

singularity exec --nv $BIND $SIF bash -lc "
  $SING_ENV

  python3 - <<'PYEOF'
import csv
import os
import numpy as np

img_csv = '$OUT_IMAGE_SPACE/all_scenes_avg.csv'
tok_csv = '$OUT_TOKEN_SPACE/all_scenes_avg.csv'
out_png = '$PROJ/experiments/degrade_avg_comparison.png'
out_csv = '$PROJ/experiments/degrade_avg_comparison.csv'

def read_avg_csv(path):
    steps, psnr, ssim, lpips_v = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            psnr.append(float(row['avg_psnr']))
            ssim.append(float(row['avg_ssim']))
            lpips_v.append(float(row['avg_lpips']))
    return np.array(steps), np.array(psnr), np.array(ssim), np.array(lpips_v)

img_steps, img_psnr, img_ssim, img_lpips = read_avg_csv(img_csv)
tok_steps, tok_psnr, tok_ssim, tok_lpips = read_avg_csv(tok_csv)

# Save combined CSV
with open(out_csv, 'w') as f:
    f.write('step,img_avg_psnr,img_avg_ssim,img_avg_lpips,tok_avg_psnr,tok_avg_ssim,tok_avg_lpips\n')
    for i in range(len(img_steps)):
        ti = i if i < len(tok_steps) else len(tok_steps) - 1
        f.write(f'{img_steps[i]},{img_psnr[i]:.4f},{img_ssim[i]:.6f},{img_lpips[i]:.6f},'
                f'{tok_psnr[ti]:.4f},{tok_ssim[ti]:.6f},{tok_lpips[ti]:.6f}\n')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    c_img, c_tok = '#2196F3', '#F44336'

    axes[0].plot(img_steps, img_psnr, '-o', color=c_img, ms=3, lw=1.5, label='Image Space', alpha=0.85)
    axes[0].plot(tok_steps, tok_psnr, '-s', color=c_tok, ms=3, lw=1.5, label='Token Space', alpha=0.85)
    axes[0].set_xlabel('Iteration Step'); axes[0].set_ylabel('PSNR (dB) ↑')
    axes[0].set_title('Avg PSNR Degradation', fontweight='bold'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(img_steps, img_ssim, '-o', color=c_img, ms=3, lw=1.5, label='Image Space', alpha=0.85)
    axes[1].plot(tok_steps, tok_ssim, '-s', color=c_tok, ms=3, lw=1.5, label='Token Space', alpha=0.85)
    axes[1].set_xlabel('Iteration Step'); axes[1].set_ylabel('SSIM ↑')
    axes[1].set_title('Avg SSIM Degradation', fontweight='bold'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(img_steps, img_lpips, '-o', color=c_img, ms=3, lw=1.5, label='Image Space', alpha=0.85)
    axes[2].plot(tok_steps, tok_lpips, '-s', color=c_tok, ms=3, lw=1.5, label='Token Space', alpha=0.85)
    axes[2].set_xlabel('Iteration Step'); axes[2].set_ylabel('LPIPS ↓')
    axes[2].set_title('Avg LPIPS Degradation', fontweight='bold'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    n = int(open(img_csv).readlines()[1].split(',')[-1])
    fig.suptitle(f'Avg Iterative Editing Degradation ({n} scenes)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {out_png}')
except ImportError:
    print('matplotlib not available — skipping plot, CSV saved to', out_csv)
PYEOF
"

############################
# Done
############################
echo ""
echo "=============================================="
echo "Both degradation tests finished."
echo "=============================================="
echo "Image-space results: $OUT_IMAGE_SPACE"
echo "Token-space results: $OUT_TOKEN_SPACE"
echo "Combined CSV       : $PROJ/experiments/degrade_avg_comparison.csv"
echo "Plot               : $PROJ/experiments/degrade_avg_comparison.png"
echo ""
