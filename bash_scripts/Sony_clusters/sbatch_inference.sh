#!/bin/bash
#SBATCH --job-name=lvsm_infer_dense_samples
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=bash_scripts/Sony_clusters/sbatch_inference.sh%x.%N.%j.err

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/data,/music-shared-disk"

export DATA_LIST="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/data_samples/objaverse_processed_with_envmaps/test/full_list.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"

############################
# Logging (optional but useful)
############################
echo "Host: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "PROJ: $PROJ"
echo "DATA_LIST: $DATA_LIST"
echo "CKPT_DIR: $CKPT_DIR"
echo "PY_SITE: $PY_SITE"
echo "----------------------------------"

############################
# Run inference
############################
srun singularity exec --nv $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
  cd $PROJ

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29506 \
    inference.py --config configs/LVSM_scene_encoder_decoder.yaml \
    training.dataset_path = \"$DATA_LIST\" \
    training.batch_size_per_gpu = 4 \
    training.target_has_input = false \
    training.num_views = 12 \
    training.square_crop = true \
    training.num_input_views = 4 \
    training.num_target_views = 8 \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    inference.view_idx_file_path = \"data/evaluation_index_objaverse_dense_samples.json\" \
    inference_out_dir = experiments/evaluation/test_dense_reconstruction_samples_sbatch
"
