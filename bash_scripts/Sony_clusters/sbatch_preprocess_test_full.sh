#!/bin/bash
#SBATCH --job-name=preprocess_test_full
#SBATCH --partition=ct_l40s
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=32
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=/group2/ct/yiwen/logs/%x.%N.%j.err

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/data,/music-shared-disk"

# Input/Output paths (Sony cluster)
export INPUT_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse/rendered_dense_polyhaven"
export OUTPUT_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse/lvsm_scenes_test"
export OUTPUT_TAR="/music-shared-disk/group/ct/yiwen/data/objaverse/lvsm_scenes_test_tar"
export HDRI_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse/hdris"

############################
# Logging
############################
echo "Host: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "PROJ: $PROJ"
echo "INPUT_DIR: $INPUT_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "OUTPUT_TAR: $OUTPUT_TAR"
echo "HDRI_DIR: $HDRI_DIR"
echo "PY_SITE: $PY_SITE"
echo "----------------------------------"

############################
# Run preprocessing
############################
srun singularity exec $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
  cd $PROJ

  python preprocess_scripts/preprocess_objaverse.py \
    --input \"$INPUT_DIR\" \
    --output \"$OUTPUT_DIR\" \
    --output-tar \"$OUTPUT_TAR\" \
    --split test \
    --hdri-dir \"$HDRI_DIR\"
"

echo ""
echo "Preprocessing complete!"
echo "Output: $OUTPUT_DIR"
echo "Tar output: $OUTPUT_TAR"
