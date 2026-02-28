#!/bin/bash
# Preprocess test data on Sony cluster (full list)
# Usage: source bash_scripts/Sony_clusters/interactive_preprocess_test_full.sh
#   or:  bash bash_scripts/Sony_clusters/interactive_preprocess_test_full.sh

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"

# Input/Output paths (Sony cluster)
export INPUT_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse/rendered_dense_polyhaven"
export OUTPUT_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm"
export OUTPUT_TAR="/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm_tar"
export HDRI_DIR="/music-shared-disk/group/ct/yiwen/data/objaverse/hdris_256"

############################
# Logging
############################
echo "Host: $(hostname)"
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
singularity exec $BIND $SIF bash -lc "
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
