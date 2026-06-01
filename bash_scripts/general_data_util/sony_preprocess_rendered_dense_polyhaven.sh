#!/usr/bin/env bash
set -euo pipefail

# Preprocess rendered_dense_polyhaven on Sony cluster.
# Default output keeps train/test under:
#   /music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm
#
# Usage examples:
#   bash bash_scripts/general_data_util/sony_preprocess_rendered_dense_polyhaven.sh
#   REFRESH_ONLY=1 bash bash_scripts/general_data_util/sony_preprocess_rendered_dense_polyhaven.sh
#   SPLITS="test" MAX_OBJECTS=50 bash bash_scripts/general_data_util/sony_preprocess_rendered_dense_polyhaven.sh

#############################################
# Paths and cluster environment
#############################################
PROJ="${PROJ:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp}"
INPUT_ROOT="${INPUT_ROOT:-/music-shared-disk/group/ct/yiwen/data/objaverse/rendered_dense_polyhaven}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/music-shared-disk/group/ct/yiwen/data/objaverse/polyhaven_lvsm}"
HDRI_DIR="${HDRI_DIR:-/music-shared-disk/group/ct/yiwen/data/objaverse/hdris}"

# Optional location-B tar output. Leave empty to disable tar export.
OUTPUT_TAR_ROOT="${OUTPUT_TAR_ROOT:-}"

PY_SITE="${PY_SITE:-/scratch2/$USER/py_lvsmexp}"
SIF="${SIF:-/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif}"
BIND="${BIND:--B /group2,/scratch2,/music-shared-disk}"

#############################################
# Behavior knobs
#############################################
SPLITS="${SPLITS:-test}"
MAX_OBJECTS="${MAX_OBJECTS:-}"
POINT_LIGHT_RAYS_N="${POINT_LIGHT_RAYS_N:-8192}"
SCENE_SPHERE_RADIUS="${SCENE_SPHERE_RADIUS:-3.0}"

# If set to 1, only regenerate full_list.txt from existing metadata.
REFRESH_ONLY="${REFRESH_ONLY:-0}"

echo "========================================"
echo "Sony preprocess: rendered_dense_polyhaven"
echo "Host: $(hostname)"
echo "PROJ: ${PROJ}"
echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "OUTPUT_TAR_ROOT: ${OUTPUT_TAR_ROOT:-<disabled>}"
echo "SPLITS: ${SPLITS}"
echo "REFRESH_ONLY: ${REFRESH_ONLY}"
echo "========================================"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "ERROR: INPUT_ROOT does not exist: ${INPUT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${PROJ}" ]]; then
  echo "ERROR: PROJ does not exist: ${PROJ}" >&2
  exit 1
fi
if [[ ! -f "${SIF}" ]]; then
  echo "ERROR: SIF image not found: ${SIF}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/test" "${OUTPUT_ROOT}/train"
if [[ -n "${OUTPUT_TAR_ROOT}" ]]; then
  mkdir -p "${OUTPUT_TAR_ROOT}/test" "${OUTPUT_TAR_ROOT}/train"
fi

run_preprocess_split() {
  local split="$1"
  local tar_args=""
  if [[ -n "${OUTPUT_TAR_ROOT}" ]]; then
    tar_args="--output-tar \"${OUTPUT_TAR_ROOT}\""
  else
    tar_args="--no-output-tar"
  fi

  local max_objects_args=""
  if [[ -n "${MAX_OBJECTS}" ]]; then
    max_objects_args="--max-objects ${MAX_OBJECTS}"
  fi

  echo ""
  echo "[Preprocess] split=${split}"
  singularity exec --nv ${BIND} "${SIF}" bash -lc "
    set -euo pipefail
    export PYTHONPATH=\"${PY_SITE}:\${PYTHONPATH:-}\"
    export OPENCV_IO_ENABLE_OPENEXR=1
    export QT_QPA_PLATFORM=offscreen
    export PYOPENGL_PLATFORM=egl
    cd \"${PROJ}\"
    python preprocess_scripts/preprocess_objaverse.py \
      --input \"${INPUT_ROOT}\" \
      --output \"${OUTPUT_ROOT}\" \
      ${tar_args} \
      --split \"${split}\" \
      --hdri-dir \"${HDRI_DIR}\" \
      --point-light-rays-n ${POINT_LIGHT_RAYS_N} \
      --scene-sphere-radius ${SCENE_SPHERE_RADIUS} \
      ${max_objects_args}
  "
}

refresh_full_list_split() {
  local split="$1"
  local tar_args=""
  if [[ -n "${OUTPUT_TAR_ROOT}" ]]; then
    tar_args="--output-tar \"${OUTPUT_TAR_ROOT}\""
  else
    tar_args="--no-output-tar"
  fi

  echo ""
  echo "[Refresh full_list] split=${split}"
  singularity exec --nv ${BIND} "${SIF}" bash -lc "
    set -euo pipefail
    export PYTHONPATH=\"${PY_SITE}:\${PYTHONPATH:-}\"
    cd \"${PROJ}\"
    python preprocess_scripts/preprocess_objaverse.py \
      --output \"${OUTPUT_ROOT}\" \
      ${tar_args} \
      --split \"${split}\" \
      --full-list-only
  "
}

if [[ "${REFRESH_ONLY}" != "1" ]]; then
  for split in ${SPLITS}; do
    run_preprocess_split "${split}"
  done
else
  echo "REFRESH_ONLY=1, skipping preprocessing and only rebuilding full_list.txt"
fi

# Always rebuild list from metadata to ensure full_list contains all valid scenes.
for split in ${SPLITS}; do
  refresh_full_list_split "${split}"
done

echo ""
echo "Done."
echo "Output lists:"
for split in ${SPLITS}; do
  echo "  ${OUTPUT_ROOT}/${split}/full_list.txt"
done
