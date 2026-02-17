#!/bin/bash
# Interactive script for preview_scenes and remove_broken_scenes on Sony cluster.
# Usage: source bash_scripts/Sony_clusters/interactive_preview_remove_broken.sh
#   or:  bash bash_scripts/Sony_clusters/interactive_preview_remove_broken.sh

set -euo pipefail

############################
# Paths & environment
############################
export PROJ=/music-shared-disk/group/ct/yiwen/codes/LVSMExp
export PY_SITE=/scratch2/$USER/py_lvsmexp
export SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/data,/music-shared-disk"

# Sony cluster paths (map from shortcut.sh /projects/vig/...)
export FULL_LIST_TEST="/music-shared-disk/group/ct/yiwen/data/objaverse/lvsmPlus_objaverse/test/full_list.txt"
export FULL_LIST_TRAIN="/music-shared-disk/group/ct/yiwen/data/objaverse/lvsmPlus_objaverse/train/full_list.txt"
export PREVIEW_OUTPUT="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/scene_preview/preview.png"
export BROKEN_SCENE_FILE="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/scene_preview/broken_scene.txt"

############################
# Step 1: Update paths (run first)
############################
echo "Updating paths (old -> new)..."
singularity exec $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
  cd $PROJ

  python3 preprocess_scripts/update_paths.py \
    --old-path \"/scratch/chen.yiwe/temp_objaverse\" \
    --new-path \"/music-shared-disk/group/ct/yiwen/data/objaverse\" \
    --root-dir /music-shared-disk/group/ct/yiwen/data/objaverse/lvsmPlus_objaverse/test \
    --extensions json txt \
    --backup
"
echo "Path update done."
echo ""

############################
# Menu
############################
echo "========================================"
echo "Preview / Remove Broken Scenes (Sony)"
echo "========================================"
echo "1) preview_scenes.py   - Generate scene preview grid"
echo "2) remove_broken_scenes.py - Remove broken scenes from full_list"
echo "3) Both (preview then remove)"
echo "4) Exit"
echo "----------------------------------------"
read -p "Choice [1-4]: " choice

run_preview() {
  singularity exec $BIND $SIF bash -lc "
    set -euo pipefail
    export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
    cd $PROJ

    python preprocess_scripts/preview_scenes.py \
      --full-list $FULL_LIST_TEST \
      --output $PREVIEW_OUTPUT \
      --image-idx 64 \
      --grid-cols 8 \
      --grid-rows 4 \
      --images-per-grid 32
  "
}

run_remove_broken() {
  singularity exec $BIND $SIF bash -lc "
    set -euo pipefail
    export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
    cd $PROJ

    python preprocess_scripts/remove_broken_scenes.py \
      --broken-scene $BROKEN_SCENE_FILE \
      --full-list $FULL_LIST_TEST
  "
}

case "$choice" in
  1) run_preview ;;
  2) run_remove_broken ;;
  3) run_preview && run_remove_broken ;;
  4) echo "Exiting." ; exit 0 ;;
  *) echo "Invalid choice." ; exit 1 ;;
esac
