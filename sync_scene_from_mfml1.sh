#!/bin/bash
# Sync specific scene from mfml1 to local
# Usage: bash sync_scene_from_mfml1.sh

set -euo pipefail

############################
# Configuration
############################
REMOTE_HOST="mfml1"
REMOTE_PATH="/music-shared-disk/group/ct/yiwen/data/objaverse/rendered_scenes_dense/6c70f11081e4438b878f4c007a48ab65"
LOCAL_PATH="data_samples/scene_light_combined"

############################
# Create local directory
############################
mkdir -p "$LOCAL_PATH"

############################
# Sync using rsync
############################
echo "Syncing from mfml1..."
echo "Remote: $REMOTE_HOST:$REMOTE_PATH"
echo "Local:  $LOCAL_PATH"
echo ""

rsync -avzP \
  --human-readable \
  "$REMOTE_HOST:$REMOTE_PATH" \
  "$LOCAL_PATH/"

############################
# Done
############################
echo ""
echo "✓ Sync complete!"
echo "Files saved to: $LOCAL_PATH/6c70f11081e4438b878f4c007a48ab65/"
