#!/bin/bash
# Sync specific scene from mfml1 to local
# Usage: bash sync_scene_from_mfml1_generic.sh <scene_id>
# Example: bash sync_scene_from_mfml1_generic.sh 0c09dbb153a44de0826179d04883a1fd

set -euo pipefail

############################
# Configuration
############################
if [ $# -eq 0 ]; then
    echo "Usage: $0 <scene_id>"
    echo "Example: $0 0c09dbb153a44de0826179d04883a1fd"
    exit 1
fi

SCENE_ID="$1"
REMOTE_HOST="mfml1"
REMOTE_BASE="/music-shared-disk/group/ct/yiwen/data/objaverse/rendered_scenes_dense"
REMOTE_PATH="$REMOTE_BASE/$SCENE_ID"
LOCAL_PATH="data_samples/scene_light_combined"

############################
# Create local directory
############################
mkdir -p "$LOCAL_PATH"

############################
# Sync using rsync
############################
echo "Syncing scene $SCENE_ID from mfml1..."
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
echo "Files saved to: $LOCAL_PATH/$SCENE_ID/"
