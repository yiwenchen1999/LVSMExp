#!/usr/bin/env bash
# Reorganize infer_stanfordORB_512_editor previews into flat/ and single_image/.
# supervision_* strips are split as input | gt | pred per target view.
#
# Usage:
#   bash bash_scripts/realworld_exps/reorganize_infer_stanfordORB_previews.sh
#   bash bash_scripts/realworld_exps/reorganize_infer_stanfordORB_previews.sh --dry-run
#   bash bash_scripts/realworld_exps/reorganize_infer_stanfordORB_previews.sh result_previews/infer_stanfordORB_512_editor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate
cd "${REPO_ROOT}"

exec python bash_scripts/realworld_exps/reorganize_infer_stanfordORB_previews.py "$@"
