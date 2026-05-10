#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 "${SCRIPT_DIR}/make_realworld_eval_qualitative_figure.py" \
  --base "${REPO_ROOT}/result_previews/realworld_eval/single_image" \
  --list "${REPO_ROOT}/result_previews/realworld_eval/demo_scene.txt" \
  --output "${REPO_ROOT}/result_previews/realworld_eval/qualitative_8scene_4row.jpg" \
  "$@"
