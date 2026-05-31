#!/usr/bin/env bash
# Build a row-per-sample result table (input 2x2 grid | gt relit | pred relit)
# for the samples listed in scene_id.txt, sourcing single_image tiles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_NAME="${RUN_NAME:-infer_stanfordORB_512_editor}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/result_previews/realworld_eval/single_image/${RUN_NAME}}"
LIST_FILE="${LIST_FILE:-${REPO_ROOT}/result_previews/realworld_eval/${RUN_NAME}_flattened/scene_id.txt}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/result_previews/realworld_eval/result_table_${RUN_NAME}.jpg}"

python3 "${SCRIPT_DIR}/make_realworld_eval_result_table.py" \
  --base "${BASE_DIR}" \
  --list "${LIST_FILE}" \
  --output "${OUTPUT}" \
  "$@"
