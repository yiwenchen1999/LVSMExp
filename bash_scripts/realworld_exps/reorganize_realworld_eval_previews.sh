#!/usr/bin/env bash
# Reorganize pulled realworld relight previews into:
#   1) <run_name>_flattened
#   2) single_image/<run_name>/...
#
# Usage:
#   bash bash_scripts/realworld_exps/reorganize_realworld_eval_previews.sh
#   RUN_NAME=test_relight_obj-with-light \
#     bash bash_scripts/realworld_exps/reorganize_realworld_eval_previews.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_DIR="${BASE_DIR:-${REPO_ROOT}/result_previews/realworld_eval}"
RUN_NAME="${RUN_NAME:-test_relight_stanfordORB}"

python3 "${REPO_ROOT}/bash_scripts/realworld_exps/reorganize_realworld_eval_previews.py" \
  --base "${BASE_DIR}" \
  --infer "${RUN_NAME}" \
  "$@"
