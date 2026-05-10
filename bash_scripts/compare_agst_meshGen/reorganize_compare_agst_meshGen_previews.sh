#!/usr/bin/env bash
# Flatten + single_image for runs under result_previews/compare_agst_meshGen.
# Reuses bash_scripts/realworld_exps/reorganize_realworld_eval_previews.py (variable #input views).
#
# Usage:
#   bash bash_scripts/compare_agst_meshGen/reorganize_compare_agst_meshGen_previews.sh
#   RUN_NAMES="test_relight_2view_objaverse" bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_DIR="${BASE_DIR:-${REPO_ROOT}/result_previews/compare_agst_meshGen}"
RUN_NAMES="${RUN_NAMES:-test_relight_1view_objaverse test_relight_1view_polyhaven test_relight_2view test_relight_2view_objaverse}"

# shellcheck disable=SC2086
python3 "${REPO_ROOT}/bash_scripts/realworld_exps/reorganize_realworld_eval_previews.py" \
  --base "${BASE_DIR}" \
  --infer ${RUN_NAMES} \
  "$@"
