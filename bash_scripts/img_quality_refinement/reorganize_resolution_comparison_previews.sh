#!/usr/bin/env bash
# Wrapper for reorganize_resolution_comparison_previews.py (flattened grids + single_image tiles).
# Pass-through args: --infer infer_256, --dry-run, --base DIR, etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/reorganize_resolution_comparison_previews.py" "$@" 


#bash_scripts/img_quality_refinement/reorganize_resolution_comparison_previews.sh
#bash bash_scripts/img_quality_refinement/reorganize_resolution_comparison_previews.sh --infer infer_256
# bash bash_scripts/img_quality_refinement/reorganize_resolution_comparison_previews.sh \
#   --base /path/to/parent \
#   --infer my_infer_folder

# bash bash_scripts/img_quality_refinement/compute_single_image_relit_metrics.sh --skip-lpips
# bash bash_scripts/img_quality_refinement/compute_single_image_relit_metrics.sh --infer infer_256 --skip-lpips