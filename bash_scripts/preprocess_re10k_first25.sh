#!/usr/bin/env bash
# Preprocess only RealEstate10K scenes listed in data/re10k_c5_64_first25.json (same pipeline as README / process_data.py).
#
# Raw layout expected: <RE10K_RAW_PARENT>/test/*.torch  (i.e. README's YOUR_RAW_DATAPATH with mode "test")
#
# Override any variable by exporting it before running, e.g.:
#   export OUTPUT_DIR=/scratch/me/re10k_first25
#   bash bash_scripts/preprocess_re10k_first25.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RE10K_RAW_PARENT="${RE10K_RAW_PARENT:-/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/scenetokExps/dataset/re10k}"
SCENE_KEYS_JSON="${SCENE_KEYS_JSON:-${REPO_ROOT}/data/re10k_c5_64_first25.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/preprocessed_data/re10k_c5_64_first25}"
NUM_PROCESSES="${NUM_PROCESSES:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"

if [[ ! -f "${SCENE_KEYS_JSON}" ]]; then
  echo "Scene key JSON not found: ${SCENE_KEYS_JSON}" >&2
  exit 1
fi

if [[ ! -d "${RE10K_RAW_PARENT}/test" ]]; then
  echo "Expected raw test shards at: ${RE10K_RAW_PARENT}/test (*.torch)" >&2
  exit 1
fi

echo "Repo:           ${REPO_ROOT}"
echo "Raw re10k root: ${RE10K_RAW_PARENT} (mode=test)"
echo "Scene list:     ${SCENE_KEYS_JSON}"
echo "Output:         ${OUTPUT_DIR}"

python "${REPO_ROOT}/preprocess_scripts/process_data.py" \
  --base_path "${RE10K_RAW_PARENT}" \
  --output_dir "${OUTPUT_DIR}" \
  --mode test \
  --scene_keys_json "${SCENE_KEYS_JSON}" \
  --num_processes "${NUM_PROCESSES}" \
  --chunk_size "${CHUNK_SIZE}"

echo "Done. Dataset list: ${OUTPUT_DIR}/test/full_list.txt"
