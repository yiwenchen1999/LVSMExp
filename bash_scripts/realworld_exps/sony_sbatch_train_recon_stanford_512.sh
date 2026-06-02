#!/usr/bin/env bash
#SBATCH --job-name=recon_stanford_512
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=/group2/ct/yiwen/logs/%x.%N.%j.out
#SBATCH --error=/group2/ct/yiwen/logs/%x.%N.%j.err

set -euo pipefail

export PROJ="${PROJ:-/music-shared-disk/group/ct/yiwen/codes/LVSMExp}"

# Default distributed config for sbatch jobs; can still be overridden by env.
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    export NPROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
  else
    export NPROC_PER_NODE=1
  fi
fi

# Default rendezvous port for this sbatch launcher.
export MASTER_PORT="${MASTER_PORT:-29533}"

echo "=============================================="
echo "SBATCH: Stanford ORB recon-only @ 512 (Sony)"
echo "=============================================="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-N/A}"
echo "PROJ: $PROJ"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "----------------------------------------------"

bash "$PROJ/bash_scripts/realworld_exps/sony_train_recon_stanford_512.sh"

