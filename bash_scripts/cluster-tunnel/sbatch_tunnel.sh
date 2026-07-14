#!/bin/bash
#SBATCH --job-name=tunnel
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=tunnel-%j.out
#SBATCH --error=tunnel-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_sshd.sh"
