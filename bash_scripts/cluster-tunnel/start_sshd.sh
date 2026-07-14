#!/bin/bash
# Run on an allocated Explorer compute node (inside the SLURM job).
# Starts a user-space sshd on TUNNEL_PORT so local `ssh explorer_tunnel` works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

TUNNEL_PORT="${TUNNEL_PORT:-2222}"
TUNNEL_HOST_KEY="${TUNNEL_HOST_KEY:-$HOME/.ssh/tunnel_host_key}"
TUNNEL_HOST_KEY="${TUNNEL_HOST_KEY/#\$HOME/$HOME}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: SLURM_JOB_ID is not set. Run this inside an srun/sbatch allocation." >&2
    exit 1
fi

if ! command -v sshd >/dev/null 2>&1; then
    echo "ERROR: sshd not found on PATH." >&2
    exit 1
fi

if [[ ! -f "$TUNNEL_HOST_KEY" ]]; then
    echo "Generating tunnel host key: $TUNNEL_HOST_KEY"
    mkdir -p "$(dirname "$TUNNEL_HOST_KEY")"
    ssh-keygen -t ed25519 -f "$TUNNEL_HOST_KEY" -N ""
fi

RUN_DIR="/tmp/sshd_${USER}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

NODE="$(hostname -s)"
echo "Tunnel sshd starting on ${NODE}:${TUNNEL_PORT} (job ${SLURM_JOB_ID}, name ${SLURM_JOB_NAME:-unknown})"
echo "From your laptop: ssh ${SSH_TUNNEL_HOST:-explorer_tunnel}"

exec /usr/sbin/sshd -D -p "$TUNNEL_PORT" \
    -o "HostKey=${TUNNEL_HOST_KEY}" \
    -o "AuthorizedKeysFile=${HOME}/.ssh/authorized_keys" \
    -o "PidFile=${RUN_DIR}/sshd.pid" \
    -o "StrictModes=no" \
    -o "PasswordAuthentication=no" \
    -o "UsePAM=no" \
    -o "X11Forwarding=yes"
