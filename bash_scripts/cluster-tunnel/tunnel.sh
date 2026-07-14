#!/bin/bash
# Explorer cluster SSH tunnel helper.
#
# Usage:
#   bash bash_scripts/cluster-tunnel/tunnel.sh setup       # one-time remote prep
#   bash bash_scripts/cluster-tunnel/tunnel.sh sync        # push scripts to explorer
#   bash bash_scripts/cluster-tunnel/tunnel.sh submit        # sbatch a background tunnel job
#   bash bash_scripts/cluster-tunnel/tunnel.sh interactive # srun + start sshd (foreground)
#   bash bash_scripts/cluster-tunnel/tunnel.sh status        # show job + port probe
#   bash bash_scripts/cluster-tunnel/tunnel.sh connect       # wait for tunnel, then ssh
#   bash bash_scripts/cluster-tunnel/tunnel.sh stop          # cancel running tunnel jobs
#
# Requires ~/.ssh/config hosts `explorer` and `explorer_tunnel`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

LOCAL_IDENTITY_FILE="${LOCAL_IDENTITY_FILE/#\$HOME/$HOME}"

usage() {
    sed -n '3,12p' "$0"
}

sync_scripts() {
    ssh "$SSH_LOGIN_HOST" "mkdir -p \"\$HOME/${REMOTE_TUNNEL_DIR}\""
    rsync -av \
        "$SCRIPT_DIR/config.env" \
        "$SCRIPT_DIR/start_sshd.sh" \
        "$SCRIPT_DIR/sbatch_tunnel.sh" \
        "${SSH_LOGIN_HOST}:${REMOTE_TUNNEL_DIR}/"
    ssh "$SSH_LOGIN_HOST" "chmod +x \"\$HOME/${REMOTE_TUNNEL_DIR}/start_sshd.sh\" \"\$HOME/${REMOTE_TUNNEL_DIR}/sbatch_tunnel.sh\""
    echo "Synced tunnel scripts to ${SSH_LOGIN_HOST}:~/${REMOTE_TUNNEL_DIR}"
}

ensure_remote_host_key() {
    ssh "$SSH_LOGIN_HOST" "bash -s" <<EOF
set -euo pipefail
key="\${TUNNEL_HOST_KEY:-\$HOME/.ssh/tunnel_host_key}"
key="\${key/#\\\$HOME/\$HOME}"
if [[ ! -f "\$key" ]]; then
    mkdir -p "\$(dirname "\$key")"
    ssh-keygen -t ed25519 -f "\$key" -N ""
    echo "Created \$key"
else
    echo "Host key already exists: \$key"
fi
EOF
}

ensure_local_pubkey_on_remote() {
    if [[ ! -f "$LOCAL_IDENTITY_FILE" ]]; then
        echo "ERROR: local public key not found: $LOCAL_IDENTITY_FILE" >&2
        exit 1
    fi
    local pubkey
    pubkey="$(cat "$LOCAL_IDENTITY_FILE")"
    ssh "$SSH_LOGIN_HOST" "bash -s" <<EOF
set -euo pipefail
mkdir -p "\$HOME/.ssh"
chmod 700 "\$HOME/.ssh"
touch "\$HOME/.ssh/authorized_keys"
chmod 600 "\$HOME/.ssh/authorized_keys"
if ! grep -qF '$pubkey' "\$HOME/.ssh/authorized_keys"; then
    printf '%s\n' '$pubkey' >> "\$HOME/.ssh/authorized_keys"
    echo "Added local public key to authorized_keys"
else
    echo "Local public key already in authorized_keys"
fi
EOF
}

remote_squeue_node() {
    ssh "$SSH_LOGIN_HOST" \
        "squeue --me --name=${TUNNEL_JOB_NAME} --states=R -h -O NodeList | head -1 | tr -d '[:space:]'"
}

probe_tunnel_port() {
    local node="$1"
    if [[ -z "$node" ]]; then
        return 1
    fi
    ssh "$SSH_LOGIN_HOST" "nc -z -w 3 '$node' '${TUNNEL_PORT}'" >/dev/null 2>&1
}

cmd_setup() {
    ensure_local_pubkey_on_remote
    ensure_remote_host_key
    sync_scripts
    echo
    echo "Setup complete. Start a tunnel with:"
    echo "  bash $SCRIPT_DIR/tunnel.sh submit"
    echo "  bash $SCRIPT_DIR/tunnel.sh connect"
}

cmd_sync() {
    sync_scripts
}

cmd_submit() {
    sync_scripts
    ssh "$SSH_LOGIN_HOST" "bash -s" <<EOF
set -euo pipefail
cd "\$HOME/${REMOTE_TUNNEL_DIR}"
job_id=\$(sbatch --job-name='${TUNNEL_JOB_NAME}' \
    --partition='${SBATCH_PARTITION}' \
    --nodes='${SBATCH_NODES}' \
    --ntasks='${SBATCH_NTASKS}' \
    --gres='${SBATCH_GRES}' \
    --mem='${SBATCH_MEM}' \
    --time='${SBATCH_TIME}' \
    --output='tunnel-%j.out' \
    --error='tunnel-%j.err' \
    --wrap="\$HOME/${REMOTE_TUNNEL_DIR}/start_sshd.sh")
echo "\$job_id"
EOF
    echo
    echo "Tunnel job submitted. Check status with:"
    echo "  bash $SCRIPT_DIR/tunnel.sh status"
    echo "Then connect with:"
    echo "  bash $SCRIPT_DIR/tunnel.sh connect"
}

cmd_interactive() {
    sync_scripts
    echo "Starting interactive tunnel on ${SSH_LOGIN_HOST}."
    echo "Keep this terminal open. Connect from another local terminal with:"
    echo "  bash $SCRIPT_DIR/tunnel.sh connect"
    echo
    ssh -t "$SSH_LOGIN_HOST" "bash -s" <<EOF
set -euo pipefail
cd "\$HOME/${REMOTE_TUNNEL_DIR}"
exec srun --job-name='${TUNNEL_JOB_NAME}' \
    --partition='${SBATCH_PARTITION}' \
    --nodes='${SBATCH_NODES}' \
    --ntasks='${SBATCH_NTASKS}' \
    --gres='${SBATCH_GRES}' \
    --mem='${SBATCH_MEM}' \
    --time='${SBATCH_TIME}' \
    --pty \
    "\$HOME/${REMOTE_TUNNEL_DIR}/start_sshd.sh"
EOF
}

cmd_status() {
    echo "=== SLURM jobs named ${TUNNEL_JOB_NAME} ==="
    ssh "$SSH_LOGIN_HOST" "squeue --me --name=${TUNNEL_JOB_NAME} -O JobID:10,Partition:12,Name:10,State:8,TimeUsed:10,NodeList:16" || true
    echo
    local node
    node="$(remote_squeue_node || true)"
    if [[ -z "$node" ]]; then
        echo "No running tunnel job found."
        return 1
    fi
    echo "Running node: $node"
    if probe_tunnel_port "$node"; then
        echo "Port ${TUNNEL_PORT} on $node is reachable from login node."
        echo "Ready: ssh ${SSH_TUNNEL_HOST}"
    else
        echo "Port ${TUNNEL_PORT} on $node is not reachable yet (sshd may still be starting)."
        return 1
    fi
}

cmd_connect() {
    local elapsed=0
    local node=""
    echo "Waiting for tunnel job '${TUNNEL_JOB_NAME}' (up to ${CONNECT_WAIT_SECS}s)..."
    while (( elapsed < CONNECT_WAIT_SECS )); do
        node="$(remote_squeue_node || true)"
        if [[ -n "$node" ]] && probe_tunnel_port "$node"; then
            echo "Tunnel ready on ${node}:${TUNNEL_PORT}"
            exec ssh "$SSH_TUNNEL_HOST"
        fi
        sleep "$CONNECT_POLL_SECS"
        elapsed=$((elapsed + CONNECT_POLL_SECS))
    done
    echo "ERROR: tunnel not ready after ${CONNECT_WAIT_SECS}s." >&2
    echo "Check: bash $SCRIPT_DIR/tunnel.sh status" >&2
    exit 1
}

cmd_stop() {
    ssh "$SSH_LOGIN_HOST" "scancel --name=${TUNNEL_JOB_NAME} --user=\$USER 2>/dev/null || true"
    echo "Cancelled tunnel jobs named ${TUNNEL_JOB_NAME} (if any)."
}

main() {
    local cmd="${1:-}"
    case "$cmd" in
        setup) cmd_setup ;;
        sync) cmd_sync ;;
        submit) cmd_submit ;;
        interactive|interactive_tunnel|srun) cmd_interactive ;;
        status) cmd_status ;;
        connect) cmd_connect ;;
        stop) cmd_stop ;;
        -h|--help|help|"")
            usage
            ;;
        *)
            echo "Unknown command: $cmd" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
