#!/usr/bin/env bash
set -euo pipefail

#############################################
# Data Transfer Script: NEU -> Local -> Sony Cluster
#
# This script transfers data from NEU Discovery cluster to Sony AI cluster
# via a local staging area (external drive or local directory).
#
# Usage:
#   # Use default settings
#   bash data_transfer.sh
#
#   # Override settings via environment variables
#   STAGE_ROOT="/Volumes/TOSHIBA EXT" \
#   NEU_SRC="/path/to/source" \
#   SONY_DST="/music-shared-disk/group/ct/yiwen/data/target" \
#   bash data_transfer.sh
#
# Features:
#   - Automatic retry on network failures (up to 20 attempts)
#   - Resume interrupted transfers (--partial and --inplace)
#   - SSH connection keepalive to prevent timeouts
#   - Progress logging to files
#   - Space checking before transfer
#
# Storage locations on Sony cluster:
#   - Team space: /music-shared-disk/group/TEAM/username/data/ (recommended for datasets)
#   - User space: /scratch2/USERNAME/ (for temporary files)
#   - Home: /home/USERNAME/ (limited to 100GB, slow)
#
#############################################
# User-configurable settings
#############################################

# NEU source (remote)
# Format: username@host or use SSH config alias
NEU_HOST="${NEU_HOST:-chen.yiwe@xfer.discovery.neu.edu}"
NEU_SRC="${NEU_SRC:-/projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_format}"

# Local staging (external drive or local directory)
# Set to external drive path like "/Volumes/TOSHIBA EXT" or local directory like "./local_buffer"
STAGE_ROOT="${STAGE_ROOT:-/Volumes/TOSHIBA/FileTransferNode}"
STAGE_DIR="${STAGE_ROOT}/lvsm_format"

# Sony destination (remote)
# Use mfml1 (recommended) or mfmsc as login node
# Storage options:
#   - Team space: /music-shared-disk/group/TEAM/username/data/...
#   - User space: /scratch2/USERNAME/...
#   - Home: /home/USERNAME/... (limited to 100GB, slow)
SONY_HOST="${SONY_HOST:-mfml1}"
SONY_DST="${SONY_DST:-/music-shared-disk/group/ct/yiwen/data/objaverse/lvsm_format}"

# Capacity guardrail (GB). You said disk is 800GB total.
# We'll require at least this much FREE space to proceed.
MIN_FREE_GB=10

# Rsync behavior:
# - By default we DO NOT delete at destination; set to 1 if you want exact mirroring.
DELETE_ON_SYNC=0

# Optional: limit bandwidth (MiB/s) to keep your connection usable. 0 = unlimited.
BW_LIMIT_MIB=0

#############################################
# Helpers
#############################################

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

log() { echo "[$(timestamp)] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1"
    exit 1
  }
}

# Free space in GB on the staging disk
free_gb() {
  # df -k gives KB blocks; convert to GB
  df -k "$STAGE_ROOT" | awk 'NR==2 {printf "%.0f\n", $4/1024/1024}'
}

# Remote directory size in GB (best-effort; may take time)
remote_size_gb() {
  local host="$1"
  local path="$2"
  # du -sk gives KB; convert to GB
  ssh "$host" "du -sk \"$path\" 2>/dev/null | awk '{printf \"%.0f\\n\", \$1/1024/1024}'" || echo "0"
}

# Rsync with retry mechanism for network issues
rsync_with_retry() {
  local max_attempts=20
  local attempt=1
  local delay=10  # seconds between retries
  
  while [[ $attempt -le $max_attempts ]]; do
    log "Rsync attempt $attempt/$max_attempts"
    if rsync "$@"; then
      log "Rsync completed successfully"
      return 0
    else
      local exit_code=$?
      if [[ $attempt -lt $max_attempts ]]; then
        log "Rsync failed (exit code: $exit_code), retrying in ${delay}s..."
        sleep $delay
        delay=$((delay * 2))  # exponential backoff
        attempt=$((attempt + 1))
      else
        log "Rsync failed after $max_attempts attempts (exit code: $exit_code)"
        return $exit_code
      fi
    fi
  done
}

rsync_common_flags=(
  -a              # archive (preserve perms/times, recurse)
  -v              # verbose
  -h              # human-readable numbers
  --partial       # keep partially transferred files (resume)
  --inplace       # resume large files efficiently
  --progress      # show progress (compatible with older rsync versions)
)

# SSH options for stable connections (prevent timeouts)
ssh_opts=(
  -e
  "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes"
)

# Excludes (optional): uncomment if you want to skip caches/tmp artifacts
rsync_excludes=(
  # --exclude ".DS_Store"
  # --exclude "__pycache__/"
)

delete_flag=()
if [[ "$DELETE_ON_SYNC" == "1" ]]; then
  delete_flag+=(--delete --delete-delay)
fi

bw_flag=()
if [[ "$BW_LIMIT_MIB" != "0" ]]; then
  bw_flag+=(--bwlimit="$((BW_LIMIT_MIB*1024))")  # rsync expects KiB/s
fi

#############################################
# Main
#############################################

require_cmd ssh
require_cmd rsync
require_cmd df
require_cmd awk

log "Step 0: Sanity checks"
# Create staging directory if it doesn't exist (for local directories)
if [[ ! -d "$STAGE_ROOT" ]]; then
  log "Creating staging root directory: $STAGE_ROOT"
  mkdir -p "$STAGE_ROOT" || {
    echo "Failed to create staging directory: $STAGE_ROOT"
    exit 1
  }
fi

log "Checking Sony SSH connectivity..."
ssh -o BatchMode=yes -o ConnectTimeout=10 "$SONY_HOST" "echo ok >/dev/null" || {
  echo "Cannot SSH to Sony host '$SONY_HOST'. Fix SSH config / MFA first."
  echo "Try: ssh $SONY_HOST"
  exit 1
}

log "Checking NEU SSH connectivity..."
ssh -o BatchMode=yes -o ConnectTimeout=10 "$NEU_HOST" "echo ok >/dev/null" || {
  echo "Cannot SSH to NEU host '$NEU_HOST'. You may need Duo/MFA approval."
  echo "Try: ssh $NEU_HOST"
  exit 1
}

log "Step 1: Estimate NEU source size (may take a bit)..."
NEU_SIZE_GB="$(remote_size_gb "$NEU_HOST" "$NEU_SRC")"
log "NEU source size (approx): ${NEU_SIZE_GB} GB"

FREE_GB="$(free_gb)"
log "External drive free space: ${FREE_GB} GB (min required free: ${MIN_FREE_GB} GB)"

# Guardrail: require at least MIN_FREE_GB free AND ideally enough for the dataset.
# If NEU_SIZE_GB is 0 (du failed), we only enforce MIN_FREE_GB.
if [[ "$NEU_SIZE_GB" != "0" ]]; then
  if (( FREE_GB < NEU_SIZE_GB + MIN_FREE_GB )); then
    log "WARNING: Not enough free space on staging drive."
    log "Need approx: $((NEU_SIZE_GB + MIN_FREE_GB)) GB free, but have: ${FREE_GB} GB."
    log "Proceeding anyway (rsync will fail if truly out of space)..."
    # Don't exit, let rsync handle it
  else
    log "✓ Sufficient free space available"
  fi
else
  if (( FREE_GB < MIN_FREE_GB )); then
    log "WARNING: Low free space on staging drive (free=${FREE_GB} GB, min=${MIN_FREE_GB} GB)"
    log "Proceeding anyway..."
  else
    log "✓ Minimum free space available"
  fi
fi

log "Creating staging directory: $STAGE_DIR"
mkdir -p "$STAGE_DIR"

LOG_DIR="${STAGE_ROOT}/neu_buffer/_logs"
mkdir -p "$LOG_DIR"

LOG1="${LOG_DIR}/rsync_neu_to_local_$(timestamp).log"
LOG2="${LOG_DIR}/rsync_local_to_sony_$(timestamp).log"

log "Step 2: Sync NEU -> local staging"
log "Source: ${NEU_HOST}:${NEU_SRC}"
log "Destination: ${STAGE_DIR}"
log "Logging to: $LOG1"
rsync_args=("${rsync_common_flags[@]}" "${ssh_opts[@]}")
[[ ${#rsync_excludes[@]} -gt 0 ]] && rsync_args+=("${rsync_excludes[@]}")
[[ ${#bw_flag[@]} -gt 0 ]] && rsync_args+=("${bw_flag[@]}")
rsync_with_retry "${rsync_args[@]}" \
  "${NEU_HOST}:${NEU_SRC}/" \
  "${STAGE_DIR}/" | tee "$LOG1"

log "Step 3: Sync local staging -> Sony cluster"
log "Source: ${STAGE_DIR}"
log "Destination: ${SONY_HOST}:${SONY_DST}"
log "Logging to: $LOG2"
rsync_args2=("${rsync_common_flags[@]}" "${ssh_opts[@]}")
[[ ${#rsync_excludes[@]} -gt 0 ]] && rsync_args2+=("${rsync_excludes[@]}")
[[ ${#bw_flag[@]} -gt 0 ]] && rsync_args2+=("${bw_flag[@]}")
[[ ${#delete_flag[@]} -gt 0 ]] && rsync_args2+=("${delete_flag[@]}")
rsync_with_retry "${rsync_args2[@]}" \
  "${STAGE_DIR}/" \
  "${SONY_HOST}:${SONY_DST}/" | tee "$LOG2"

log "Step 4: Quick verification (counts + sample listing)"
log "Local file count:"
find "$STAGE_DIR" -type f | wc -l

log "Sony file count (may take time on huge dirs):"
ssh "$SONY_HOST" "find \"$SONY_DST\" -type f | wc -l" || true

log "DONE ✅"
log "If interrupted, re-run the script; rsync will resume."
