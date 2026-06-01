#!/usr/bin/env bash
# Wrapper for make_progressive_trio_composite.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/make_progressive_trio_composite.py" "$@"

# Example:
# bash bash_scripts/scene/make_progressive_trio_composite.sh --overwrite
