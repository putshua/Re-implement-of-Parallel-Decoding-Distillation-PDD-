#!/usr/bin/env bash
# Backward-compatible filename; inherits the main entry defaults (currently 4 x 8 GPUs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
