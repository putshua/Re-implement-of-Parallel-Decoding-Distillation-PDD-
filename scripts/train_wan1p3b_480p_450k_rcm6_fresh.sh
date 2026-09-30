#!/usr/bin/env bash
# Compatibility alias; the original launcher is the fresh RCM6 entry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
