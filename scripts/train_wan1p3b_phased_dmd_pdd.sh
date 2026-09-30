#!/usr/bin/env bash
# Explicit name for the existing joint-training entry point.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT=phased_dmd_pdd
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
