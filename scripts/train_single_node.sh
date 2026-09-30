#!/usr/bin/env bash
# Portable entry point: use the active environment unless PYTHON_BIN is supplied.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export PYTHON_BIN NNODES=1
: "${CONFIG:?Set CONFIG to your training JSON}"
exec bash "$ROOT/scripts/train_pdd.sh" "$@"
