#!/usr/bin/env bash
# Run the same command on every node with a shared rendezvous and configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export PYTHON_BIN
: "${CONFIG:?Set CONFIG to your training JSON}"
: "${NNODES:?Set NNODES to the number of participating nodes}"
: "${RDZV_ENDPOINT:?Set RDZV_ENDPOINT to reachable master IP:port}"
: "${RDZV_ID:?Set RDZV_ID to a unique job identifier shared by all nodes}"
if (( NNODES < 2 )); then
  echo 'Multi-node training requires NNODES >= 2.' >&2
  exit 2
fi
export NNODES RDZV_ENDPOINT RDZV_ID
exec bash "$ROOT/scripts/train_pdd.sh" "$@"
