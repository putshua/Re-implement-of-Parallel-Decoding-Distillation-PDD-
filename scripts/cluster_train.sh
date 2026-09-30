#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-$ROOT/configs/wan13b_480p.json}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" NNODES="${NNODES:-1}"
# Target global batch 256. Batch 4 was tested on A800 80GB; probe the actual allocation.
export BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_MICROBATCH=$((NNODES * NPROC_PER_NODE * BATCH_SIZE))
export GRAD_ACCUM="${GRAD_ACCUM:-$(((256 + TOTAL_MICROBATCH - 1) / TOTAL_MICROBATCH))}"
exec bash "$ROOT/scripts/train_pdd.sh" "$@"
