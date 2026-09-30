#!/usr/bin/env bash
# Run the same command on each of two 8-GPU keeper nodes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PDD_TRAIN_CONFIG="$ROOT/configs/wan1p3b_480p_450k_dmd_teacherinit_gbs32_1k.json"
export NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=16
export BATCH_SIZE=1 GRAD_ACCUM=2 MAX_ITER=1000
export RESUME_PATH="${RESUME_PATH:-auto}"
export OUTPUT_DIR="$ROOT/outputs/wan13b_dmd_teacherinit_cm4_gbs32_1000step_20260929"
export RDZV_ID="${RDZV_ID:-wan13b-dmd-teacherinit-cm4-gbs32-1k-20260929}"
if [[ ! -s "$ROOT/assets/vbench16_seed20260929/embeddings/prompts_000000000.pt" ]]; then
  echo 'VBench16 UMT5 embeddings missing; run prompt preparation first.' >&2
  exit 2
fi
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
