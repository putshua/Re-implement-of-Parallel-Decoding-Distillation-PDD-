#!/usr/bin/env bash
# Run once per node with the same MASTER_ADDR / RDZV_ENDPOINT.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
MASTER_ADDR=<node0> NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=16 \
  bash scripts/train_wan21_14b_480p_phased_dmd_pdd.sh
Wan2.1 14B, 832x480/81f, 450k prompts, original teacher initialization.
PDD:DMD=1:1, fake:G=5:1; BS=1, GA auto-selects global batch128 (16 GPUs: GA8).
GRAD_ACCUM=16 gives global batch256 on16 GPUs. No automatic resume from 1.3B.
Student/teacher/fake use FSDP; activation and both Adam states use CPU offload.
Rolling checkpoint every5, permanent every50, 32 high-motion prompt previews every25.
Use DRY_RUN=1 to inspect. Local tests must explicitly set FSDP_SHARD_SIZE.
HELP
  exit 0
fi
export NNODES="${NNODES:-2}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-16}" BATCH_SIZE="${BATCH_SIZE:-1}"
if ! [[ "$NNODES" =~ ^[1-9][0-9]*$ && "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ && "$FSDP_SHARD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NNODES, NPROC_PER_NODE and FSDP_SHARD_SIZE must be positive integers.' >&2; exit 2
fi
if (( NNODES * NPROC_PER_NODE < FSDP_SHARD_SIZE )); then
  echo 'Requested FSDP shard exceeds total GPUs; explicitly choose a smaller shard for local tests.' >&2; exit 2
fi
export PDD_TRAIN_CONFIG="$ROOT/configs/wan21_14b_480p_450k_phased_dmd_pdd.json"
export CHECKPOINT="${CHECKPOINT:-/mnt/data/butong/Wan2.2/checkpoints/Wan2.1-T2V-14B}"
export OUTPUT="${OUTPUT_DIR:-${OUTPUT:-$ROOT/outputs/wan21_14b_480p_450k_phased_dmd_pdd_fsdp16_teacherinit_w1_fake5}}"
export RDZV_ID="${RDZV_ID:-wan21-14b-phased-dmd-pdd-fsdp16}"
export LR="${LR:-2e-5}" SAVE_EVERY="${SAVE_EVERY:-5}"
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
