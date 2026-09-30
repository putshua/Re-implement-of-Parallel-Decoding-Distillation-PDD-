#!/usr/bin/env bash
# Same command on both 8-GPU keeper nodes; 1000 optimizer updates = 800 fake + 200 G.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PDD_TRAIN_CONFIG="$ROOT/configs/wan1p3b_480p_dmd_teacherinit_cycle1to4_f4g1_1000updates.json"
export NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=16
export BATCH_SIZE=1 GRAD_ACCUM=2 MAX_ITER=1000
export SAVE_EVERY=25
export FIXED_PROMPT_NFE='[4]'
export RESUME_PATH="${RESUME_PATH:-auto}"
export OUTPUT_DIR="$ROOT/outputs/wan13b_dmd_teacherinit_cycle1to4_f4g1_gbs32_1000updates_20260930"
export RDZV_ID="${RDZV_ID:-wan13b-dmd-cycle1to4-f4g1-1000updates-20260930}"
if [[ ! -s "$ROOT/assets/vbench16_seed20260929/embeddings/prompts_000000000.pt" ]]; then
  echo 'VBench16 UMT5 embeddings missing.' >&2
  exit 2
fi
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
