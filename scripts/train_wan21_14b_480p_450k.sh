#!/usr/bin/env bash
# Run the same command on every node, as with the existing H3-style entry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 bash scripts/train_wan21_14b_480p_450k.sh
Set the same MASTER_ADDR/RDZV_ENDPOINT on every node; no NODE_RANK required.
Defaults: Wan2.1 T2V 14B, 832x480/81f, 450613 prompts, batch1, accum64,
          effective batch1024 on16 GPUs; student AND frozen teacher FSDP;
          CPU retained trajectories; 250 updates; checkpoint/RCM6 preview every25.
Override GRAD_ACCUM, BATCH_SIZE, MAX_ITER, OUTPUT_DIR, RESUME_PATH as usual.
GRAD_ACCUM=128 gives batch2048 on16 GPUs, with roughly twice the update time.
Shared Python is selected by scripts/shared_env.sh. DRY_RUN=1 prints the command.
HELP
  exit 0
fi
export PDD_TRAIN_CONFIG="$ROOT/configs/wan21_14b_480p_450k_16gpu.json"
export CHECKPOINT="${CHECKPOINT:-/mnt/data/butong/Wan2.2/checkpoints/Wan2.1-T2V-14B}"
export NNODES="${NNODES:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}" GRAD_ACCUM="${GRAD_ACCUM:-64}"
export OUTPUT="${OUTPUT_DIR:-${OUTPUT:-$ROOT/outputs/wan21_14b_480p_450k_rcm6_fsdp16}}"
export RDZV_ID="${RDZV_ID:-pdd-wan21-14b-480p-450k}"
export RDZV_TIMEOUT="${RDZV_TIMEOUT:-3600}"
export SAVE_EVERY="${SAVE_EVERY:-25}"
export LR="${LR:-1e-5}"
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
