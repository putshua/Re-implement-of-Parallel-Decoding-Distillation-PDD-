#!/usr/bin/env bash
# Standard final-endpoint DMD initialized from the completed pure-PDD step-100 model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
MASTER_ADDR=<node0> NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=16 \
  bash scripts/train_wan1p3b_480p_450k_dmd_endpoint_from_pdd.sh
Standard final-endpoint DMD only, initialized weights-only from pure-PDD step100.
Student LR=2e-5, fake LR=1e-5; PDD trajectory weight=0; final x0 endpoint only.
BS=1 per rank, GA=4 on16 GPUs gives global batch64. Set GRAD_ACCUM=8 for128.
Fake score uses one head and five fake updates per generator update.
Uses normalized x0-DMD and unweighted fake flow matching, score times [0.02,0.98].
No optimizer resume is used: student_init imports only model weights.
HELP
  exit 0
fi
export PDD_TRAIN_CONFIG="$ROOT/configs/wan1p3b_480p_450k_dmd_endpoint_from_pdd.json"
export NNODES="${NNODES:-2}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-16}" BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export MAX_ITER="${MAX_ITER:-250}"
export OUTPUT="${OUTPUT_DIR:-${OUTPUT:-$ROOT/outputs/wan1p3b_480p_450k_dmd_endpoint_from_pdd_step100_gbs64_fsdp16}}"
export RESUME_PATH="none"
export CHECKPOINT="${CHECKPOINT:-/mnt/data/butong/Wan2.2/checkpoints/Wan2.1-T2V-1.3B}"
export RDZV_ID="${RDZV_ID:-wan1p3b-dmd-endpoint-from-pdd-step100-gbs64}"
export RDZV_TIMEOUT="${RDZV_TIMEOUT:-3600}"
export LR="${LR:-2e-5}" SAVE_EVERY="${SAVE_EVERY:-5}"
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
