#!/usr/bin/env bash
# Full global-batch128 feasibility check: joint save/resume + matched control.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/shared_env.sh"
BASE="${SMOKE_OUTPUT_ROOT:-$ROOT/outputs/joint_smoke_$(date +%Y%m%d_%H%M%S)}"
export NNODES=1 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=4 BATCH_SIZE=1 GRAD_ACCUM=32
export SAVE_EVERY=1 FIXED_PROMPT_AT_START=0 FIXED_PROMPT_AT_END=1 FIXED_PROMPT_COUNT=1
export NCCL_DEBUG=WARN
unset RESUME_PATH PDD_TRAIN_CONFIG
EXPERIMENT=phased_dmd_pdd MAX_ITER=1 RESUME=none OUTPUT_DIR="$BASE/joint" \
  bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh"
EXPERIMENT=phased_dmd_pdd MAX_ITER=2 RESUME=auto OUTPUT_DIR="$BASE/joint" \
  bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh"
EXPERIMENT=pdd_matched MAX_ITER=1 RESUME=none OUTPUT_DIR="$BASE/matched" \
  bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh"
"$PYTHON_BIN" "$ROOT/scripts/verify_joint_run.py" "$BASE/joint"
"$PYTHON_BIN" "$ROOT/scripts/verify_joint_run.py" "$BASE/matched"
