#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/shared_env.sh"
export PYTHONPATH="$ROOT/AnyFlow${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
CONFIG="${CONFIG:-$ROOT/configs/wan13b_480p.json}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if [[ "$NNODES" == 1 ]]; then
  RENDEZVOUS=(--standalone)
else
  if [[ -z "${MASTER_ADDR:-}" && -z "${RDZV_ENDPOINT:-}" && "${DRY_RUN:-0}" != 1 ]]; then
    echo 'Multi-node training requires shared MASTER_ADDR or RDZV_ENDPOINT.' >&2
    exit 2
  fi
  RENDEZVOUS=(--rdzv-backend c10d
    --rdzv-endpoint "${RDZV_ENDPOINT:-${MASTER_ADDR:-localhost}:${MASTER_PORT:-29571}}"
    --rdzv-id "${RDZV_ID:-pdd-wan1p3b-480p-450k}"
    --rdzv-conf "join_timeout=${RDZV_TIMEOUT:-900}")
fi
ARGS=(--config "$CONFIG")
[[ -z "${BATCH_SIZE:-}" ]] || ARGS+=(--batch-size "$BATCH_SIZE")
[[ -z "${GRAD_ACCUM:-}" ]] || ARGS+=(--grad-accum "$GRAD_ACCUM")
[[ -z "${STEPS:-}" ]] || ARGS+=(--steps "$STEPS")
[[ -z "${OUTPUT:-}" ]] || ARGS+=(--output "$OUTPUT")
[[ -z "${RESUME:-}" ]] || ARGS+=(--resume "$RESUME")
for KEY in CHECKPOINT WEIGHT_INDEX EMBEDDING_DIR PROMPT_EMBEDDINGS NEGATIVE_EMBEDDINGS STUDENT_INIT; do
  VALUE="${!KEY:-}"
  [[ -z "$VALUE" ]] || ARGS+=(--set "${KEY,,}=$VALUE")
done
if [[ -n "${FSDP_SHARD_SIZE:-}" ]]; then
  ARGS+=(--set distributed_strategy=fsdp --set "fsdp_shard_size=$FSDP_SHARD_SIZE")
fi
CMD=("$PYTHON_BIN" -m torch.distributed.run --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE"
 "${RENDEZVOUS[@]}"
 -m pdd.train "${ARGS[@]}" "$@")
if [[ "${DRY_RUN:-0}" != 1 && "${PDD_LOGGING_ACTIVE:-0}" != 1 ]]; then
  # Direct low-level launches may obtain the output directory from their JSON.
  PDD_LOG_OUTPUT="${OUTPUT:-}"
  if [[ -z "$PDD_LOG_OUTPUT" ]]; then
    PDD_LOG_OUTPUT="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$CONFIG")"
  fi
  PDD_EXPECT_OUTPUT=0
  for PDD_ARGUMENT in "$@"; do
    if [[ "$PDD_EXPECT_OUTPUT" == 1 ]]; then PDD_LOG_OUTPUT="$PDD_ARGUMENT"; PDD_EXPECT_OUTPUT=0; fi
    [[ "$PDD_ARGUMENT" != --output ]] || PDD_EXPECT_OUTPUT=1
    case "$PDD_ARGUMENT" in --output=*) PDD_LOG_OUTPUT="${PDD_ARGUMENT#--output=}" ;; esac
  done
  OUTPUT="$PDD_LOG_OUTPUT" source "$ROOT/scripts/train_logging.sh"
fi
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/check_runtime.py"
  exec "${CMD[@]}"
fi
