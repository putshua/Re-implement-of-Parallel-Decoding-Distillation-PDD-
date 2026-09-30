#!/usr/bin/env bash
# H3-style launch: run the SAME command once on every allocated node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/shared_env.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
NNODES=4 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 bash scripts/train_wan1p3b_480p_450k.sh
Multi-node: export the same MASTER_ADDR or RDZV_ENDPOINT on every node.
c10d assigns node ranks automatically; NODE_RANK is not needed.
Shared Python: /mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python
Defaults: Phased DMD + PDD4, 480p/81f, 450613 weighted prompts, batch1,
          global batch128 via accumulation, 250 generator updates, rolling save every5, permanent save every50, fixed prompts every25.
EXPERIMENT=pdd_matched selects the same-rollout no-DMD control.
Both experiments initialize from original Wan1.3B; fresh optimizers.
Joint loss: PDD=1, DMD=1 from update1 (no ramp); fake:G=5:1.
Overrides: NNODES, NPROC_PER_NODE, FSDP_SHARD_SIZE, BATCH_SIZE, GRAD_ACCUM,
           MAX_ITER/STEPS, SAVE_EVERY, LR, NUM_WORKERS, OUTPUT_DIR/OUTPUT,
           RESUME_PATH/RESUME (auto, none, or checkpoint path),
           CHECKPOINT, WEIGHT_INDEX, NEGATIVE_EMBEDDINGS,
           MASTER_ADDR, MASTER_PORT, RDZV_ENDPOINT, RDZV_ID, RDZV_TIMEOUT,
           TENSORBOARD_ENABLED, TENSORBOARD_EVERY,
           FIXED_PROMPT_ENABLED, FIXED_PROMPT_EVERY (25), FIXED_PROMPT_AT_START,
           FIXED_PROMPT_AT_END, FIXED_PROMPT_COUNT (6), FIXED_PROMPT_SEED (42),
           FIXED_PROMPT_NFE ('[4,8]'), FIXED_PROMPT_DECODE, FIXED_PROMPT_EMBEDDINGS,
           WAN_ROOT, PYTHON_BIN, LOG_DIR, DRY_RUN=1; trailing --set key=value is supported.
HELP
  exit 0
fi
EXPERIMENT="${EXPERIMENT:-phased_dmd_pdd}"
case "$EXPERIMENT" in
  phased_dmd_pdd|pdd_matched) ;;
  *) echo 'EXPERIMENT must be phased_dmd_pdd or pdd_matched' >&2; exit 2 ;;
esac
export CONFIG="${PDD_TRAIN_CONFIG:-$ROOT/configs/wan1p3b_480p_450k_${EXPERIMENT}.json}"
export RDZV_ID="${RDZV_ID:-wan1p3b-${EXPERIMENT}-gbs128-teacherinit-w1-fake5}"
export RDZV_TIMEOUT="${RDZV_TIMEOUT:-3600}"
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-8}" BATCH_SIZE="${BATCH_SIZE:-1}"
for key in NNODES NPROC_PER_NODE FSDP_SHARD_SIZE BATCH_SIZE; do
  if ! [[ "${!key}" =~ ^[1-9][0-9]*$ ]]; then echo "$key must be positive integer" >&2; exit 2; fi
done
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
EFFECTIVE_FSDP=$((FSDP_SHARD_SIZE < TOTAL_GPUS ? FSDP_SHARD_SIZE : TOTAL_GPUS))
if (( TOTAL_GPUS % EFFECTIVE_FSDP )); then
  echo 'Total GPUs must be divisible by effective FSDP_SHARD_SIZE.' >&2; exit 2
fi
case "${DRY_RUN:-0}" in
  1|true|True) export DRY_RUN=1 ;; 0|false|False) export DRY_RUN=0 ;;
  *) echo 'DRY_RUN must be 0/1 or false/true.' >&2; exit 2 ;;
esac
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR:-}" && -z "${RDZV_ENDPOINT:-}" && "$DRY_RUN" != 1 ]]; then
  echo 'Set shared MASTER_ADDR or RDZV_ENDPOINT on all nodes; NODE_RANK is not required.' >&2; exit 2
fi
if [[ -z "${GRAD_ACCUM:-}" ]] && (( 128 % (TOTAL_GPUS * BATCH_SIZE) )); then
  echo 'Set GRAD_ACCUM explicitly: global batch128 is not divisible by GPUs*batch.' >&2; exit 2
fi
export GRAD_ACCUM="${GRAD_ACCUM:-$((128 / (TOTAL_GPUS * BATCH_SIZE)))}"
export STEPS="${MAX_ITER:-${STEPS:-250}}"
SAVE_EVERY="${SAVE_EVERY:-5}"
for key in GRAD_ACCUM STEPS SAVE_EVERY; do
  if ! [[ "${!key}" =~ ^[1-9][0-9]*$ ]]; then echo "$key must be positive integer" >&2; exit 2; fi
done
export OUTPUT="${OUTPUT_DIR:-${OUTPUT:-$ROOT/outputs/wan1p3b_480p_450k_${EXPERIMENT}_gbs128_teacherinit_w1_fake5}}"
source "$ROOT/scripts/train_logging.sh"
export RESUME="${RESUME_PATH:-${RESUME:-auto}}"
export WEIGHT_INDEX="${WEIGHT_INDEX:-/mnt/data/butong/datasets/taxonomy_pipeline/output/youtube_sports_added/training_balanced/wds_weights/wds_weight_index.json}"
export CHECKPOINT="${CHECKPOINT:-/mnt/data/butong/Wan2.2/checkpoints/Wan2.1-T2V-1.3B}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
unset EMBEDDING_DIR PROMPT_EMBEDDINGS
printf 'PDD: nodes=%s x GPUs=%s, shard=%s (effective=%s), replicas=%s, batch=%s, accum=%s, global_batch=%s\n' \
 "$NNODES" "$NPROC_PER_NODE" "$FSDP_SHARD_SIZE" "$EFFECTIVE_FSDP" "$((TOTAL_GPUS / EFFECTIVE_FSDP))" \
 "$BATCH_SIZE" "$GRAD_ACCUM" "$((TOTAL_GPUS * BATCH_SIZE * GRAD_ACCUM))"
if [[ "$DRY_RUN" != 1 ]]; then
  "${PYTHON_BIN}" "$ROOT/scripts/check_data.py" --config "$CONFIG" \
    --weight-index "$WEIGHT_INDEX" --checkpoint "$CHECKPOINT" --check-shards --require-rows 450613
fi
MONITOR_ARGS=()
for mapping in TENSORBOARD_ENABLED:tensorboard_enabled TENSORBOARD_EVERY:tensorboard_every \
  FIXED_PROMPT_ENABLED:fixed_prompt_enabled FIXED_PROMPT_EVERY:fixed_prompt_every \
  FIXED_PROMPT_AT_START:fixed_prompt_at_start FIXED_PROMPT_AT_END:fixed_prompt_at_end \
  FIXED_PROMPT_COUNT:fixed_prompt_count FIXED_PROMPT_SEED:fixed_prompt_seed \
  FIXED_PROMPT_NFE:fixed_prompt_nfe FIXED_PROMPT_DECODE:fixed_prompt_decode \
  FIXED_PROMPT_EMBEDDINGS:fixed_prompt_embeddings WAN_ROOT:wan_root; do
  name="${mapping%%:*}"; key="${mapping#*:}"
  if [[ -n "${!name:-}" ]]; then
    value="${!name}"
    case "$name" in
      TENSORBOARD_ENABLED|FIXED_PROMPT_ENABLED|FIXED_PROMPT_AT_START|FIXED_PROMPT_AT_END|FIXED_PROMPT_DECODE)
        case "$value" in
          1|true|True) value=true ;; 0|false|False) value=false ;;
          *) echo "$name must be 0/1 or false/true" >&2; exit 2 ;;
        esac ;;
    esac
    MONITOR_ARGS+=(--set "$key=$value")
  fi
done
exec bash "$ROOT/scripts/train_pdd.sh" --set "save_every=$SAVE_EVERY" \
  --set "lr=${LR:-2e-5}" --set "workers=${NUM_WORKERS:-0}" "${MONITOR_ARGS[@]}" "$@"
