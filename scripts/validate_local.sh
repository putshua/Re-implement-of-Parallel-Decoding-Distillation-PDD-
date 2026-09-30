#!/usr/bin/env bash
# Full local engineering validation on manifest-derived prompts. No full-data quality claim.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/shared_env.sh"
export CONFIG="${CONFIG:-$ROOT/configs/local128_480p.json}"
export BATCH_SIZE="${BATCH_SIZE:-2}" GRAD_ACCUM="${GRAD_ACCUM:-1}" STEPS="${STEPS:-8}"
export OUTPUT="${OUTPUT:-$ROOT/outputs/local_validation_480p}"
bash "$ROOT/scripts/train_pdd.sh"
STEP_DIR="$(printf '%s/step_%06d' "$OUTPUT" "$STEPS")"
CONFIG="$CONFIG" OUTPUT="$OUTPUT/evaluation" bash "$ROOT/scripts/eval_pdd.sh" \
 --student "$STEP_DIR" --limit 4 --seeds 42 43 --nfe 2 4 8 --decode
"${PYTHON_BIN}" "$ROOT/scripts/summarize_eval.py" "$OUTPUT/evaluation"
