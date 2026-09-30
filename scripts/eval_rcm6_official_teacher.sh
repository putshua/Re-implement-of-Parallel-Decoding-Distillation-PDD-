#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/shared_env.sh"
export CONFIG="$ROOT/configs/wan1p3b_480p_450k_16gpu.json"
export PROMPTS="$ROOT/assets/rcm_fixed_prompts/prompts_000000000.pt"
export OUTPUT="${OUTPUT:-$ROOT/outputs/eval_rcm6_step100_pdd4_official_teacher_5seeds}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}" OMP_NUM_THREADS=4
bash "$ROOT/scripts/eval_pdd.sh" \
 --student "${STUDENT:-$ROOT/outputs/wan1p3b_480p_450k_rcm6_fsdp16_20260912/step_000100}" \
 --limit 6 --seeds 42 43 44 45 46 --nfe 4 --decode \
 --official-teacher-root "$ROOT/assets/wan21_official" \
 --official-negative "$ROOT/assets/wan21_official_negative/prompts_000000000.pt" "$@"
"$PYTHON_BIN" "$ROOT/scripts/summarize_eval.py" "$OUTPUT"
"$PYTHON_BIN" "$ROOT/scripts/eval_gallery.py" "$OUTPUT"
"$PYTHON_BIN" "$ROOT/scripts/compare_eval.py" "$OUTPUT"
"$PYTHON_BIN" "$ROOT/scripts/compare_latency.py" "$OUTPUT"
