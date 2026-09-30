#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/shared_env.sh"
export PYTHONPATH="$ROOT/AnyFlow${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 OMP_NUM_THREADS=4
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE:-4}" \
 -m pdd.evaluate --wan-root "$ROOT/../Wan2.2" --config "${CONFIG:-$ROOT/configs/wan13b_480p.json}" \
 --prompts "${PROMPTS:-$ROOT/../codebase/Bidirectional/test_prompts/ti2v_eval_curated_64_wan22_extended_umt5.pt}" \
 --output "${OUTPUT:-$ROOT/outputs/evaluation}" "$@"
