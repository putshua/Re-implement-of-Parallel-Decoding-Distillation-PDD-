#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/shared_env.sh"
export PYTHONPATH="$ROOT/AnyFlow${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 OMP_NUM_THREADS=4
: "${CONFIG:?Set CONFIG to the saved training configuration}"
: "${PROMPTS:?Set PROMPTS to your evaluation embeddings}"
: "${WAN_ROOT:?Set WAN_ROOT to your Wan source checkout}"
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE:-4}" \
 -m pdd.evaluate --wan-root "$WAN_ROOT" --config "$CONFIG" \
 --prompts "$PROMPTS" \
 --output "${OUTPUT:-$ROOT/outputs/evaluation}" "$@"
