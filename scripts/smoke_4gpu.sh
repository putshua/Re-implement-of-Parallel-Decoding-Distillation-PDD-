#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-$ROOT/configs/smoke.json}" NPROC_PER_NODE=4
exec bash "$ROOT/scripts/train_pdd.sh" "$@"
