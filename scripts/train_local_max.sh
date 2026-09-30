#!/usr/bin/env bash
# Resolution for local batch validation: 4 x A800 80GB, 832x480, 81 frames, N=128 Midpoint.
# Uses the explicitly prepared 128-prompt engineering subset; cluster_train.sh uses full data.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-$ROOT/configs/local128_480p.json}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}" BATCH_SIZE="${BATCH_SIZE:-4}" GRAD_ACCUM="${GRAD_ACCUM:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export OUTPUT="${OUTPUT:-$ROOT/outputs/local_maxbatch}"
exec bash "$ROOT/scripts/train_pdd.sh" "$@"
