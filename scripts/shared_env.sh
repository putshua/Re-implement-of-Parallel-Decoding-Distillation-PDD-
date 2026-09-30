#!/usr/bin/env bash
# Shared interpreter used by every node; never fall back to node-local Python.
export PYTHON_BIN="${PYTHON_BIN:-/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Shared Python is not executable: $PYTHON_BIN. Check the shared mount on this node." >&2
  exit 2
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONNOUSERSITE=1
