#!/usr/bin/env bash
# Source after OUTPUT is resolved; child launchers inherit the same log stream.
if [[ "${DRY_RUN:-0}" != 1 && "${PDD_LOGGING_ACTIVE:-0}" != 1 ]]; then
  PDD_LOG_DIR="${LOG_DIR:-${OUTPUT:?OUTPUT must be set}/logs}"
  mkdir -p "$PDD_LOG_DIR"
  PDD_LOG_HOST="$(hostname)"
  PDD_LOG_HOST="${PDD_LOG_HOST//[^a-zA-Z0-9_.-]/_}"
  export PDD_LOG_FILE
  PDD_LOG_FILE="$(mktemp "$PDD_LOG_DIR/train_${PDD_LOG_HOST}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  export PDD_LOGGING_ACTIVE=1 PYTHONUNBUFFERED=1
  exec > >(tee -a "$PDD_LOG_FILE") 2>&1
  printf '[%s] Training log: %s\n' "$(date -Is)" "$PDD_LOG_FILE"
fi
