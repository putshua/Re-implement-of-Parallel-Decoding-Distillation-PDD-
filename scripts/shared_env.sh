#!/usr/bin/env bash
# Select Python from the active environment, or use the caller's explicit path.
if ! PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")" || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable. Activate an environment or set PYTHON_BIN." >&2
  exit 2
fi
export PYTHON_BIN
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONNOUSERSITE=1
