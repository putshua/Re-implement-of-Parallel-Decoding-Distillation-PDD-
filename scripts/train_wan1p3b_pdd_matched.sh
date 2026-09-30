#!/usr/bin/env bash
# Same-rollout control with DMD disabled by its existing config.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT=pdd_matched
exec bash "$ROOT/scripts/train_wan1p3b_480p_450k.sh" "$@"
