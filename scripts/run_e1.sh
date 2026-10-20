#!/usr/bin/env bash
# Compatibility entry point for Task Preservation.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: run_e1.sh <summarizer_model> [variant]}"
VARIANT="${2:-claude}"

exec python3 run.py --track task --model "$MODEL" --prompt "$VARIANT"
