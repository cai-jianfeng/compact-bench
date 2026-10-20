#!/usr/bin/env bash
# E2 anchor-fidelity track: 5 production prompts x summarizer + anchor_index.
# Usage: bash scripts/run_e2.sh <summarizer_model> [n_sessions]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: run_e2.sh <summarizer_model> [n_sessions]}"
N="${2:-0}"
exec python3 run.py --track retention --model "$MODEL" --limit "$N"
