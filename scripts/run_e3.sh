#!/usr/bin/env bash
# E3 drift track: deterministic, no model calls.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 run.py --track production
