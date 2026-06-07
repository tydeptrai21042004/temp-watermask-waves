#!/usr/bin/env bash
set -Eeuo pipefail

# One-command Arnold-Hess WAVES-style distortion benchmark.
# Override by passing arguments, e.g.:
#   ./run_arnold_hess_waves.sh --limit 5000 --attack-preset distortion-full

python scripts/run_arnold_hess_waves_benchmark.py "$@"
