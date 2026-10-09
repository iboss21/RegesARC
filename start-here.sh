#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 setup.py --list
python3 setup.py --toolchain
echo
echo "RegesARC did not download a model and did not bind a port."
