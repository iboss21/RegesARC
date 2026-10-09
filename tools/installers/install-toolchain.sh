#!/usr/bin/env bash
set -euo pipefail
echo "RegesARC toolchain installer"
if ! command -v gcc >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y build-essential
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed --noconfirm base-devel
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y gcc make
  else
    echo "Install gcc and make, then rerun." >&2
    exit 1
  fi
fi
echo "CUDA is optional."
