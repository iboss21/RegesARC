"""Toolchain probe. Writes an installer. Never runs it. Never binds a port."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER_DIR = ROOT / "tools" / "installers"


def _which(name: str) -> str | None:
    return shutil.which(name)


def probe() -> dict:
    gcc = _which("gcc") or _which("x86_64-w64-mingw32-gcc")
    make = _which("make") or _which("mingw32-make")
    python = _which("python") or _which("python3")
    nvcc = _which("nvcc")
    msys = Path(r"C:\msys64\mingw64\bin\gcc.exe")
    return {
        "gcc": {"present": bool(gcc or msys.is_file()), "how": "MSYS2 mingw-w64 gcc, not MSVC"},
        "make": {"present": bool(make), "how": "make from the same mingw64 shell"},
        "python": {"present": bool(python), "how": "Python 3 for this launcher"},
        "nvcc": {"present": bool(nvcc), "how": "optional CUDA toolkit; card path is faster with it"},
    }


def write_installers() -> tuple[Path, Path]:
    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)
    bat = INSTALLER_DIR / "install-toolchain.bat"
    sh = INSTALLER_DIR / "install-toolchain.sh"
    bat.write_text(_BAT, encoding="utf-8", newline="\r\n")
    sh.write_text(_SH, encoding="utf-8", newline="\n")
    os.chmod(sh, 0o755)
    return bat, sh


def plan(tools: dict | None = None) -> str:
    tools = tools if tools is not None else probe()
    lines = ["Toolchain plan (nothing installed, port 8080 untouched):"]
    for key in ("gcc", "make", "python", "nvcc"):
        flag = "ok" if tools[key]["present"] else "missing"
        lines.append(f"  {key:8} {flag:8} {tools[key]['how']}")
    missing = [k for k in ("gcc", "make") if not tools[k]["present"]]
    if missing:
        lines.append("Run later, Strata stopped: tools\\installers\\install-toolchain.bat")
    return "\n".join(lines)


_BAT = r"""@echo off
setlocal
echo RegesARC toolchain installer. Stop any local server on 8080 first.
where gcc >nul 2>&1 && goto :python
if exist C:\msys64\usr\bin\bash.exe (
  C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-make"
  echo Add C:\msys64\mingw64\bin to PATH, then reopen the shell.
  goto :python
)
winget install -e --id MSYS2.MSYS2 --accept-package-agreements --accept-source-agreements
if exist C:\msys64\usr\bin\bash.exe (
  C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-make"
)
:python
where python >nul 2>&1 || winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
echo CUDA is optional and not installed here.
exit /b 0
"""

_SH = """#!/usr/bin/env bash
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
"""
