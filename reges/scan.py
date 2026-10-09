"""Machine scan. Stdlib only. Never downloads, never binds a port."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _ram_gb() -> float:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
        except Exception:
            return 0.0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                kb = float(line.split()[1])
                return kb / (1024**2)
    return 0.0


def _disk_free_gb(path: Path) -> float:
    target = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return 0.0
    return usage.free / (1024**3)


def _gpu_gb() -> float:
    """Best-effort. nvidia-smi is optional and must not fail the scan."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    total = 0.0
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            total += int(line) / 1024
    return total


def scan(store: Path) -> dict:
    return {
        "ram_gb": round(_ram_gb(), 1),
        "gpu_gb": round(_gpu_gb(), 1),
        "disk_free_gb": round(_disk_free_gb(store), 1),
        "store": str(store),
    }
