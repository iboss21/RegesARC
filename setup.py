#!/usr/bin/env python3
"""RegesARC setup. Scan, verdict, write a toolchain installer. Never download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from c.setup_toolchain import plan, probe, write_installers  # noqa: E402
from reges import __version__  # noqa: E402
from reges.catalog import CATALOG  # noqa: E402
from reges.place import place  # noqa: E402
from reges.scan import scan  # noqa: E402

BANNER = r"""
 ____  _____ ____ _____ ____    _    ____   ____
|  _ \| ____/ ___| ____/ ___|  / \  |  _ \ / ___|
| |_) |  _|| |  _|  _| \___ \ / _ \ | |_) | |
|  _ <| |__| |_| | |___ ___) / ___ \|  _ <| |___
|_| \_\_____\____|_____|____/_/   \_\_| \_\\____|
"""


def store_dir() -> Path:
    return ROOT.parent / "RegesModels"


def cmd_list() -> int:
    print(BANNER)
    print(f"RegesARC {__version__}  ·  one launcher, card path or disk path")
    machine = scan(store_dir())
    print(
        f"ram {machine['ram_gb']} GB   gpu {machine['gpu_gb']} GB   "
        f"disk free {machine['disk_free_gb']} GB"
    )
    print(f"models -> {machine['store']}")
    print()
    for row in CATALOG:
        verdict = place(row, machine)
        mark = {"card": "CARD", "disk": "DISK", "refuse": "NO  "}[verdict["verdict"]]
        print(f"  {mark}  {row['label']:24} {row['params']:12} {verdict['reason']}")
    print()
    print("Nothing downloaded. Engine binary is not in this tree yet.")
    return 0


def cmd_toolchain() -> int:
    tools = probe()
    write_installers()
    print(plan(tools))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="setup.py")
    parser.add_argument("--list", action="store_true", help="scan and verdict every model")
    parser.add_argument("--toolchain", action="store_true", help="write the installer, do not run it")
    args = parser.parse_args()
    if args.toolchain:
        return cmd_toolchain()
    return cmd_list()


if __name__ == "__main__":
    raise SystemExit(main())
