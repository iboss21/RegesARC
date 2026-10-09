#!/usr/bin/env python3
"""Which engine can serve this model on this machine, and how to start it.

This module is the thing neither Strata nor Colibri has. Each of them *is* one way of
placing a model: Strata keeps the dense trunk and the hot experts on the card, Colibri
keeps the trunk in RAM and streams routed experts from disk. RegesARC chooses between
them per model, per machine, and then serves whichever it chose behind one port.

The policy has three outcomes and the third one is not an error:

    card   the card can hold the trunk plus a working set of experts  -> the fast path
    disk   the card cannot, RAM can hold the trunk                    -> experts stream
    none   neither                                                    -> say so plainly

`none` matters. A tool that quietly starts a 744B model on a 12 GB card and reports 2
tokens/s has lied to the user.

Nothing here binds a port unless it is starting a server, and the port it starts on is
chosen free. A Strata already running on 8080 is found and reported, never restarted,
never signalled, never connected to.
"""
import os
import re
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GB = 1024 ** 3

#: Engines we know how to drive. The old Colibri binary names stay as candidates so an
#: engine someone already built is still found — the rename to reges-* happens with the
#: Makefile, not before it (docs/STATUS.md).
ENGINES = {
    "strata": {
        "path": "card",
        "binary": ("strata.exe", "strata"),
        "search": (r"G:\AI\RegesArc\Strata-main", "~/Strata", "/opt/strata"),
        "note": "keeps the dense trunk and the hot experts on the card",
    },
    "colibri": {
        "path": "disk",
        "binary": ("reges-{family}.exe", "reges-{family}", "{family}.exe", "{family}"),
        "search": (ROOT,),
        "note": "trunk in RAM, routed experts streamed from disk",
    },
}

# What the card has to hold on the card path. Not the experts: both engines keep the
# dense trunk (shared attention and feed-forward, embedding, router) on the card and pull
# routed experts up from system RAM as the router asks for them. That is what makes a
# 125B or 170B MoE run on 12 GB, and it is why this test is about the trunk.
#
# 0.70 is not a guess taken from thin air: on this machine a Qwen3.8-Flash-Next with an
# 8 GB trunk runs on a 12 GB card (8/12 = 0.67), and a 35B model with an 18 GB trunk does
# not. The remaining 30% is the KV cache, the CUDA context and the activation buffers,
# which is what the engines cannot shrink. Replace this with the engine's own residency
# report as soon as one exposes it (docs/STATUS.md).
CARD_TRUNK_FRACTION = 0.70
#: RAM left for the OS and the page cache on the disk path.
RAM_HEADROOM = 0.85


# ---------------------------------------------------------------- engine detection

def _expand(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _candidate_names(spec, family):
    names = []
    for name in spec["binary"]:
        if "{family}" in name:
            if family:
                names.append(name.format(family=family))
        else:
            names.append(name)
    return names


def _look_in(dirs, names):
    for directory in dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for sub in ("", "bin", "build", "c", "release"):
            base = os.path.join(directory, sub) if sub else directory
            for name in names:
                hit = os.path.join(base, name)
                if os.path.isfile(hit):
                    return hit
    return None


def _engine_version(path):
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True,
                             errors="replace", timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", (out.stdout or "") + (out.stderr or ""))
    return match.group(1) if match else None


def detect_engines(family=None):
    """What is installed and runnable. Read-only: no process is started, no port is used.

    Returns {engine: {found, path, version, kind, note}}. `kind` is "binary" for a built
    engine, "source" for a tree that could be built, None for absent.
    """
    found = {}
    for name, spec in ENGINES.items():
        info = {"found": False, "path": None, "version": None, "kind": None,
                "note": spec["note"]}
        names = _candidate_names(spec, family)
        on_path = next((shutil.which(n) for n in names if shutil.which(n)), None)
        hit = on_path or _look_in([_expand(d) for d in spec["search"]]
                                  + [os.environ.get("REGES_ENGINE_DIR", "")], names)
        if hit:
            info.update(found=True, path=hit, kind="binary")
            info["version"] = _engine_version(hit)
        else:
            # A checkout that could be run or built. Strata is normally run as Python with
            # a prebuilt engine beside it, so its marker is serve/server.py, not a binary.
            markers = {"strata": ("serve/server.py",), "colibri": ("Makefile",)}
            for directory in [_expand(d) for d in spec["search"]]:
                marker = next((os.path.join(directory, m) for m in markers[name]
                               if os.path.isfile(os.path.join(directory, m))), None)
                if marker:
                    info.update(found=True, path=marker, kind="source")
                    break
        found[name] = info
    return found


def listening_ports():
    """Ports already taken on localhost, read from the process table only.

    Deliberately does not open a socket to them: a request to a model server can wake a
    model, and the user may be mid-conversation in one. We only need the number.
    """
    ports = []
    for cmd in (["netstat", "-ano", "-p", "tcp"], ["netstat", "-ano"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                                 timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode != 0 and not out.stdout:
            continue
        for line in (out.stdout or "").splitlines():
            if "LISTENING" not in line.upper() and "LISTEN" not in line.upper():
                continue
            match = re.search(r"127\.0\.0\.1:(\d+)\s", line)
            if match:
                ports.append(int(match.group(1)))
        if ports:
            break
    return sorted(set(ports))


def free_port(prefer=None, avoid=()):
    """A port nothing is listening on. A running Strata keeps its port."""
    avoid = set(avoid) | set(listening_ports())
    if prefer and prefer not in avoid:
        return prefer
    for candidate in range(8100, 8200):
        if candidate in avoid:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    raise OSError("no free port between 8100 and 8199")


# ---------------------------------------------------------------- the policy

def _vram_gb(hw):
    """The largest usable card in GiB, or None when there is no usable GPU."""
    best = None
    for card in hw.get("nvidia") or []:
        # setup_hw normalises nvidia-smi output to bytes under these names.
        total = card.get("total_bytes") or card.get("memory.total") or card.get("memory_total")
        if total:
            best = max(best or 0.0, float(total) / GB)
    device = (hw.get("gpu") or {}).get("vulkan")
    if device:
        total = device.get("memory") or device.get("memory_bytes")
        if total:
            best = max(best or 0.0, float(total) / GB)
    return best


def _ram_gb(hw):
    """(total, available) in GiB. Both, because they answer different questions.

    Total is what the machine can do when it is not doing something else. Available is
    what it can do right now — and right now something like Strata may be holding most of
    it for a running model, which is a fact about the moment, not about the machine. The
    verdict is against total; the difference is reported.
    """
    memory = hw.get("memory") or {}
    total = memory.get("total")
    available = memory.get("available")
    to_gb = lambda v: float(v) / GB if v else None
    return to_gb(total), to_gb(available)


def _pick(engines, path, family):
    """The first engine that can serve this path and is actually here."""
    for name, spec in ENGINES.items():
        if spec["path"] != path:
            continue
        info = engines.get(name) or {}
        if not info.get("found"):
            continue
        if info.get("kind") == "binary":
            return name
        if info.get("kind") == "source":
            # A checkout that is not built. Colibri needs a compiler; Strata is normally
            # run as Python against a prebuilt engine, which is a different thing to get.
            return f"{name} (needs building)" if path == "disk" else f"{name} (needs its engine)"
    return None


def place(entry, hw, engines=None):
    """The verdict for one catalog model on this machine.

    `entry` is a setup_catalog.CatalogModel. Its `dense_gb` is the part that must stay
    resident — shared attention and feed-forward, embedding, router; the rest is routed
    experts, which is what both engines move around.

    The card-path test is a working-set estimate, not a measurement: it asks whether the
    trunk plus a third of the expert bytes fits in VRAM with headroom. Replace the
    fraction with measured residency once an engine reports it (docs/STATUS.md).
    """
    engines = engines if engines is not None else detect_engines(entry.family)
    vram = _vram_gb(hw)
    ram, ram_free = _ram_gb(hw)
    free = (hw.get("disk") or {}).get("free_bytes")
    disk_gb = float(free) / GB if free else None
    dense = float(getattr(entry, "dense_gb", 0.0) or 0.0)
    experts = max(float(getattr(entry, "disk_gb", 0.0) or 0.0) - dense, 0.0)
    facts = {"vram_gb": vram, "ram_gb": ram, "ram_free_gb": ram_free, "disk_gb": disk_gb,
             "dense_gb": dense, "expert_gb": round(experts, 1)}
    # A machine busy with something else is not a smaller machine. The verdict is against
    # what the machine has; what is free at this second is reported alongside it.
    busy = (f" Note: only {ram_free:.0f} GB of it is free right now, so nothing else can "
            "run until whatever holds the rest is stopped."
            if ram and ram_free and ram_free < dense <= ram * RAM_HEADROOM else "")

    if disk_gb is not None and disk_gb < float(entry.disk_gb):
        return {"path": "none", "engine": None, "model": entry.id, "facts": facts,
                "why": f"the weights are {entry.disk_gb:.0f} GB and that folder has "
                       f"{disk_gb:.0f} GB free. That is a disk question, not a GPU one - "
                       "another drive may hold it."}

    if vram and dense and dense <= vram * CARD_TRUNK_FRACTION:
        engine = _pick(engines, "card", entry.family)
        if engine:
            return {"path": "card", "engine": engine, "model": entry.id,
                    "facts": {**facts, "vram_budget_gb": round(vram * CARD_TRUNK_FRACTION, 1)},
                    "why": f"the {dense:.1f} GB trunk fits the {vram:.0f} GB card with "
                           "room left for the KV cache; routed experts are pulled up from "
                           "RAM as the router asks for them"}
        return {"path": "none", "engine": None, "model": entry.id, "facts": facts,
                "why": "this machine could take the card path, but no card-path engine is "
                       "installed here. See: python setup.py --toolchain"}

    if ram and dense and dense <= ram * RAM_HEADROOM:
        engine = _pick(engines, "disk", entry.family)
        if engine:
            return {"path": "disk", "engine": engine, "model": entry.id,
                    "facts": {**facts, "ram_budget_gb": round(ram * RAM_HEADROOM, 1)},
                    "why": f"the card cannot hold it; the {dense:.1f} GB trunk fits in RAM "
                           f"and the {experts:.0f} GB of routed experts stream from disk. "
                           "A fast NVMe is what makes this usable." + busy}
        return {"path": "none", "engine": None, "model": entry.id, "facts": facts,
                "why": "this machine could take the disk path, but no engine is built for "
                       "it. See: python setup.py --toolchain"}

    if ram and dense:
        return {"path": "none", "engine": None, "model": entry.id, "facts": facts,
                "why": f"the dense trunk alone is {dense:.0f} GB, and this machine has "
                       f"{ram:.0f} GB of RAM. Not pretending otherwise."}

    return {"path": "none", "engine": None, "model": entry.id, "facts": facts,
            "why": "no usable memory or GPU report, so no honest verdict."}


# ---------------------------------------------------------------- starting one

def launch(engine, model_dir, *, port=None, host="127.0.0.1", extra=(), log=None):
    """Start one engine on its own port and hand back the process.

    Never binds a port something else has. Never signals a process it did not start.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    info = detect_engines().get(engine) or {}
    binary = info.get("path")
    if not binary or info.get("kind") != "binary":
        raise FileNotFoundError(f"{engine} is not built or installed here; run "
                                "python setup.py --toolchain")
    port = port or free_port(avoid=[8080])
    cmd = [binary, "--model", str(model_dir), "--host", host, "--port", str(port), *extra]
    handle = open(log, "ab") if log else subprocess.DEVNULL
    process = subprocess.Popen(cmd, stdout=handle, stderr=handle)
    return {"engine": engine, "pid": process.pid, "port": port, "process": process,
            "base_url": f"http://{host}:{port}/v1"}


def report(verdicts, out=print):
    counts = {"card": 0, "disk": 0, "none": 0}
    for verdict in verdicts:
        counts[verdict["path"]] = counts.get(verdict["path"], 0) + 1
        mark = {"card": ">>", "disk": "+ ", "none": "  "}[verdict["path"]]
        out(f" {mark} {verdict['model']:<24} {verdict['path']:<5} "
            f"{verdict.get('engine') or '-'}")
        out(f"      {verdict['why']}")
    out(f"\n  card {counts.get('card', 0)}   disk {counts.get('disk', 0)}   "
        f"not here {counts.get('none', 0)}")


if __name__ == "__main__":
    sys.exit("setup_engines.py is a module; run `python setup.py --engines`")
