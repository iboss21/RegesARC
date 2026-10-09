# RegesARC — Project Handoff v0.1.0

## What This Is

One launcher for MoE models that don't fit in a consumer GPU. Combines Strata's aggressive single-model optimization with Colibri's multi-family expert residency. **Not a wrapper.** The whole system lives here.

```
python setup.py --list     # scan machine, verdict every model
python setup.py --toolchain  # write installers (does not run them)
```

## Current State (2026-10-07)

### Working
- `reges/scan.py` — reads RAM, GPU (nvidia-smi), disk free space. Stdlib only.
- `reges/catalog.py` — 9 models with placement metadata (disk_gb, trunk_gb, hot_gb).
- `reges/place.py` — verdict logic: CARD / DISK / REFUSE based on hardware.
- `setup.py --list` — runs scan + place for every catalog row. Prints verdicts.
- `tools/installers/` — bash installer script exists (Linux/Mac).

### Broken
- **`c/setup_toolchain.py` does not exist.** `setup.py --toolchain` imports it and will crash with ImportError. This is the #1 priority fix.
- No engine binary source code has been ported yet. `engines/` directory is empty.
- No download, serve, or chat functionality implemented.

### Missing (per Project File corrections)
- Windows toolchain installer (`install-toolchain.bat`) — only `.sh` exists.
- Full model catalog — needs all MoE families the engine will support (currently 9, should be ~20+).
- RegesCore 397B as #1 recommendation in catalog and install flow.
- Download/install flow that asks user to choose models and downloads them.

## Architecture (from docs/ARCHITECTURE.md)

```
reges serve
    scan  ->  gpu, ram, disk
    pick  ->  catalog verdict
    place ->  card  |  disk
    serve ->  OpenAI + Anthropic on one port
```

**Card path:** Dense trunk + hot experts on GPU. Cold experts stream. (Strata mechanism)
**Disk path:** Dense trunk in RAM. Routed experts on SSD, load when router selects them. (Colibri mechanism)

The scanner picks the path. The user does not.

## Critical Fixes Needed (In Order)

1. **Create `c/setup_toolchain.py`** — Windows + Linux toolchain installer with compiler detection, CUDA check, and .bat/.sh output. This is blocking `--toolchain`.
2. **Expand catalog** — Add all MoE families: deepseek_v41, glm53, laya, olmoe, gliner_decide, plus RegesCore 397B as first entry.
3. **Create download flow** — User chooses models, installer downloads to `RegesModels/` beside checkout.
4. **Engine abstraction** — `engines/` directory with family-specific adapters (qwen38, deepseek_v4, glm, etc.).
5. **Serve layer** — OpenAI + Anthropic compatible API on one port.

## What NOT to Do

- Do NOT put Strata or Colibri source inside this repo. They are not checkouts here.
- Do NOT use MIT/Apache license. AGPL-3.0 is set.
- Do NOT assume a specific disk size — this runs on anyone's machine.
- Do NOT hardcode model paths to one user's system.

## File Tree (Current)

```
RegesARC/
├── reges/                  # Python control plane
│   ├── __init__.py         # v0.1.0
│   ├── scan.py             # Machine scanner ✓
│   ├── catalog.py          # Model catalog (9 models) ✓
│   └── place.py            # Placement verdicts ✓
├── c/                      # Engine-side helpers
│   ├── __init__.py         # Empty placeholder
│   ├── tools/              # Offline utilities (qpack, conversion)
│   └── setup_toolchain.py  # ❌ MISSING — must be created
├── engines/                # Engine adapters (empty — next phase)
├── docs/                   # Architecture, status, handoff
├── tools/installers/       # Toolchain install scripts
├── setup.py               # Main entry point ✓
├── START-HERE.bat         # Launch script ✓
├── LICENSE                # AGPL-3.0 ✓
├── NOTICE                 # Copyright notices
└── README.md              # Project overview ✓
```

## RegesCore 397B — Priority Model

Per David's instructions, RegesCore must be the #1 recommendation in the catalog and install flow, just like Qwen is prioritized in Strata. This is the flagship model for RegesARC.

## Next Three Actions

1. Create `c/setup_toolchain.py` with Windows (.bat) + Linux (.sh) installer generation
2. Expand `reges/catalog.py` to include all MoE families with RegesCore 397B first
3. Run `python setup.py --list` and `python setup.py --toolchain` to verify both commands work
