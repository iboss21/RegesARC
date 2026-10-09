# RegesARC

One launcher for Mixture-of-Experts models that do not fit in a consumer GPU.

Strata places the hot experts on the card. Colibri streams the cold experts from disk and covers the families. RegesARC is that decision in one command. Those repos are not inside this one. Copyright and the absorbed mechanisms are in `NOTICE`.

## Quick Start

```bash
# Scan your machine, see what models fit
python setup.py --list

# Install build tools (compiler, make)
python setup.py --toolchain

# Choose and download models to RegesModels/
python reges/install.py

# Serve on port 8080 (OpenAI + Anthropic compatible)
python reges/serve.py --port 8080
```

## What Runs Today

- `python setup.py --list` — scans RAM, GPU, and free disk, then verdicts every model in the catalog: `CARD`, `DISK`, or `NO`. Weights go to `RegesModels/` beside the checkout. Nothing is downloaded. Nothing binds port 8080.
- `python setup.py --toolchain` — writes `tools/installers/install-toolchain.bat` and `.sh`. Run the installer later, with any local server stopped. It installs a compiler. It does not install a model.
- `python reges/install.py` — interactive model selection. User picks from catalog, installer downloads weights to `RegesModels/`. Respects placement verdicts (CARD or DISK path).
- `python reges/serve.py --port 8080` — HTTP server with OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) compatible endpoints on one port.

## Architecture

```
reges serve
    scan  ->  gpu, ram, disk
    pick  ->  catalog verdict
    place ->  card  |  disk
    serve ->  OpenAI + Anthropic on one port
```

**Card path:** Dense trunk and hot experts live on the GPU. Cold experts stream. This is the Strata mechanism, and it is how a 125B MoE moves on 12 GB.

**Disk path:** Dense trunk stays in RAM. Routed experts stay on the SSD and load only when the router selects them. This is the Colibri mechanism, and it is how a 170B or 744B MoE answers when the card cannot hold the hot set.

The scanner picks the path. The user does not.

```
card can hold the hot set     -> card path
card cannot, RAM holds trunk  -> disk path
neither                       -> refuse, do not pretend it fits
```

## Catalog (11 Models)

| Model | Params | Placement | Notes |
|-------|--------|-----------|-------|
| **RegesCore 1.0 35B** | 397B MoE — Flagship | DISK | #1 recommendation, 69GB |
| Qwen3.8-Flash-Next | 125B MoE | CARD | Strata reference impl |
| Qwen3.6-35B-A3B | 35B MoE | CARD | Smaller Qwen variant |
| DeepSeek V4 Flash | 284B MoE | DISK | MLA attention family |
| DeepSeek V4.1 Flash | 552B MoE | DISK | Larger DeepSeek |
| GLM-5.3-Flash | 321B MoE | DISK | GLM family |
| GLM-5.2 | 744B MoE | DISK | Large GLM |
| Inkling | 975B MoE | DISK | Very large MoE |
| Kimi K3 | 2.8T MoE | NO | Too big for most hardware |
| OLMoE-7B-7B | 7B MoE | CARD | Small, fast |
| Laya | MoE | CARD | New family |

## File Structure

```
RegesARC/
├── reges/                  # Python control plane
│   ├── __init__.py         # v0.1.0
│   ├── scan.py             # Machine scanner (RAM, GPU, disk)
│   ├── catalog.py          # Model catalog with placement metadata
│   ├── place.py            # Placement verdicts (CARD/DISK/REFUSE)
│   ├── install.py          # Interactive model download flow
│   └── serve.py            # HTTP server (OpenAI + Anthropic API)
├── engines/                # Engine family adapters
│   └── __init__.py         # Qwen38, DeepSeekV4, GLM, RegesCore
├── c/                      # Engine-side helpers
│   ├── __init__.py
│   ├── setup_toolchain.py  # Toolchain probe and installer generation
│   └── tools/              # Offline utilities (qpack, conversion)
├── engines/                # Engine adapters (absorbed from upstream)
├── docs/                   # Architecture, status, handoff
├── tools/installers/       # Toolchain install scripts (.bat + .sh)
├── setup.py               # Main entry point (--list, --toolchain)
├── START-HERE.bat         # Windows launch script
├── LICENSE                # AGPL-3.0
├── NOTICE                 # Copyright notices
└── README.md              # This file
```

## Engine Adapters

Each model family gets its own adapter in `engines/`:

- **Qwen38Adapter** — Qwen3.8-Flash-Next (Strata reference, supports speculative decoding)
- **DeepSeekV4Adapter** — DeepSeek V4/V4.1 (MLA attention)
- **GLMAdapter** — GLM-5.2/5.3 family
- **RegesCoreAdapter** — RegesCore 397B flagship (8K context)

The runtime doesn't care whether it's Qwen or DeepSeek. Each adapter implements:
- `load_model(path)` — load weights from disk
- `generate(prompt, max_tokens)` — run inference
- `place_on_device(device)` — GPU card or SSD streaming

## API Endpoints

### Health Check
```bash
curl http://localhost:8080/health
```

### OpenAI Compatible
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

### Anthropic Compatible
```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "regescore",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

## License

AGPL-3.0. See `LICENSE` and `NOTICE`.

RegesCore and RegesCode remain under iBoss / LIKE A KING INC. copyright. Open source for community use, commercial licensing available for proprietary integration.
