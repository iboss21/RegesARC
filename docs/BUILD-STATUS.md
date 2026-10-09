# RegesARC Build Status

## Current State (2026-10-07)

### ✅ Working Components

**Python Control Plane:**
- `reges.scan.py` — Machine scanner (RAM, GPU, disk free space) ✓
- `reges.catalog.py` — 11 models with placement metadata ✓
- `reges.place.py` — Placement verdicts (CARD/DISK/REFUSE) ✓
- `reges.registry.py` — Model registry with download URLs ✓
- `reges.install.py` — Interactive model installer ✓
- `reges.downloader.py` — HTTP streaming downloader with resume/checksums ✓
- `reges.serve.py` — OpenAI + Anthropic compatible API on one port ✓
- `reges.chat.py` — Terminal chat client with streaming support ✓
- `reges.verify.py` — Model verification (GGUF header, checksums) ✓

**Engine Adapters:**
- `engines/__init__.py` — Python engine fallback (stub implementations) ✓
- Supports: qwen38, deepseek-v4, glm, regescore families ✓

**Build System:**
- `c/CMakeLists.txt` — CMake build configuration ✓
- `c/setup_toolchain.py` — Toolchain probe and installer generation ✓
- `setup.py --list` — Scan machine and verdict every model ✓
- `setup.py --toolchain` — Write installers (does not run them) ✓

### ⚠️ Pending Components

**C/C++ Engine:**
- Source code exists in `c/` directory (headers, .c/.cpp files) ✓
- CMakeLists.txt configured for CUDA/HIP/CPU backends ✓
- **Missing:** cmake and make not installed on system
  - CMake: winget install failed (not found in registry)
  - Make: pacman/mingw64 packages not available
  - Status: Python fallback works, C engine requires manual toolchain installation

**Model Downloads:**
- Registry has placeholder URLs for all 11 models ✓
- Downloader supports resume, checksums, concurrent downloads ✓
- **Status:** No models downloaded yet (awaiting user selection)

## System Verification

### Machine Specs
```
RAM: 79.9 GB
GPU: 12.0 GB (NVIDIA RTX 4060 Ti)
Disk Free: 549.6 GB
```

### Model Verdicts
| Status | Model | Params | Reason |
|--------|-------|--------|--------|
| DISK | RegesCore 1.0 35B | 397B MoE | trunk 16 GB fits in 79.9 GB RAM; experts stream from disk |
| CARD | Qwen3.8-Flash-Next | 125B MoE | hot set 12 GB fits in 12.0 GB GPU |
| CARD | Qwen3.6-35B-A3B | 35B MoE | hot set 8 GB fits in 12.0 GB GPU |
| DISK | DeepSeek V4 Flash | 284B MoE | trunk 16 GB fits in 79.9 GB RAM; experts stream from disk |
| DISK | DeepSeek V4.1 Flash | 552B MoE | trunk 24 GB fits in 79.9 GB RAM; experts stream from disk |
| DISK | GLM-5.3-Flash | 321B MoE | trunk 16 GB fits in 79.9 GB RAM; experts stream from disk |
| DISK | GLM-5.2 | 744B MoE | trunk 12 GB fits in 79.9 GB RAM; experts stream from disk |
| DISK | Inkling | 975B MoE | trunk 24 GB fits in 79.9 GB RAM; experts stream from disk |
| NO | Kimi K3 | 2.8T MoE | need ~1400 GB free, have 549.6 GB |
| CARD | OLMoE-7B-7B | 7B MoE | hot set 6 GB fits in 12.0 GB GPU |
| CARD | Laya | MoE | hot set 10 GB fits in 12.0 GB GPU |

## How to Run

### Option 1: Python Fallback (Immediate)
```bash
cd G:/AI/RegesArc/RegesARC

# Scan machine and show verdicts
python setup.py --list

# Start serve layer (OpenAI + Anthropic API on port 8080)
python reges/serve.py --port 8080

# In another terminal, start chat client
python reges/chat.py --model regescore-1.0-35
```

### Option 2: Full C Engine (Requires Toolchain)
```bash
# Install cmake and make manually:
# - Download CMake from https://cmake.org/download/
# - Or use MSYS2: pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-make

cd G:/AI/RegesArc/RegesARC/c

# Build engine
mkdir build && cd build
cmake .. -DUSE_CUDA=ON  # or OFF for CPU-only
cmake --build . --config Release

# Copy binaries to project root
cp reges_engine.exe ../reges_engine.exe
```

## Architecture

```
RegesARC Launcher
├── Python Control Plane (working)
│   ├── scan.py → machine specs
│   ├── catalog.py → 11 models
│   ├── place.py → verdict logic
│   ├── registry.py → download URLs
│   ├── install.py → interactive installer
│   ├── downloader.py → HTTP streaming
│   ├── serve.py → OpenAI + Anthropic API
│   ├── chat.py → terminal client
│   └── verify.py → model verification
│
├── Engine Adapters (Python fallback)
│   └── engines/__init__.py
│       ├── Qwen38Adapter
│       ├── DeepSeekV4Adapter
│       ├── GLMAdapter
│       └── RegesCoreAdapter
│
├── C/C++ Engine (pending toolchain)
│   ├── c/ directory (source code exists)
│   ├── CMakeLists.txt (configured)
│   └── family_*.cpp (adapter implementations)
│
└── Models Storage
    └── RegesModels/ (beside checkout, never in git)
```

## Next Steps

1. **Immediate:** Use Python fallback for testing and development
2. **Short-term:** Install cmake/make to enable C engine build
3. **Medium-term:** Download first model (Qwen3.8-Flash-Next recommended for CARD path)
4. **Long-term:** Optimize C engine with CUDA kernels for production use

## Notes

- Python fallback returns placeholder responses for testing
- Real inference requires either:
  - C/C++ engine compiled from `c/` source, OR
  - Integration with external inference backend (llama.cpp, vLLM, etc.)
- Model downloads use placeholder URLs — real URLs needed when repositories are identified
- Toolchain installer (`tools/installers/install-toolchain.bat`) ready to run once cmake/make available
