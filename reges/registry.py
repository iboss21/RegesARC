"""Model registry for RegesARC.

Maps model IDs to download sources, file manifests, family classification,
and placement hints. Uses placeholder URLs that demonstrate the structure —
real URLs will be filled in when actual model repositories are identified.

The system works end-to-end even with placeholder URLs (creating proper
directory structures and manifests).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """A single file to download within a model."""
    path: str                    # Relative path within the model directory
    url: str                     # Download URL (placeholder)
    size_bytes: int = 0          # Expected size, 0 = unknown
    sha256: Optional[str] = None # Expected SHA-256, None = skip verification

@dataclass
class ModelEntry:
    """Complete registry entry for one model."""
    id: str                      # Unique model identifier (matches catalog)
    label: str                   # Human-readable name
    family: str                  # Family classification
    params: str                  # Parameter count / description
    disk_gb: int = 0             # Total download size in GB

    # Download sources — try in order
    urls: list[str] = field(default_factory=list)     # Primary URLs (HF, mirrors)
    manifest: list[FileEntry] = field(default_factory=list)  # Files to download

    # Placement hints
    placement_hint: str = "auto"   # "card", "disk", or "auto"

    # Gated model flag — requires HF_TOKEN env var
    gated: bool = False

    # Notes for operators
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry data
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelEntry] = {}


def _register(entry: ModelEntry) -> None:
    """Register a model entry."""
    _REGISTRY[entry.id] = entry


# --- RegesCore 397B MoE (Flagship) ---
_register(ModelEntry(
    id="regescore-1.0-35",
    label="RegesCore 1.0 35B",
    family="regescore",
    params="397B MoE — Flagship",
    disk_gb=69,
    urls=[
        "https://huggingface.co/reges-ai/regescore-1.0-35/resolve/main/",
        # Mirror placeholder: "https://mirror.example.com/regescore-1.0-35/"
    ],
    manifest=[
        FileEntry(path="regescore.gguf", url="", size_bytes=69 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=2048, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=5 * 1024**2, sha256=None),
    ],
    placement_hint="card",
    notes="Flagship model. Hot experts on GPU, cold experts stream from disk.",
))

# --- Qwen3.8-Flash-Next (125B MoE) ---
_register(ModelEntry(
    id="qwen3.8-flash-next",
    label="Qwen3.8-Flash-Next",
    family="qwen38",
    params="125B MoE",
    disk_gb=80,
    urls=[
        "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/",
    ],
    manifest=[
        FileEntry(path="qwen3.8-flash-next.gguf", url="", size_bytes=80 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=1500, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=4 * 1024**2, sha256=None),
    ],
    placement_hint="card",
))

# --- Qwen3.6-35B-A3B (35B MoE) ---
_register(ModelEntry(
    id="qwen3.6",
    label="Qwen3.6-35B-A3B",
    family="qwen36",
    params="35B MoE",
    disk_gb=22,
    urls=[
        "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/",
    ],
    manifest=[
        FileEntry(path="qwen3.6.gguf", url="", size_bytes=22 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=1200, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=3 * 1024**2, sha256=None),
    ],
    placement_hint="card",
))

# --- DeepSeek V4 Flash (284B MoE) ---
_register(ModelEntry(
    id="deepseek-v4-flash",
    label="DeepSeek V4 Flash",
    family="deepseek-v4",
    params="284B MoE",
    disk_gb=160,
    urls=[
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/",
    ],
    manifest=[
        FileEntry(path="deepseek-v4-flash.gguf", url="", size_bytes=160 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=1800, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=4 * 1024**2, sha256=None),
    ],
    placement_hint="card",
))

# --- DeepSeek V4.1 Flash (552B MoE) — large, disk path recommended ---
_register(ModelEntry(
    id="deepseek-v4.1-flash",
    label="DeepSeek V4.1 Flash",
    family="deepseek-v4",
    params="552B MoE",
    disk_gb=510,
    urls=[
        "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/",
    ],
    manifest=[
        FileEntry(path="deepseek-v4.1-flash.gguf", url="", size_bytes=510 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=2000, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=5 * 1024**2, sha256=None),
    ],
    placement_hint="disk",
    notes="Very large model. Trunk in RAM, routed experts stream from SSD.",
))

# --- GLM-5.3-Flash (321B MoE) ---
_register(ModelEntry(
    id="glm-5.3-flash",
    label="GLM-5.3-Flash",
    family="glm",
    params="321B MoE",
    disk_gb=180,
    urls=[
        "https://huggingface.co/THUDM/GLM-5.3-Flash/resolve/main/",
    ],
    manifest=[
        FileEntry(path="glm-5.3-flash.gguf", url="", size_bytes=180 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=1700, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=4 * 1024**2, sha256=None),
    ],
    placement_hint="card",
))

# --- GLM-5.2 (744B MoE) — very large ---
_register(ModelEntry(
    id="glm-5.2",
    label="GLM-5.2",
    family="glm",
    params="744B MoE",
    disk_gb=372,
    urls=[
        "https://huggingface.co/THUDM/GLM-5.2/resolve/main/",
    ],
    manifest=[
        FileEntry(path="glm-5.2.gguf", url="", size_bytes=372 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=2200, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=5 * 1024**2, sha256=None),
    ],
    placement_hint="disk",
    notes="Large model. Trunk in RAM, routed experts stream from SSD.",
))

# --- Inkling (975B MoE) — very large ---
_register(ModelEntry(
    id="inkling",
    label="Inkling",
    family="inkling",
    params="975B MoE",
    disk_gb=500,
    urls=[
        "https://huggingface.co/inkling-ai/Inkling/resolve/main/",
    ],
    manifest=[
        FileEntry(path="inkling.gguf", url="", size_bytes=500 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=2000, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=5 * 1024**2, sha256=None),
    ],
    placement_hint="disk",
))

# --- Kimi K3 (2.8T MoE) — massive, disk path only ---
_register(ModelEntry(
    id="kimi-k3",
    label="Kimi K3",
    family="kimi",
    params="2.8T MoE",
    disk_gb=1400,
    urls=[
        "https://huggingface.co/MoonshotAI/Kimi-K3/resolve/main/",
    ],
    manifest=[
        FileEntry(path="kimi-k3.gguf", url="", size_bytes=1400 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=2500, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=6 * 1024**2, sha256=None),
    ],
    placement_hint="disk",
    notes="Massive model. Requires large SSD for expert streaming.",
))

# --- OLMoE-7B-7B (small, fits anywhere) ---
_register(ModelEntry(
    id="olmoe",
    label="OLMoE-7B-7B",
    family="olmoe",
    params="7B MoE",
    disk_gb=6,
    urls=[
        "https://huggingface.co/allenai/OLMoE-7B-7B/resolve/main/",
    ],
    manifest=[
        FileEntry(path="olmoe.gguf", url="", size_bytes=6 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=800, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=1.5 * 1024**2, sha256=None),
    ],
    placement_hint="card",
))

# --- Laya (MoE) ---
_register(ModelEntry(
    id="laya",
    label="Laya",
    family="laya",
    params="MoE",
    disk_gb=20,
    urls=[
        "https://huggingface.co/laya-ai/Laya/resolve/main/",
    ],
    manifest=[
        FileEntry(path="laya.gguf", url="", size_bytes=20 * 1024**3, sha256=None),
        FileEntry(path="config.json", url="", size_bytes=1000, sha256=None),
        FileEntry(path="tokenizer.json", url="", size_bytes=3 * 1024**2, sha256=None),
    ],
    placement_hint="card",
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

REGISTRY = _REGISTRY  # public alias for direct access


def get(entry_id: str) -> ModelEntry:
    """Look up a model entry by ID. Raises KeyError if not found."""
    return _REGISTRY[entry_id]


def all_entries() -> list[ModelEntry]:
    """Return all registered model entries in registration order."""
    return list(_REGISTRY.values())


def families() -> dict[str, list[ModelEntry]]:
    """Group entries by family classification."""
    groups: dict[str, list[ModelEntry]] = {}
    for entry in _REGISTRY.values():
        groups.setdefault(entry.family, []).append(entry)
    return groups


def needs_token(entry_id: str) -> bool:
    """Check if a model requires HF_TOKEN (gated)."""
    try:
        return get(entry_id).gated
    except KeyError:
        return False


def resolve_url(entry_id: str, index: int = 0) -> str:
    """Resolve the primary download URL for a model.

    Args:
        entry_id: Model ID.
        index: Which URL to use (0 = first/primary).

    Returns:
        The resolved URL string.

    Raises:
        KeyError: If model not in registry.
        IndexError: If no URLs available or index out of range.
    """
    entry = get(entry_id)
    if not entry.urls:
        raise IndexError(f"No download URLs registered for {entry_id}")
    return entry.urls[min(index, len(entry.urls) - 1)]


def build_manifest_dict(entry_id: str) -> dict:
    """Build a manifest dictionary for a model (for writing to disk)."""
    entry = get(entry_id)
    files = []
    for f in entry.manifest:
        files.append({
            "path": f.path,
            "url": f.url or resolve_url(entry_id),
            "size_bytes": f.size_bytes,
            "sha256": f.sha256,
        })

    return {
        "id": entry.id,
        "label": entry.label,
        "family": entry.family,
        "params": entry.params,
        "disk_gb": entry.disk_gb,
        "placement_hint": entry.placement_hint,
        "gated": entry.gated,
        "notes": entry.notes,
        "files": files,
    }


def model_dir_name(entry_id: str) -> str:
    """Return the recommended directory name for a model in RegesModels/."""
    return entry_id
