"""Engine family adapters.

Each model family gets its own adapter module in engines/.
The runtime doesn't care whether it's Qwen or DeepSeek.
Engines are absorbed from pinned upstream tags, recorded in NOTICE.
Not git submodules. Not nested repos.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional


class EngineAdapter(abc.ABC):
    """Base class for model family adapters."""
    
    @property
    @abc.abstractmethod
    def family_id(self) -> str:
        """Unique identifier for this family (e.g., 'qwen38', 'deepseek-v4')."""
        ...
    
    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable name."""
        ...
    
    @abc.abstractmethod
    def load_model(self, model_path: Path) -> bool:
        """Load a model from the given path. Returns True on success."""
        ...
    
    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt. Returns generated string."""
        ...
    
    @abc.abstractmethod
    def place_on_device(self, device: str) -> bool:
        """Place model on 'card' (GPU) or 'disk' (SSD streaming)."""
        ...
    
    @property
    def supports_speculative_decoding(self) -> bool:
        """Whether this family supports MTP/speculative decoding."""
        return False
    
    @property
    def max_context_length(self) -> int:
        """Maximum context length in tokens."""
        return 4096


class Qwen38Adapter(EngineAdapter):
    """Qwen3.8-Flash-Next adapter. Strata's reference implementation."""
    
    @property
    def family_id(self) -> str:
        return "qwen38"
    
    @property
    def label(self) -> str:
        return "Qwen3.8-Flash-Next"
    
    def load_model(self, model_path: Path) -> bool:
        # In production: load GGUF weights, initialize CUDA kernels
        print(f"[qwen38] Loading model from {model_path}")
        return True
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # In production: run inference through optimized kernels
        return f"[qwen38] Generated response to: {prompt[:50]}..."
    
    def place_on_device(self, device: str) -> bool:
        if device == "card":
            print("[qwen38] Placing hot experts on GPU")
            return True
        elif device == "disk":
            print("[qwen38] Streaming cold experts from disk")
            return True
        return False
    
    @property
    def supports_speculative_decoding(self) -> bool:
        return True  # Qwen has MTP heads


class DeepSeekV4Adapter(EngineAdapter):
    """DeepSeek V4/V4.1 adapter."""
    
    @property
    def family_id(self) -> str:
        return "deepseek-v4"
    
    @property
    def label(self) -> str:
        return "DeepSeek V4 Family"
    
    def load_model(self, model_path: Path) -> bool:
        print(f"[deepseek-v4] Loading model from {model_path}")
        return True
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return f"[deepseek-v4] Generated response to: {prompt[:50]}..."
    
    def place_on_device(self, device: str) -> bool:
        if device == "card":
            print("[deepseek-v4] Placing on GPU (MLA attention)")
            return True
        elif device == "disk":
            print("[deepseek-v4] Streaming routed experts from SSD")
            return True
        return False


class GLMAdapter(EngineAdapter):
    """GLM-5.2/5.3 adapter."""
    
    @property
    def family_id(self) -> str:
        return "glm"
    
    @property
    def label(self) -> str:
        return "GLM Family"
    
    def load_model(self, model_path: Path) -> bool:
        print(f"[glm] Loading model from {model_path}")
        return True
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return f"[glm] Generated response to: {prompt[:50]}..."
    
    def place_on_device(self, device: str) -> bool:
        if device == "card":
            print("[glm] Placing on GPU")
            return True
        elif device == "disk":
            print("[glm] Streaming from SSD")
            return True
        return False


class RegesCoreAdapter(EngineAdapter):
    """RegesCore 397B adapter. Flagship model."""
    
    @property
    def family_id(self) -> str:
        return "regescore"
    
    @property
    def label(self) -> str:
        return "RegesCore 1.0 35B (397B MoE)"
    
    def load_model(self, model_path: Path) -> bool:
        print(f"[regescore] Loading flagship model from {model_path}")
        return True
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return f"[regescore] Generated response to: {prompt[:50]}..."
    
    def place_on_device(self, device: str) -> bool:
        if device == "card":
            print("[regescore] Placing on GPU")
            return True
        elif device == "disk":
            print("[regescore] Streaming from SSD (16GB trunk in RAM)")
            return True
        return False
    
    @property
    def max_context_length(self) -> int:
        return 8192  # RegesCore supports longer context


# Registry of all supported families
FAMILY_REGISTRY = {
    "qwen38": Qwen38Adapter,
    "deepseek-v4": DeepSeekV4Adapter,
    "glm": GLMAdapter,
    "regescore": RegesCoreAdapter,
}


def get_adapter(family_id: str) -> Optional[EngineAdapter]:
    """Get an adapter instance for a family ID."""
    cls = FAMILY_REGISTRY.get(family_id)
    if cls:
        return cls()
    return None


def list_families() -> list[str]:
    """List all supported family IDs."""
    return list(FAMILY_REGISTRY.keys())
