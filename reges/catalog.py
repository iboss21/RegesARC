"""RegesARC model catalog.

Weights are not in this repo. A row is a verdict target: how big the trunk
is, how big the hot set is, which placement can run it. Numbers are the
public operating points of the upstream engines, not a benchmark claim.
"""

from __future__ import annotations

# disk_gb: converted container, order of magnitude, for the scan verdict.
# trunk_gb: resident dense path the disk engine keeps in RAM.
# hot_gb: what the card path wants on the GPU for a useful hit rate.
CATALOG = (
    {
        "id": "regescore-1.0-35",
        "label": "RegesCore 1.0 35B",
        "params": "397B MoE — Flagship",
        "disk_gb": 69,
        "trunk_gb": 16,
        "hot_gb": 24,
        "family": "regescore",
    },
    {
        "id": "qwen3.8-flash-next",
        "label": "Qwen3.8-Flash-Next",
        "params": "125B MoE",
        "disk_gb": 80,
        "trunk_gb": 16,
        "hot_gb": 12,
        "family": "qwen38",
    },
    {
        "id": "qwen3.6",
        "label": "Qwen3.6-35B-A3B",
        "params": "35B MoE",
        "disk_gb": 22,
        "trunk_gb": 8,
        "hot_gb": 8,
        "family": "qwen36",
    },
    {
        "id": "deepseek-v4-flash",
        "label": "DeepSeek V4 Flash",
        "params": "284B MoE",
        "disk_gb": 160,
        "trunk_gb": 16,
        "hot_gb": 24,
        "family": "deepseek-v4",
    },
    {
        "id": "deepseek-v4.1-flash",
        "label": "DeepSeek V4.1 Flash",
        "params": "552B MoE",
        "disk_gb": 510,
        "trunk_gb": 24,
        "hot_gb": 48,
        "family": "deepseek-v4",
    },
    {
        "id": "glm-5.3-flash",
        "label": "GLM-5.3-Flash",
        "params": "321B MoE",
        "disk_gb": 180,
        "trunk_gb": 16,
        "hot_gb": 24,
        "family": "glm",
    },
    {
        "id": "glm-5.2",
        "label": "GLM-5.2",
        "params": "744B MoE",
        "disk_gb": 372,
        "trunk_gb": 12,
        "hot_gb": 48,
        "family": "glm",
    },
    {
        "id": "inkling",
        "label": "Inkling",
        "params": "975B MoE",
        "disk_gb": 500,
        "trunk_gb": 24,
        "hot_gb": 48,
        "family": "inkling",
    },
    {
        "id": "kimi-k3",
        "label": "Kimi K3",
        "params": "2.8T MoE",
        "disk_gb": 1400,
        "trunk_gb": 32,
        "hot_gb": 80,
        "family": "kimi",
    },
    {
        "id": "olmoe",
        "label": "OLMoE-7B-7B",
        "params": "7B MoE",
        "disk_gb": 6,
        "trunk_gb": 4,
        "hot_gb": 6,
        "family": "olmoe",
    },
    {
        "id": "laya",
        "label": "Laya",
        "params": "MoE",
        "disk_gb": 20,
        "trunk_gb": 8,
        "hot_gb": 10,
        "family": "laya",
    },
)


def by_id(model_id: str) -> dict:
    for row in CATALOG:
        if row["id"] == model_id:
            return row
    raise KeyError(model_id)
