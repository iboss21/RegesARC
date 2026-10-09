"""Placement. Card if the hot set fits, else disk if the trunk fits, else refuse."""

from __future__ import annotations


def place(model: dict, machine: dict) -> dict:
    gpu = machine["gpu_gb"]
    ram = machine["ram_gb"]
    disk = machine["disk_free_gb"]
    if disk < model["disk_gb"]:
        return {
            "model": model["id"],
            "verdict": "refuse",
            "reason": f"need ~{model['disk_gb']} GB free, have {disk} GB",
        }
    if gpu >= model["hot_gb"] and ram >= model["trunk_gb"]:
        return {
            "model": model["id"],
            "verdict": "card",
            "reason": f"hot set {model['hot_gb']} GB fits in {gpu} GB GPU",
        }
    if ram >= model["trunk_gb"]:
        return {
            "model": model["id"],
            "verdict": "disk",
            "reason": (
                f"trunk {model['trunk_gb']} GB fits in {ram} GB RAM; "
                "experts stream from disk"
            ),
        }
    return {
        "model": model["id"],
        "verdict": "refuse",
        "reason": f"trunk needs {model['trunk_gb']} GB RAM, have {ram} GB",
    }
