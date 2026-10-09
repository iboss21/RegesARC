"""Model installer for RegesARC.

Downloads selected models to RegesModels/ using the new downloader and registry.
Features:
- Interactive model selection with hardware-aware filtering
- Download progress display (terminal-based, cross-platform)
- Post-download verification (checksums, file structure validation)
- Model directory structure creation in RegesModels/
- Manifest.json generation for each installed model
- Support for partial installs and resume from interruption

Usage:
    python -m reges.install          # Interactive mode
    python -m reges.install --all    # Install all available models
    python -m reges.install qwen3.6  # Install specific model by ID
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports when run directly
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reges.catalog import CATALOG, by_id  # noqa: E402
from reges.scan import scan  # noqa: E402
from reges.place import place  # noqa: E402
from reges.registry import get as registry_get, build_manifest_dict  # noqa: E402
from reges.downloader import Downloader, DownloadError, verify_sha256  # noqa: E402


def store_dir() -> Path:
    """Where model weights live. Beside the checkout, never in git."""
    return Path(__file__).resolve().parent.parent / "RegesModels"


def list_models() -> None:
    """Show catalog with verdicts for current machine."""
    print("\nRegesARC Model Catalog")
    print("=" * 60)

    machine = scan(store_dir())
    print(f"RAM: {machine['ram_gb']} GB | GPU: {machine['gpu_gb']} GB | Disk Free: {machine['disk_free_gb']} GB\n")

    for row in CATALOG:
        verdict = place(row, machine)
        mark = {"card": "✓ CARD", "disk": "✓ DISK", "refuse": "✗ NO"}[verdict["verdict"]]
        print(f"  {mark:10} {row['label']:24} {row['params']}")

    print("\n" + "=" * 60)


def select_models(args: list[str]) -> list[str]:
    """Interactive model selection. Returns list of model IDs."""
    machine = scan(store_dir())

    available = []
    for row in CATALOG:
        verdict = place(row, machine)
        if verdict["verdict"] != "refuse":
            available.append(row)

    # If args provided, use those as explicit selections
    if args:
        selected_ids = []
        for arg in args:
            # Try to match by ID or partial label
            matched = False
            for row in available:
                if row["id"] == arg or arg.lower() in row["label"].lower():
                    selected_ids.append(row["id"])
                    matched = True
                    break
            if not matched:
                print(f"Warning: '{arg}' not found in available models. Skipping.")
        return selected_ids

    # Interactive mode
    print("\nAvailable Models (based on your hardware):")
    print("-" * 60)

    for i, row in enumerate(available, 1):
        verdict = place(row, machine)
        mark = {"card": "CARD", "disk": "DISK"}[verdict["verdict"]]
        print(f"  {i:2}. {row['label']:24} [{mark}]")

    print("\nEnter model numbers to install (comma-separated, or 'all' for all available):")
    choice = input("> ").strip().lower()

    if choice == "all":
        return [row["id"] for row in available]

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = []
        for idx in indices:
            if 0 <= idx < len(available):
                selected.append(available[idx]["id"])
            else:
                print(f"Invalid index {idx + 1}. Skipping.")
        return selected
    except ValueError:
        print("Invalid input. Returning empty selection.")
        return []


def create_model_structure(model_id: str, store: Path) -> dict[str, Path]:
    """Create the directory structure for a model in RegesModels/.

    Returns dict mapping logical paths to actual Paths.
    """
    model_dir = store / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories based on placement hint from registry
    try:
        entry = registry_get(model_id)
        hint = entry.placement_hint
    except KeyError:
        hint = "auto"

    paths: dict[str, Path] = {"model_dir": model_dir}

    if hint == "card":
        # Card path: hot experts on GPU, cold on disk
        (model_dir / "hot_experts").mkdir(exist_ok=True)
        (model_dir / "cold_experts").mkdir(exist_ok=True)
        paths["hot"] = model_dir / "hot_experts"
        paths["cold"] = model_dir / "cold_experts"
    elif hint == "disk":
        # Disk path: trunk in RAM, routed experts on SSD
        (model_dir / "trunk").mkdir(exist_ok=True)
        (model_dir / "routed_experts").mkdir(exist_ok=True)
        paths["trunk"] = model_dir / "trunk"
        paths["experts"] = model_dir / "routed_experts"

    return paths


def download_model(model_id: str, store: Path, downloader: Downloader) -> bool:
    """Download a single model to the store directory.

    Uses registry for URLs and manifest, downloader for actual transfer.
    Supports resume from .part files.
    """
    try:
        entry = registry_get(model_id)
    except KeyError:
        print(f"Unknown model: {model_id}")
        return False

    # Check if already partially or fully installed
    model_dir = store / model_id
    existing_files = list(model_dir.rglob("*")) if model_dir.exists() else []
    part_files = [f for f in existing_files if f.name.endswith(".part")]

    if part_files:
        print(f"Resuming partial download from {len(part_files)} .part file(s)...")
    elif existing_files and not any(f.name == "manifest.json" for f in existing_files):
        print(f"Found {len(existing_files)} existing files. Will overwrite.")

    # Create directory structure
    paths = create_model_structure(model_id, store)

    # Build download tasks from registry manifest
    download_tasks = []
    for file_entry in entry.manifest:
        if not file_entry.url:
            # Use primary URL as base, append filename
            base_url = entry.urls[0] if entry.urls else ""
            url = f"{base_url.rstrip('/')}/{file_entry.path}" if base_url else ""
        else:
            url = file_entry.url

        dest_path = paths["model_dir"] / file_entry.path
        download_tasks.append({
            "url": url,
            "dest": dest_path,
            "size_hint": file_entry.size_bytes,
            "expected_hash": file_entry.sha256,
            "label": file_entry.path,
        })

    if not download_tasks:
        print(f"No files to download for {model_id}. Creating placeholder structure.")
        _write_placeholder_manifest(model_id, store)
        return True

    # Download all files concurrently
    print(f"\nDownloading {len(download_tasks)} file(s) for {entry.label}...")
    results = downloader.download_many(download_tasks, label_prefix=f"[{model_id}] ")

    # Check results
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count

    if fail_count > 0:
        print(f"\nWarning: {fail_count}/{len(results)} files failed to download.")
        print("You can re-run install to resume from .part files.")

    # Write manifest.json
    _write_manifest(model_id, store, results)

    # Clean up any remaining .part files for successful downloads
    for dest_path in [t["dest"] for t in download_tasks if results.get(str(t["dest"]), False)]:
        part = dest_path.with_suffix(dest_path.suffix + ".part") if dest_path.suffix else dest_path.parent / (dest_path.name + ".part")
        if part.exists():
            try:
                part.unlink()
            except OSError:
                pass

    print(f"\n✓ Model installation {'complete' if fail_count == 0 else 'partial'}: {model_id}")
    print(f"  Location: {paths['model_dir']}")
    return success_count > 0


def _write_placeholder_manifest(model_id: str, store: Path) -> None:
    """Write a placeholder manifest for models with no registry entry."""
    try:
        catalog_entry = by_id(model_id)
    except KeyError:
        catalog_entry = {"id": model_id, "label": model_id, "params": "?", "family": "?"}

    machine = scan(store)
    verdict = place(catalog_entry, machine)

    manifest = {
        "id": model_id,
        "label": catalog_entry.get("label", model_id),
        "params": catalog_entry.get("params", "?"),
        "family": catalog_entry.get("family", "?"),
        "placement": verdict["verdict"],
        "status": "placeholder",
        "files": [],
    }

    (store / model_id / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _write_manifest(model_id: str, store: Path, download_results: dict[str, bool]) -> None:
    """Write manifest.json with download results and checksums."""
    try:
        registry_entry = registry_get(model_id)
        manifest_data = build_manifest_dict(model_id)
    except KeyError:
        # Fallback to catalog
        try:
            catalog_entry = by_id(model_id)
        except KeyError:
            catalog_entry = {"id": model_id, "label": model_id, "params": "?", "family": "?"}

        machine = scan(store)
        verdict = place(catalog_entry, machine)

        manifest_data = {
            "id": model_id,
            "label": catalog_entry.get("label", model_id),
            "family": catalog_entry.get("family", "?"),
            "params": catalog_entry.get("params", "?"),
            "disk_gb": catalog_entry.get("disk_gb", 0),
            "placement_hint": verdict["verdict"],
            "gated": False,
            "notes": "",
            "files": [],
        }

    # Update file statuses based on download results
    for file_info in manifest_data["files"]:
        dest_str = str(store / model_id / file_info["path"])
        if dest_str in download_results:
            file_info["status"] = "ok" if download_results[dest_str] else "failed"

            # If successful, compute actual checksum
            if download_results[dest_str]:
                actual_path = store / model_id / file_info["path"]
                if actual_path.exists():
                    try:
                        file_info["actual_sha256"] = verify_sha256(actual_path)
                    except Exception as e:
                        file_info["checksum_error"] = str(e)

    # Add metadata
    manifest_data["download_timestamp"] = __import__("datetime").datetime.now().isoformat()
    manifest_data["installer_version"] = "1.0"

    # Write to disk
    manifest_path = store / model_id / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )


def verify_installation(model_id: str, store: Path) -> bool:
    """Verify a model installation is complete and valid.

    Returns True if all files present and checksums match.
    """
    from reges.verify import verify_model  # Local import to avoid circular deps

    print(f"\nVerifying {model_id}...")
    success, issues = verify_model(store / model_id)

    if success:
        print(f"✓ {model_id} verification passed.")
        return True
    else:
        print(f"✗ {model_id} verification failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False


def install_selected(args: list[str]) -> None:
    """Main install flow: select models, download them."""
    store = store_dir()

    if not store.exists():
        store.mkdir(parents=True, exist_ok=True)
        print(f"Created model store: {store}")

    # Show catalog
    list_models()

    # Select models
    selected = select_models(args)

    if not selected:
        print("\nNo models selected. Exiting.")
        return

    print(f"\nInstalling {len(selected)} model(s)...")
    print("=" * 60)

    # Create downloader with appropriate settings
    downloader = Downloader(
        store_dir=store,
        max_retries=5,
        concurrent=4,
    )

    success_count = 0
    for i, model_id in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Installing {model_id}...")
        if download_model(model_id, store, downloader):
            # Verify after download
            if verify_installation(model_id, store):
                success_count += 1

    print("\n" + "=" * 60)
    print(f"Installation complete: {success_count}/{len(selected)} models ready.")
    print(f"\nNext steps:")
    print("  1. Run engine to load models (requires compiled binary)")
    print("  2. Models are in:", store)


if __name__ == "__main__":
    # Parse command-line args: python -m reges.install [model_id ...]
    install_selected(sys.argv[1:])
