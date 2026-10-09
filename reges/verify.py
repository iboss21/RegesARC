"""Post-install verification for RegesARC models.

Checks:
- All expected files exist (from manifest.json)
- Checksums match manifest (SHA-256)
- GGUF header validation if applicable (.gguf files)
- Reports any missing or corrupted files

Usage:
    python -m reges.verify                  # Verify all installed models
    python -m reges.verify model-id         # Verify specific model
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Optional


def verify_gguf_header(filepath: Path) -> tuple[bool, str]:
    """Validate GGUF file header.

    GGUF format:
    - Magic bytes: b'GGUF' (4 bytes)
    - Version: uint32 (usually 3 for current spec)
    - Tensor count: uint64
    - Metadata tensor count: uint64

    Returns (is_valid, message).
    """
    if not filepath.exists():
        return False, f"File does not exist: {filepath}"

    try:
        with open(filepath, "rb") as f:
            # Read magic bytes
            magic = f.read(4)
            if magic != b'GGUF':
                return False, f"Invalid GGUF magic: expected b'GGUF', got {magic!r}"

            # Read version
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                return False, "File too short for GGUF header"

            version = struct.unpack('<I', version_bytes)[0]
            if version not in (1, 2, 3):
                return False, f"Unsupported GGUF version: {version}"

            # Read tensor count
            tensor_count_bytes = f.read(8)
            if len(tensor_count_bytes) < 8:
                return False, "File too short for tensor count"

            tensor_count = struct.unpack('<Q', tensor_count_bytes)[0]

            # Read metadata tensor count
            meta_count_bytes = f.read(8)
            if len(meta_count_bytes) < 8:
                return False, "File too short for metadata count"

            meta_count = struct.unpack('<Q', meta_count_bytes)[0]

        return True, f"Valid GGUF v{version}: {tensor_count} tensors, {meta_count} metadata entries"

    except Exception as e:
        return False, f"Error reading GGUF header: {e}"


def verify_model(model_dir: Path) -> tuple[bool, list[str]]:
    """Verify a single model installation.

    Args:
        model_dir: Path to the model directory in RegesModels/.

    Returns:
        (success, issues) where success is True if all checks pass,
        and issues is a list of problem descriptions.
    """
    issues: list[str] = []

    # Check manifest.json exists
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return False, ["No manifest.json found in model directory"]

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return False, [f"Failed to read manifest.json: {e}"]

    # Check each file in manifest
    files = manifest.get("files", [])
    if not files:
        # No files expected — check if directory is empty or has only manifest
        other_files = list(model_dir.rglob("*"))
        non_manifest = [f for f in other_files if f != manifest_path]
        if non_manifest:
            issues.append(f"Directory contains {len(non_manifest)} unexpected files")
        return (not bool(issues)), issues

    for file_info in files:
        rel_path = file_info.get("path", "")
        expected_hash = file_info.get("sha256")
        status = file_info.get("status", "unknown")

        # Check file exists
        full_path = model_dir / rel_path
        if not full_path.exists():
            issues.append(f"Missing file: {rel_path}")
            continue

        # If download failed, note it
        if status == "failed":
            issues.append(f"Download failed for: {rel_path}")
            continue

        # Verify checksum if expected and available
        if expected_hash:
            actual_hash = None
            try:
                import hashlib
                sha256 = hashlib.sha256()
                with open(full_path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        sha256.update(chunk)
                actual_hash = sha256.hexdigest()

                if actual_hash.lower() != expected_hash.lower():
                    issues.append(
                        f"Checksum mismatch for {rel_path}: "
                        f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                    )
            except Exception as e:
                issues.append(f"Checksum verification failed for {rel_path}: {e}")

        # Validate GGUF header if applicable
        if str(rel_path).endswith(".gguf"):
            gguf_valid, gguf_msg = verify_gguf_header(full_path)
            if not gguf_valid:
                issues.append(f"Invalid GGUF header for {rel_path}: {gguf_msg}")

    return (not bool(issues)), issues


def verify_all(store_dir: Path) -> tuple[int, int, list[tuple[str, bool, list[str]]]]:
    """Verify all installed models in the store directory.

    Args:
        store_dir: Path to RegesModels/.

    Returns:
        (total, passed, results) where results is a list of
        (model_id, success, issues) tuples.
    """
    if not store_dir.exists():
        return 0, 0, []

    # Find all model directories (those with manifest.json)
    manifest_files = list(store_dir.rglob("manifest.json"))
    total = len(manifest_files)
    passed = 0
    results: list[tuple[str, bool, list[str]]] = []

    for manifest_path in sorted(manifest_files):
        model_id = manifest_path.parent.name
        success, issues = verify_model(manifest_path.parent)

        if success:
            passed += 1

        results.append((model_id, success, issues))

    return total, passed, results


def main() -> None:
    """CLI entry point for verification."""
    from reges.install import store_dir as get_store_dir

    store = get_store_dir()

    if not store.exists():
        print(f"Model store does not exist: {store}")
        return

    # If specific model requested, verify just that one
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
        model_dir = store / model_id

        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            return

        success, issues = verify_model(model_dir)

        if success:
            print(f"✓ {model_id} verification passed.")
        else:
            print(f"✗ {model_id} verification failed:")
            for issue in issues:
                print(f"  - {issue}")
        return

    # Verify all models
    print("\nRegesARC Model Verification")
    print("=" * 60)

    total, passed, results = verify_all(store)

    if total == 0:
        print("No installed models found.")
        return

    for model_id, success, issues in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"\n{status} {model_id}")
        if not success:
            for issue in issues:
                print(f"  - {issue}")

    print("\n" + "=" * 60)
    print(f"Verification complete: {passed}/{total} models passed.")


if __name__ == "__main__":
    main()
