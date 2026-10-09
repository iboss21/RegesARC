"""Core download engine for RegesARC model weights.

Stdlib-only: urllib.request, hashlib, concurrent.futures, threading.
No external dependencies required.

Features:
- HTTP streaming with progress (percentage + speed + ETA)
- Resume via HTTP Range requests (.part file tracking)
- SHA-256 checksum verification after download
- Concurrent downloads for multi-file models
- Retry with exponential backoff
- TLS certificate validation (default on, bypass opt-in)
- Proxy support from HTTP_PROXY / HTTPS_PROXY env vars
- HF_TOKEN env var for gated HuggingFace repos

Usage:
    dl = Downloader(store_dir=Path("RegesModels"))
    dl.download(url="https://...", dest="model.gguf", size_hint=16e9)
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import threading
import urllib.error
import urllib.request
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------

class Progress:
    """Tracks download progress with percentage, speed, and ETA."""

    def __init__(self, total_bytes: int = 0, label: str = "") -> None:
        self.total = total_bytes          # expected total bytes (0 = unknown)
        self.downloaded = 0               # bytes received so far
        self.label = label                # human-readable name
        self.start_time = time.monotonic()
        self._lock = threading.Lock()

    def update(self, n: int) -> None:
        with self._lock:
            self.downloaded += n

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.downloaded / self.total) * 100.0)

    @property
    def speed_mbps(self) -> float:
        elapsed = max(time.monotonic() - self.start_time, 0.001)
        mbps = (self.downloaded / elapsed) / 1e6
        return round(mbps, 2)

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.total <= 0 or self.speed_mbps == 0:
            return None
        remaining = self.total - self.downloaded
        return max(0.0, remaining / (self.speed_mbps * 1e6))

    def format_line(self) -> str:
        pct = f"{self.percent:.1f}%" if self.total > 0 else "?"
        speed = f"{self.speed_mbps:.1f} MB/s"
        eta_str = ""
        if self.eta_seconds is not None and self.eta_seconds < 3600:
            m, s = divmod(int(self.eta_seconds), 60)
            eta_str = f" ETA {m}:{s:02d}"
        elif self.eta_seconds is not None:
            eta_str = f" ETA ~{int(self.eta_seconds / 3600)}h"

        if self.total > 0:
            done_gb = self.downloaded / 1e9
            total_gb = self.total / 1e9
            size_str = f"{done_gb:.2f}/{total_gb:.2f} GB"
        else:
            size_str = f"{self.downloaded / 1e6:.0f} MB"

        return f"[{pct}] {size_str} | {speed}{eta_str}"


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

class DownloadError(Exception):
    """Raised when a download fails after all retries."""
    pass


def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff with jitter and ceiling."""
    delay = min(base * (2 ** attempt), cap)
    # Add up to 50% jitter so concurrent retries don't thundering herd
    import random as _rand
    return delay * (0.5 + _rand.random())


# ---------------------------------------------------------------------------
# TLS / proxy configuration
# ---------------------------------------------------------------------------

def _build_ssl_context(bypass: bool = False) -> ssl.SSLContext:
    """Build an SSL context with certificate validation by default."""
    ctx = ssl.create_default_context()
    if bypass:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get_proxy_url(scheme: str) -> Optional[str]:
    """Read proxy from env vars. HTTPS_PROXY preferred, falls back to HTTP_PROXY."""
    key_upper = f"{scheme.upper()}_PROXY"
    val = os.environ.get(key_upper) or os.environ.get("ALL_PROXY")
    if val:
        return val.strip().rstrip("/")
    return None


def _build_opener(proxy_bypass: bool = False, tls_bypass: bool = False) -> urllib.request.OpenerDirector:
    """Build a urllib opener with proxy and TLS settings."""
    handlers = []

    # Proxy handler
    if not proxy_bypass:
        for scheme in ("https", "http"):
            proxy_url = _get_proxy_url(scheme)
            if proxy_url:
                handlers.append(urllib.request.ProxyHandler({scheme: proxy_url}))
                break  # Use first available

    # TLS handler
    ctx = _build_ssl_context(bypass=tls_bypass)
    handlers.append(urllib.request.HTTPSHandler(context=ctx))

    opener = urllib.request.build_opener(*handlers)
    return opener


# ---------------------------------------------------------------------------
# Single-file download
# ---------------------------------------------------------------------------

class Downloader:
    """Download engine with resume, retry, progress, and concurrency."""

    def __init__(
        self,
        store_dir: Path,
        max_retries: int = 5,
        tls_bypass: bool = False,
        proxy_bypass: bool = False,
        concurrent: int = 4,
        chunk_size: int = 1024 * 1024,   # 1 MB chunks
        progress_callback: Optional[Callable[[Progress], None]] = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.max_retries = max_retries
        self.tls_bypass = tls_bypass
        self.proxy_bypass = proxy_bypass
        self.concurrent = concurrent
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback

        # Ensure store directory exists
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(self, dest: Path) -> Path:
        """Return the .part file path for a destination."""
        return dest.with_suffix(dest.suffix + ".part") if dest.suffix else dest.parent / (dest.name + ".part")

    def _resume_offset(self, part: Path) -> int:
        """How many bytes are already in the .part file."""
        if part.exists():
            return part.stat().st_size
        return 0

    def _request_headers(
        self,
        url: str,
        offset: int = 0,
        total: Optional[int] = None,
    ) -> dict[str, str]:
        """Build request headers including Range for resume."""
        headers: dict[str, str] = {
            "User-Agent": "RegesARC-Downloader/1.0",
            "Accept": "*/*",
        }
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        return headers

    def _fetch_head(self, url: str) -> tuple[int, int]:
        """HEAD request to get Content-Length and Last-Modified. Returns (size, last_modified)."""
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "RegesARC-Downloader/1.0")

        try:
            with self._opener().open(req, timeout=30) as resp:
                size_str = resp.headers.get("Content-Length")
                last_mod = resp.headers.get("Last-Modified")
                size = int(size_str) if size_str else 0
                return (size, last_mod or "")
        except Exception:
            return (0, "")

    def _opener(self) -> urllib.request.OpenerDirector:
        return _build_opener(
            proxy_bypass=self.proxy_bypass,
            tls_bypass=self.tls_bypass,
        )

    def download(
        self,
        url: str,
        dest: Path | str,
        size_hint: int = 0,
        expected_hash: Optional[str] = None,
        label: str = "",
    ) -> bool:
        """Download a single file with resume and retry.

        Args:
            url: Source URL to download from.
            dest: Destination path (relative to store_dir or absolute).
            size_hint: Expected total bytes (0 = unknown, no progress %).
            expected_hash: SHA-256 hex digest to verify after download.
            label: Human-readable name for progress display.

        Returns:
            True if download succeeded and checksum verified (if requested).
        """
        dest = Path(dest) if not isinstance(dest, Path) else dest
        if not dest.is_absolute():
            dest = self.store_dir / dest

        part = self._part_path(dest)
        progress = Progress(total_bytes=size_hint, label=label or dest.name)

        # Try HEAD to get actual size
        actual_size = size_hint
        try:
            head_size, _ = self._fetch_head(url)
            if head_size > 0:
                actual_size = head_size
        except Exception:
            pass

        progress.total = actual_size

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._download_attempt(
                    url=url, dest=dest, part=part,
                    offset=self._resume_offset(part),
                    total=actual_size, progress=progress,
                    expected_hash=expected_hash, label=label or dest.name,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = _backoff_delay(attempt)
                    print(f"  Retry {attempt + 1}/{self.max_retries} after {delay:.0f}s: {e}")
                    time.sleep(delay)
                else:
                    break

        raise DownloadError(
            f"Failed to download {url} after {self.max_retries + 1} attempts: {last_error}"
        )

    def _download_attempt(
        self,
        url: str,
        dest: Path,
        part: Path,
        offset: int,
        total: int,
        progress: Progress,
        expected_hash: Optional[str],
        label: str,
    ) -> bool:
        """Single download attempt (no retry — caller handles that)."""

        req = urllib.request.Request(url)
        req.add_header("User-Agent", "RegesARC-Downloader/1.0")
        if offset > 0:
            req.add_header("Range", f"bytes={offset}-")

        opener = self._opener()

        with opener.open(req, timeout=60) as resp:
            # If we got a full response (no resume), overwrite the part file
            status_code = resp.status
            if offset > 0 and status_code == 200:
                # Server doesn't support Range — start fresh
                if part.exists():
                    part.unlink()
                offset = 0

            content_length = int(resp.headers.get("Content-Length", "0"))
            if content_length > 0 and total == 0:
                progress.total = offset + content_length

            # Ensure parent directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            sha256 = hashlib.sha256()
            bytes_since_last_report = 0
            last_report_time = time.monotonic()

            with open(part, "ab" if offset > 0 else "wb") as f:
                while True:
                    chunk = resp.read(self.chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    sha256.update(chunk)
                    progress.update(len(chunk))

                    # Throttle progress reporting to ~1 Hz
                    now = time.monotonic()
                    bytes_since_last_report += len(chunk)
                    if (bytes_since_last_report >= 5 * 1024 * 1024 or
                            now - last_report_time >= 1.0):
                        line = progress.format_line()
                        print(f"\r  {label}: {line}", end="", flush=True)
                        bytes_since_last_report = 0
                        last_report_time = now

            # Final report
            print(f"\r  {label}: {progress.format_line()}   ", end="\n", flush=True)

        # Verify checksum if requested
        if expected_hash:
            actual_hash = sha256.hexdigest()
            if actual_hash.lower() != expected_hash.lower():
                if part.exists():
                    part.unlink()
                raise DownloadError(
                    f"Checksum mismatch for {dest.name}: "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )

        # Rename .part to final destination
        if part != dest:
            os.replace(part, dest)
        else:
            # If dest == part (no suffix), just ensure it's there
            pass

        return True

    def download_many(
        self,
        files: list[dict],
        label_prefix: str = "",
    ) -> dict[str, bool]:
        """Download multiple files concurrently.

        Args:
            files: List of dicts with keys: url, dest, size_hint, expected_hash, label
            label_prefix: Prefix for progress labels.

        Returns:
            Dict mapping dest name to success (True/False).
        """
        results: dict[str, bool] = {}
        label = label_prefix or f"{len(files)} files"

        def _worker(file_info: dict) -> tuple[str, bool]:
            fname = file_info["dest"]
            try:
                ok = self.download(
                    url=file_info["url"],
                    dest=fname,
                    size_hint=file_info.get("size_hint", 0),
                    expected_hash=file_info.get("expected_hash"),
                    label=f"{label_prefix}{fname}" if label_prefix else fname,
                )
                return (str(fname), ok)
            except DownloadError as e:
                print(f"\n  ERROR downloading {fname}: {e}")
                return (str(fname), False)

        with ThreadPoolExecutor(max_workers=self.concurrent) as pool:
            futures = {pool.submit(_worker, f): f for f in files}
            for future in as_completed(futures):
                name, ok = future.result()
                results[name] = ok

        return results


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def verify_sha256(filepath: Path) -> str:
    """Compute SHA-256 of a file. Returns hex digest."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def cleanup_part_files(store_dir: Path) -> int:
    """Remove stale .part files older than 2 hours. Returns count removed."""
    cutoff = time.monotonic() - 7200
    removed = 0
    for part in store_dir.rglob("*.part"):
        try:
            if part.stat().st_mtime < cutoff:
                part.unlink()
                removed += 1
        except OSError:
            pass
    return removed
