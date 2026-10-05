#!/usr/bin/env python3
"""Shared file hashing, hash sharding, and Hugging Face retry helpers."""

import hashlib
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

CHUNK_BYTES = 1024 * 1024
PDF_PAGES_BUCKET = "vomebook/pdf-pages"
PDF_OCR_INPUT_BUCKET = os.environ.get("PDF_OCR_INPUT_BUCKET", "melsm")
READER_ASSETS_BUCKET = PDF_PAGES_BUCKET

T = TypeVar("T")


def hash_file(path: Path) -> tuple[str, int]:
    """Return the (sha256 hex digest, byte size) of a file."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def hash_for_key(key: str, shard_count: int) -> int:
    """Stable hash shard assignment shared by asset queue planners."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % shard_count


def weighted_shards(records: list[T], shard_count: int, *, weight: Callable[[T], int],
                     order: Callable[[T], tuple]) -> list[list[T]]:
    """Assign records to the least-loaded shard using caller weight/order functions."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[T]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for item in sorted(records, key=order):
        index = min(range(shard_count), key=lambda value: (loads[value], value))
        shards[index].append(item)
        loads[index] += weight(item)
    return shards


def hf_status_code(exc: BaseException) -> int | None:
    """Extract the HTTP status from a Hugging Face hub error, if present."""
    return getattr(getattr(exc, "response", None), "status_code", None)


def is_retryable_hf_status(status: int | None, extra: frozenset = frozenset({409, 412})) -> bool:
    """Whether an HF API failure is worth retrying (parent race, rate limit, 5xx)."""
    if status is None:
        return False
    return status in extra or status == 429 or 500 <= status < 600


def hf_retry_delay(attempt: int, cap: int = 60, max_shift: int = 5) -> int:
    """Bounded exponential backoff shared by HF publication retries."""
    return min(cap, 2 ** min(attempt, max_shift))


def batch_bucket_files_with_retry(bucket: str, additions: list[tuple[str, str]], token: str | None,
                                  *, max_attempts: int = 8) -> None:
    """Upload a bucket batch with bounded retry for HF/Xet throttling."""
    from huggingface_hub import batch_bucket_files

    for attempt in range(max_attempts):
        try:
            batch_bucket_files(bucket, add=additions, token=token)
            return
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            text = str(error).lower()
            retryable = (
                status in {408, 425, 429, 500, 502, 503, 504}
                or "too many requests" in text
                or "xet-write-token" in text
                or "timeout" in text
                or isinstance(error, (ConnectionError, TimeoutError))
            )
            if not retryable or attempt + 1 == max_attempts:
                raise
            delay = min(300, hf_retry_delay(attempt, cap=120) + random.uniform(0, 3))
            print(f"transient HF bucket upload error ({status or type(error).__name__}); "
                  f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})", flush=True)
            time.sleep(delay)


def pdf_pages_sidecar_entry(path: str) -> dict:
    """Compact search-sidecar entry for a published PDF page stream."""
    return {"s": 2, "m": "p", "p": path, "b": PDF_PAGES_BUCKET}


def merge_pdf_ocr_sidecar_entry(current: dict | None, result: dict) -> dict | None:
    """Merge OCR metadata without losing an existing Reader asset mapping."""
    entry = dict(current or {})
    if result.get("status") == "failed":
        return entry or {"s": 4, "om": "failed", "oe": result.get("error", "OCR failed")}
    if result.get("status") == "ready":
        page_manifest = result.get("page_manifest")
        # A completed page stream must take precedence over an older optimized
        # PDF route.  OCR recognition can publish after rendering, so keeping
        # the PDF in `p` would make Reader ignore the already available pages.
        if (isinstance(page_manifest, dict) and isinstance(page_manifest.get("path"), str)):
            entry.update(pdf_pages_sidecar_entry(page_manifest["path"]))
        entry.update({
            "o": result["ocr_manifest"],
            "om": result.get("classification", ""),
        })
        if entry.get("p") and str(entry["p"]).endswith("/page-manifest.json"):
            entry["b"] = PDF_PAGES_BUCKET
        elif entry.get("p"):
            entry["ob"] = PDF_PAGES_BUCKET
        else:
            entry.update({"s": 3, "m": "p", "b": PDF_PAGES_BUCKET})
        return entry
    if entry.get("s") == 3:
        return None
    for field in ("o", "om", "op", "ob"):
        entry.pop(field, None)
    return entry or None
