#!/usr/bin/env python3
"""Shared paths and readers for the unified Reader bucket index."""

import json
import copy
import os
import tempfile
import gzip
import hashlib
import random
import time
from urllib.parse import quote
from pathlib import Path

import httpx
from huggingface_hub import batch_bucket_files
from huggingface_hub.utils import get_session, hf_raise_for_status
from huggingface_hub.errors import HfHubHTTPError

try:
    from .reader_assets import READER_ASSETS_BUCKET
    from . import reader_lifecycle
except ImportError:
    from reader_assets import READER_ASSETS_BUCKET
    import reader_lifecycle


INDEX_PREFIX = "reader-index"
INDEX_FILES = {
    "manifest": f"{INDEX_PREFIX}/manifest.json",
    "sidecar": f"{INDEX_PREFIX}/reader_assets.json.gz",
    "pdf": f"{INDEX_PREFIX}/pdf_manifest.json",
    "ocr": f"{INDEX_PREFIX}/pdf_ocr_manifest.json",
    "lifecycle": f"{INDEX_PREFIX}/reader_lifecycle.json",
    "catalog": f"{INDEX_PREFIX}/reader_catalog.json",
}


def index_path(name: str) -> str:
    return f"{INDEX_PREFIX}/{name}"


def bucket_uri(path: str, bucket: str = READER_ASSETS_BUCKET) -> str:
    return f"hf://buckets/{bucket}/{path}"


def read_bytes(path: str, token: str | None = None, bucket: str = READER_ASSETS_BUCKET) -> bytes:
    """Read one immutable bucket object without a metadata/tree API call."""
    session = get_session()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/buckets/{bucket}/resolve/{quote(path, safe='/')}"
    # Resolver quotas are account-wide and a PDF OCR wave can exhaust the
    # five-minute window while this process is waiting in the publication
    # queue. Keep retrying through the window instead of converting a
    # temporary 429 into a false missing-root/publication failure.
    for attempt in range(12):
        try:
            response = session.get(url, headers=headers, follow_redirects=True, timeout=120)
        except httpx.TransportError:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))
            continue
        if response.status_code == 404:
            raise FileNotFoundError(path)
        if response.status_code in {408, 425, 429, 500, 502, 503, 504} and attempt < 11:
            retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
            try:
                delay = min(300, max(1, int(float(retry_after)))) if retry_after else min(120, 2 ** attempt)
            except ValueError:
                delay = min(120, 2 ** attempt)
            time.sleep(delay + random.uniform(0, 1.5))
            continue
        hf_raise_for_status(response)
        return response.content
    raise RuntimeError(f"bucket read retry limit exceeded: {bucket}:{path}")


def read_json(path: str, token: str | None = None, bucket: str = READER_ASSETS_BUCKET) -> dict:
    value = json.loads(read_bytes(path, token, bucket).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid Reader bucket JSON: {path}")
    return value


def materialize(path: str, token: str | None = None, suffix: str = "",
                bucket: str = READER_ASSETS_BUCKET) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="reader-bucket-", suffix=suffix)
    os.close(descriptor)
    target = Path(name)
    target.write_bytes(read_bytes(path, token, bucket))
    return target


def stage_index(root: Path, name: str, payload: bytes | str) -> Path:
    path = root / INDEX_FILES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    path.write_bytes(payload)
    return path


def _publish_additions(additions: list[tuple[str, str]], token: str | None = None,
                       max_attempts: int = 12) -> None:
    """Publish bucket metadata while absorbing account-wide HF/Xet throttling."""
    for attempt in range(max_attempts):
        try:
            batch_bucket_files(READER_ASSETS_BUCKET, add=additions, token=token)
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
                or isinstance(error, (ConnectionError, TimeoutError, httpx.TransportError))
            )
            if not retryable or attempt + 1 == max_attempts:
                raise
            headers = getattr(response, "headers", {}) or {}
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            try:
                delay = max(0, min(300, int(float(retry_after)))) if retry_after else min(120, 2 ** attempt)
            except (TypeError, ValueError):
                delay = min(120, 2 ** attempt)
            delay += random.uniform(0, 3)
            print(f"transient Reader index upload error ({status or type(error).__name__}); "
                  f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})", flush=True)
            time.sleep(delay)
    raise RuntimeError("Reader index upload retry limit reached")


def publish_json(path: str, payload: dict, token: str | None = None) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="reader-bucket-index-", suffix=".json")
    os.close(descriptor)
    local = Path(temporary)
    try:
        local.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
        _publish_additions([(str(local), path)], token)
    finally:
        local.unlink(missing_ok=True)

def publish_bytes(path: str, payload: bytes, token: str | None = None) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="reader-bucket-index-", suffix=".bin")
    os.close(descriptor)
    local = Path(temporary)
    try:
        local.write_bytes(payload)
        _publish_additions([(str(local), path)], token)
    finally:
        local.unlink(missing_ok=True)


def publish_indexes(payloads: dict[str, bytes], token: str | None = None) -> None:
    """Upload related indexes together to the canonical current bucket."""
    with tempfile.TemporaryDirectory(prefix="reader-indexes-") as root:
        additions = []
        for number, (path, payload) in enumerate(sorted(payloads.items())):
            if not path.startswith(INDEX_PREFIX + "/") or ".." in path.split("/"):
                raise ValueError("publication must stay inside reader-index")
            local = Path(root) / str(number)
            local.write_bytes(payload)
            additions.append((str(local), path))
        if additions:
            _publish_additions(additions, token)


def publish_json_batch(payloads: dict[str, dict], token: str | None = None) -> None:
    """Upload small protection/handoff records together without changing schemas."""
    publish_indexes({path: (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
                     for path, value in payloads.items()}, token)


def publish_catalog(sidecar: dict, token: str | None = None,
                    references: list[dict] | None = None) -> str:
    """Advance the current Reader generation after its indexes are uploaded."""
    try:
        previous = read_json(INDEX_FILES["catalog"], token)
    except FileNotFoundError:
        previous = reader_lifecycle.empty_catalog()
    catalog, generation = reader_lifecycle.catalog_generation(sidecar, copy.deepcopy(previous), references)
    snapshot = json.dumps(sidecar, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    snapshot_path = f"{INDEX_PREFIX}/generations/{generation}/reader_assets.json.gz"
    entry = catalog["generations"][generation]
    compressed = gzip.compress(snapshot, mtime=0)
    entry["snapshot"] = {"bucket": READER_ASSETS_BUCKET, "path": snapshot_path,
                         "sha256": hashlib.sha256(compressed).hexdigest(), "bytes": len(compressed)}
    if catalog == previous:
        try:
            if read_bytes(snapshot_path, token) == compressed:
                return generation
        except FileNotFoundError:
            pass
    publish_indexes({snapshot_path: compressed}, token)
    publish_json(INDEX_FILES["catalog"], catalog, token)
    return generation


def acknowledge_catalog(surface: str, generation: str, sidecar_sha256: str,
                        token: str | None = None) -> str:
    if surface not in {"hf", "pages"}:
        raise ValueError("catalog surface must be hf or pages")
    catalog = read_json(INDEX_FILES["catalog"], token)
    current = catalog.get("current") or {}
    if (not generation or current.get("generation") != generation
            or generation not in catalog.get("generations", {})):
        raise ValueError("Reader catalog has no current generation")
    entry = dict(catalog["generations"][generation])
    if entry.get("sidecar_sha256") != sidecar_sha256:
        raise ValueError("Reader consumer acknowledged a different sidecar")
    acks = dict(entry.get("acks") or {})
    acks[surface] = True
    entry["acks"] = acks
    catalog["generations"][generation] = entry
    for old in catalog["generations"].values():
        if old.get("status") == "superseded":
            receipts = dict(old.get("replacement_acks") or {})
            receipts[surface] = True
            old["replacement_acks"] = receipts
    publish_json(INDEX_FILES["catalog"], catalog, token)
    return generation
