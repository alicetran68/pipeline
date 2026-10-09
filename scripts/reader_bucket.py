#!/usr/bin/env python3
"""Shared paths and readers for the unified Reader bucket index."""

import json
import os
import tempfile
import gzip
import hashlib
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

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
    fs = HfFileSystem(token=token)
    with fs.open(bucket_uri(path, bucket), "rb") as stream:
        return stream.read()


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


def publish_json(path: str, payload: dict, token: str | None = None) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="reader-bucket-index-", suffix=".json")
    os.close(descriptor)
    local = Path(temporary)
    try:
        local.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
        batch_bucket_files(READER_ASSETS_BUCKET, add=[(str(local), path)], token=token)
    finally:
        local.unlink(missing_ok=True)

def publish_bytes(path: str, payload: bytes, token: str | None = None) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="reader-bucket-index-", suffix=".bin")
    os.close(descriptor)
    local = Path(temporary)
    try:
        local.write_bytes(payload)
        batch_bucket_files(READER_ASSETS_BUCKET, add=[(str(local), path)], token=token)
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
            batch_bucket_files(READER_ASSETS_BUCKET, add=additions, token=token)


def publish_catalog(sidecar: dict, token: str | None = None,
                    references: list[dict] | None = None) -> str:
    """Advance the current Reader generation after its indexes are uploaded."""
    try:
        previous = read_json(INDEX_FILES["catalog"], token)
    except FileNotFoundError:
        previous = reader_lifecycle.empty_catalog()
    catalog, generation = reader_lifecycle.catalog_generation(sidecar, previous, references)
    snapshot = json.dumps(sidecar, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    snapshot_path = f"{INDEX_PREFIX}/generations/{generation}/reader_assets.json.gz"
    entry = catalog["generations"][generation]
    compressed = gzip.compress(snapshot, mtime=0)
    entry["snapshot"] = {"bucket": READER_ASSETS_BUCKET, "path": snapshot_path,
                         "sha256": hashlib.sha256(compressed).hexdigest(), "bytes": len(compressed)}
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
