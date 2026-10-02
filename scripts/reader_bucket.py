#!/usr/bin/env python3
"""Shared paths and readers for the unified Reader bucket index."""

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

try:
    from .reader_assets import READER_ASSETS_BUCKET
except ImportError:
    from reader_assets import READER_ASSETS_BUCKET


INDEX_PREFIX = "reader-index"
INDEX_FILES = {
    "manifest": f"{INDEX_PREFIX}/manifest.json",
    "sidecar": f"{INDEX_PREFIX}/reader_assets.json.gz",
    "pdf": f"{INDEX_PREFIX}/pdf_manifest.json",
    "ocr": f"{INDEX_PREFIX}/pdf_ocr_manifest.json",
    "lifecycle": f"{INDEX_PREFIX}/reader_lifecycle.json",
}


def index_path(name: str) -> str:
    return f"{INDEX_PREFIX}/{name}"


def bucket_uri(path: str) -> str:
    return f"hf://buckets/{READER_ASSETS_BUCKET}/{path}"


def read_bytes(path: str, token: str | None = None) -> bytes:
    fs = HfFileSystem(token=token)
    with fs.open(bucket_uri(path), "rb") as stream:
        return stream.read()


def read_json(path: str, token: str | None = None) -> dict:
    value = json.loads(read_bytes(path, token).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid Reader bucket JSON: {path}")
    return value


def materialize(path: str, token: str | None = None, suffix: str = "") -> Path:
    descriptor, name = tempfile.mkstemp(prefix="reader-bucket-", suffix=suffix)
    os.close(descriptor)
    target = Path(name)
    target.write_bytes(read_bytes(path, token))
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
