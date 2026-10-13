#!/usr/bin/env python3
"""Per-book OCR checkpoints with a small summary index.

The former single ``reader-index/pdf_ocr_progress.json`` held every retained
page record for every book. It reached 2.27 GB, and each central OCR publish
loaded, merged and re-serialized the whole file, which exhausted the hosted
runner. A publish now reads and rewrites only the books it updates.

Shards keep the legacy registry schema (``version``/``files``) so the Reader GC
graph follows the same page references. Shards are rewritten only by the
serialized central publisher; readers validate the shard's own key/generation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from huggingface_hub.utils import get_session

try:
    from . import shared
    from .reader_bucket import index_path, publish_indexes, read_bytes
except ImportError:
    import shared
    from reader_bucket import index_path, publish_indexes, read_bytes

LEGACY = "pdf_ocr_progress.json"
INDEX = "pdf_ocr_progress/index.json"
INDEX_KIND = "pdf-ocr-progress-index"
SHARD_KIND = "pdf-ocr-progress-book"
MIGRATION_BATCH = 64


def shard_name(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"pdf_ocr_progress/books/{digest[:2]}/{digest}.json.gz"


def encode_json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_shard(key: str, entry: dict) -> bytes:
    # mtime=0 keeps a rewrite of unchanged content byte-identical.
    return gzip.compress(encode_json({"version": 1, "kind": SHARD_KIND, "key": key,
                                      "files": {key: entry}}), mtime=0)


def validate_entry(key: str, entry) -> dict:
    if (not isinstance(entry, dict) or not isinstance(entry.get("generation"), str)
            or not isinstance(entry.get("pages"), dict)):
        raise ValueError(f"invalid OCR progress entry: {key}")
    for label, page in entry["pages"].items():
        if not isinstance(page, dict) or str(page.get("p")) != label:
            raise ValueError(f"invalid OCR progress page: {key}:{label}")
    return entry


def summary(entry: dict) -> dict:
    return {"generation": entry["generation"], "pages": sorted(int(label) for label in entry["pages"])}


def plain(raw: bytes) -> bytes:
    # Bucket reads may arrive transport-decoded; compare/parse the JSON content.
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def decode_shard(key: str, raw: bytes) -> dict:
    payload = json.loads(plain(raw))
    if (payload.get("version") != 1 or payload.get("kind") != SHARD_KIND or payload.get("key") != key
            or not isinstance(payload.get("files"), dict) or set(payload["files"]) != {key}):
        raise ValueError(f"invalid OCR progress shard: {key}")
    return validate_entry(key, payload["files"][key])


def validate_index(payload: dict) -> dict:
    if (payload.get("version") != 1 or payload.get("kind") != INDEX_KIND
            or not isinstance(payload.get("books"), dict)):
        raise ValueError("invalid OCR progress index")
    for key, item in payload["books"].items():
        if (not isinstance(item, dict) or not isinstance(item.get("generation"), str)
                or not isinstance(item.get("pages"), list)):
            raise ValueError(f"invalid OCR progress index entry: {key}")
    return payload


class ProgressStore:
    """Lazy mapping-style reader plus serialized per-book writer."""

    def __init__(self, token=None, read=None, publish=None):
        self.token = token
        self._read = read or (lambda path: read_bytes(path, token))
        self._publish = publish or (lambda payloads: publish_indexes(payloads, token))
        self._index = None
        self._cache = {}

    def index(self) -> dict | None:
        if self._index is None:
            try:
                self._index = validate_index(json.loads(self._read(index_path(INDEX))))
            except FileNotFoundError:
                return None
        return self._index

    def exists(self) -> bool:
        return self.index() is not None

    def get(self, key, default=None):
        index = self.index()
        if index is None or key not in index["books"]:
            return default
        if key not in self._cache:
            self._cache[key] = decode_shard(key, self._read(index_path(shard_name(key))))
        return self._cache[key]

    def summaries(self) -> dict:
        index = self.index()
        return dict(index["books"]) if index else {}

    def _write(self, payloads: dict[str, bytes]) -> None:
        self._publish(payloads)
        for path, raw in payloads.items():
            for attempt in range(3):
                if plain(self._read(path)) == plain(raw):
                    break
                if attempt == 2:
                    raise ValueError(f"OCR progress readback mismatch: {path}")
                time.sleep(2 ** attempt)

    def save(self, updates: dict, merge) -> None:
        """Merge ``updates`` into their books; the caller holds ``reader-sidecar``."""
        if not updates:
            return
        index = self.index()
        if index is None:
            raise RuntimeError("OCR progress index is missing; migrate before saving")
        index = json.loads(json.dumps(index))
        payloads = {}
        for key, value in sorted(updates.items()):
            entry = validate_entry(key, merge(self.get(key), value))
            payloads[index_path(shard_name(key))] = encode_shard(key, entry)
            index["books"][key] = summary(entry)
            self._cache[key] = entry
        # Shards first in the same batch; the index only advertises written books.
        payloads[index_path(INDEX)] = encode_json(index)
        self._write(payloads)
        self._index = index


def download_legacy(target: Path, token=None) -> tuple[str, int] | None:
    url = (f"https://huggingface.co/buckets/{shared.READER_ASSETS_BUCKET}/resolve/"
           f"{quote(index_path(LEGACY), safe='/')}")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(8):
        try:
            digest, size = hashlib.sha256(), 0
            with get_session().stream("GET", url, headers=headers, follow_redirects=True,
                                      timeout=httpx.Timeout(120, read=300)) as response:
                if response.status_code == 404:
                    return None
                if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
                    raise httpx.TransportError(f"HTTP {response.status_code}")
                response.raise_for_status()
                with target.open("wb") as stream:
                    for chunk in response.iter_bytes(shared.CHUNK_BYTES):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            return digest.hexdigest(), size
        except httpx.TransportError:
            if attempt == 7:
                raise
            delay = min(300, 5 * 2 ** attempt)
            print(f"legacy OCR progress download retry in {delay}s", flush=True)
            time.sleep(delay)


def migrate_legacy(store: ProgressStore, directory: Path, token=None) -> dict:
    """Split the legacy registry once, streaming it from disk; the legacy file is kept."""
    import ijson

    legacy = directory / "legacy-progress.json"
    identity = download_legacy(legacy, token)
    index = {"version": 1, "kind": INDEX_KIND, "books": {},
             "migrated_from": None if identity is None else {
                 "path": index_path(LEGACY), "sha256": identity[0], "bytes": identity[1]}}
    if identity is not None:
        with legacy.open("rb") as stream:
            stream.seek(max(0, identity[1] - 4096))
            if not re.search(rb'"version"\s*:\s*1\s*}\s*$', stream.read()):
                raise ValueError("legacy OCR progress registry is not version 1")
        batch, pages, shards = {}, 0, set()
        with legacy.open("rb") as stream:
            for key, entry in ijson.kvitems(stream, "files", use_float=True):
                entry = validate_entry(key, entry)
                name = shard_name(key)
                if name in shards:
                    raise ValueError("OCR progress shard collision")
                shards.add(name)
                batch[index_path(name)] = encode_shard(key, entry)
                index["books"][key] = summary(entry)
                pages += len(entry["pages"])
                if len(batch) >= MIGRATION_BATCH:
                    store._write(batch)
                    batch = {}
        if batch:
            store._write(batch)
        index["migrated_from"].update(books=len(index["books"]), pages=pages)
        legacy.unlink()
    # The index is written last: until it exists, every reader keeps the legacy path.
    store._write({index_path(INDEX): encode_json(index)})
    store._index = index
    store._cache = {}
    return index["migrated_from"] or {"books": 0, "pages": 0}


def ensure_migrated(store: ProgressStore, token=None) -> None:
    if store.exists():
        return
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ocr-progress-migration-") as directory:
        result = migrate_legacy(store, Path(directory), token)
    print("ocr progress migration " + json.dumps(
        {**result, "seconds": round(time.monotonic() - started, 1)}, sort_keys=True), flush=True)
