#!/usr/bin/env python3
"""Lifecycle records and conservative collection rules for Reader objects."""

from __future__ import annotations

from datetime import datetime, timezone

LIFECYCLE_VERSION = 1
LIFECYCLE_NAME = "reader_lifecycle.json"
TERMINAL_SUCCESS = {"done", "skipped"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty_manifest() -> dict:
    return {"version": LIFECYCLE_VERSION, "files": {}, "orphans": {}}


def staging_record(result: dict, *, consumers: dict[str, str] | None = None) -> dict:
    staging = bool(result.get("bucket_staging"))
    range_needed = int(result.get("bytes") or result.get("source_bytes") or 0) >= 4 * 1024 * 1024
    return {
        "key": result["key"],
        "path": result["path"],
        "paths": list(result.get("bucket_paths") or [result["path"]]),
        "sha256": result.get("sha256", ""),
        "bytes": int(result.get("bytes") or 0),
        "source_sha256": result.get("source_sha256", ""),
        "source_revision": result.get("source_revision", ""),
        "profile": result.get("profile", ""),
        "phase": "staging" if staging else "final",
        "consumers": dict(consumers or ({"render": "pending", "range": "pending" if range_needed else "not-needed"}
                                         if staging else {})),
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def merge(manifest: dict, updates: list[dict]) -> dict:
    if manifest.get("version") != LIFECYCLE_VERSION or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid Reader lifecycle manifest")
    result = {"version": LIFECYCLE_VERSION, "files": dict(manifest["files"]),
              "orphans": dict(manifest.get("orphans") or {})}
    for update in updates:
        key = update.get("key")
        if not key:
            continue
        previous = dict(result["files"].get(key) or {})
        merged = {**previous, **update, "updated_at": now_iso()}
        if previous.get("created_at"):
            merged["created_at"] = previous["created_at"]
        result["files"][key] = merged
    return result


def consumer_done(manifest: dict, key: str, consumer: str) -> bool:
    entry = manifest.get("files", {}).get(key, {})
    return entry.get("consumers", {}).get(consumer) in TERMINAL_SUCCESS


def collectible(entry: dict, *, require: tuple[str, ...] = ("render", "range")) -> bool:
    if not isinstance(entry, dict) or entry.get("phase") not in {"staging", "processing"}:
        return False
    consumers = entry.get("consumers")
    if not isinstance(consumers, dict):
        return False
    return all(consumers.get(name) in TERMINAL_SUCCESS or consumers.get(name) == "not-needed"
               for name in require)


def mark_consumer(manifest: dict, key: str, consumer: str, status: str) -> dict:
    entry = manifest.get("files", {}).get(key)
    if not isinstance(entry, dict):
        return manifest
    consumers = dict(entry.get("consumers") or {})
    consumers[consumer] = status
    entry = {**entry, "consumers": consumers,
             "phase": "final" if collectible({**entry, "consumers": consumers}) else "processing",
             "updated_at": now_iso()}
    return {**manifest, "files": {**manifest["files"], key: entry}}


def mark_orphans(manifest: dict, paths: set[str], today: str) -> dict:
    orphans = dict(manifest.get("orphans") or {})
    for path in paths:
        if path not in orphans:
            orphans[path] = {"since": today}
    for path in list(orphans):
        if path not in paths:
            orphans.pop(path, None)
    return {**manifest, "orphans": dict(sorted(orphans.items()))}
