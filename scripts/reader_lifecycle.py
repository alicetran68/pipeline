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


def asset_record(result: dict) -> dict:
    return {
        "key": result["key"],
        "path": result["path"],
        "paths": list(result.get("bucket_paths") or [result["path"]]),
        "sha256": result.get("sha256", ""),
        "bytes": int(result.get("bytes") or 0),
        "source_sha256": result.get("source_sha256", ""),
        "source_revision": result.get("source_revision", ""),
        "profile": result.get("profile", ""),
        "phase": "final",
        "consumers": {},
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


def mark_orphans(manifest: dict, paths: set[str], today: str) -> dict:
    orphans = dict(manifest.get("orphans") or {})
    for path in paths:
        if path not in orphans:
            orphans[path] = {"since": today}
    for path in list(orphans):
        if path not in paths:
            orphans.pop(path, None)
    return {**manifest, "orphans": dict(sorted(orphans.items()))}
