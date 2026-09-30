#!/usr/bin/env python3
"""Mark and collect unreferenced objects in the unified Reader bucket."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from datetime import date, timedelta

from huggingface_hub import batch_bucket_files, list_bucket_tree

try:
    from .reader_assets import READER_ASSETS_BUCKET, READER_STAGING_BUCKET
    from .shared import PDF_OCR_INPUT_BUCKET
    from .reader_bucket import INDEX_PREFIX, publish_json, read_bytes, read_json
    from .reader_lifecycle import LIFECYCLE_NAME, mark_orphans
except ImportError:
    from reader_assets import READER_ASSETS_BUCKET, READER_STAGING_BUCKET
    from shared import PDF_OCR_INPUT_BUCKET
    from reader_bucket import INDEX_PREFIX, publish_json, read_bytes, read_json
    from reader_lifecycle import LIFECYCLE_NAME, mark_orphans


class IndexUnavailable(RuntimeError):
    """The collector cannot prove that the bucket reference graph is complete."""


def apply_limit(paths: list[str], limit: int) -> list[str]:
    """A zero limit means all eligible paths; positive values cap one run."""
    return sorted(paths) if limit == 0 else sorted(paths)[:limit]


def decode_sidecar(raw: bytes) -> dict:
    value = json.loads(gzip.decompress(raw).decode("utf-8"))
    return value if isinstance(value, dict) else {}


def collect_paths(value, output: set[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            collect_paths(item, output)
    elif isinstance(value, list):
        for item in value:
            collect_paths(item, output)
    elif isinstance(value, str) and (
            value.startswith("objects/") or value.startswith("ebook-chapters/")
            or value.startswith("staging/")):
        output.add(value)


def bucket_files(bucket: str, token: str, prefixes: tuple[str, ...]) -> set[str]:
    files = set()
    for prefix in prefixes:
        files.update(item.path for item in list_bucket_tree(
            bucket, prefix=prefix, recursive=True, token=token) if item.type == "file")
    return files


def current_references(token: str, files: set[str], lifecycle: dict) -> set[str]:
    required = {
        f"{INDEX_PREFIX}/manifest.json",
        f"{INDEX_PREFIX}/reader_assets.json.gz",
        f"{INDEX_PREFIX}/{LIFECYCLE_NAME}",
    }
    if not required.issubset(files):
        raise IndexUnavailable("required Reader bucket indexes are missing")
    references = {path for path in files if path.startswith(INDEX_PREFIX + "/")}
    active_keys: set[str] = set()
    registry_paths = sorted(path for path in files if path.startswith(INDEX_PREFIX + "/"))
    for path in registry_paths:
        try:
            raw = read_bytes(path, token)
            payload = decode_sidecar(raw) if path.endswith(".json.gz") else json.loads(raw.decode("utf-8"))
            if path.endswith("/reader_lifecycle.json"):
                # Lifecycle owns staging inputs. A completed staging record is
                # intentionally collectible and must not keep its PDF alive.
                for key, entry in payload.get("files", {}).items():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("phase") in {"staging", "processing"} or (
                            entry.get("phase") == "final" and key in active_keys):
                        references.update(entry.get("paths") or [entry.get("path", "")])
            elif path.endswith("/manifest.json"):
                active_keys = {key for key, value in payload.get("files", {}).items()
                               if isinstance(value, dict) and value.get("status") == "ready"
                               and not value.get("bucket_staging")}
                filtered = dict(payload)
                filtered["files"] = {
                    key: value for key, value in payload.get("files", {}).items()
                    if not isinstance(value, dict) or not value.get("bucket_staging")
                }
                collect_paths(filtered, references)
            else:
                collect_paths(payload, references)
        except (OSError, ValueError, json.JSONDecodeError, gzip.BadGzipFile):
            raise IndexUnavailable(f"unreadable Reader bucket index: {path}")
    return references


def plan_gc(token: str, grace_days: int, limit: int, include_input_bucket: bool = False) -> tuple[dict, dict[str, list[str]], dict[str, int]]:
    files = bucket_files(
        READER_ASSETS_BUCKET, token,
        (INDEX_PREFIX + "/", "objects/", "ebook-chapters/", "staging/"),
    )
    try:
        lifecycle = read_json(f"{INDEX_PREFIX}/{LIFECYCLE_NAME}", token)
    except (OSError, ValueError, json.JSONDecodeError):
        lifecycle = {"version": 1, "files": {}, "orphans": {}}
    references = current_references(token, files, lifecycle)
    candidates = {f"{READER_ASSETS_BUCKET}:{path}" for path in files
                  if path not in references and not path.startswith(INDEX_PREFIX + "/")}
    input_files = (bucket_files(PDF_OCR_INPUT_BUCKET, token, ("objects/",))
                   if include_input_bucket else set())
    input_references = {path for path in references if "/ocr-input/" in path or path.endswith(".jxl")}
    candidates.update(f"{PDF_OCR_INPUT_BUCKET}:{path}" for path in input_files if path not in input_references)
    staging_files = bucket_files(READER_STAGING_BUCKET, token, ("staging/",))
    staging_references = {path for path in references if path.startswith("staging/pdf/")}
    candidates.update(f"{READER_STAGING_BUCKET}:{path}" for path in staging_files
                     if path not in staging_references)
    updated = mark_orphans(lifecycle, candidates, date.today().isoformat())
    cutoff = date.today() - timedelta(days=grace_days)
    expired: dict[str, list[str]] = {READER_ASSETS_BUCKET: [], READER_STAGING_BUCKET: [], PDF_OCR_INPUT_BUCKET: []}
    for path, entry in updated.get("orphans", {}).items():
        try:
            since = date.fromisoformat(entry["since"])
        except (KeyError, TypeError, ValueError):
            continue
        if since <= cutoff:
            bucket, separator, object_path = path.partition(":")
            if separator and bucket in expired:
                expired[bucket].append(object_path)
    for bucket in expired:
        expired[bucket] = apply_limit(expired[bucket], limit)
    counts = {READER_ASSETS_BUCKET: len(candidates), READER_STAGING_BUCKET: len(staging_files),
              PDF_OCR_INPUT_BUCKET: len(input_files)}
    return updated, expired, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--grace-days", type=int, default=14)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-input-bucket", action="store_true",
                        help="Also scan the OCR PNG/JXL input bucket after its migration is complete")
    args = parser.parse_args()
    if args.limit < 0 or args.grace_days < 0:
        raise ValueError("limit and grace-days must be non-negative")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    try:
        lifecycle, expired, counts = plan_gc(token, args.grace_days, args.limit, args.include_input_bucket)
    except IndexUnavailable as error:
        print(f"GC skipped: {error}")
        return 0
    expired_count = sum(len(paths) for paths in expired.values())
    print(f"found {sum(counts.values())} unreferenced object(s), {expired_count} past grace period")
    for bucket, paths in expired.items():
        for path in paths:
            print(f"{bucket}:{path}")
    if args.apply:
        for bucket, paths in expired.items():
            if paths:
                batch_bucket_files(bucket, delete=paths, token=token)
        expired_keys = {f"{bucket}:{path}" for bucket, paths in expired.items() for path in paths}
        lifecycle["orphans"] = {path: entry for path, entry in lifecycle.get("orphans", {}).items()
                                 if path not in expired_keys}
        publish_json(f"{INDEX_PREFIX}/{LIFECYCLE_NAME}", lifecycle, token)
        print(f"deleted {expired_count} unreferenced object(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
