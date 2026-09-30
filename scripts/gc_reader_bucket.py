#!/usr/bin/env python3
"""Mark and collect unreferenced objects in the unified Reader bucket."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import posixpath
from datetime import date, timedelta

try:
    from .reader_assets import READER_ASSETS_BUCKET, READER_STAGING_BUCKET
    from .shared import PDF_OCR_INPUT_BUCKET
    from .reader_bucket import INDEX_PREFIX
    from .reader_lifecycle import LIFECYCLE_NAME, mark_orphans
except ImportError:
    from reader_assets import READER_ASSETS_BUCKET, READER_STAGING_BUCKET
    from shared import PDF_OCR_INPUT_BUCKET
    from reader_bucket import INDEX_PREFIX
    from reader_lifecycle import LIFECYCLE_NAME, mark_orphans


class IndexUnavailable(RuntimeError):
    """The collector cannot prove that the bucket reference graph is complete."""


class S3NotFound(FileNotFoundError):
    """An expected optional S3 object does not exist."""


class S3BucketStore:
    """Small S3 adapter for Hugging Face Storage Buckets."""

    def __init__(self) -> None:
        self._access_key = os.environ.get("HF_S3_ACCESS_KEY_ID")
        self._secret_key = os.environ.get("HF_S3_SECRET_ACCESS_KEY")
        if not self._access_key or not self._secret_key:
            raise RuntimeError(
                "HF_S3_ACCESS_KEY_ID and HF_S3_SECRET_ACCESS_KEY are required"
            )
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError("boto3 is required for bucket GC") from error
        self._boto3 = boto3
        self._config = Config(
            region_name="us-east-1",
            s3={"addressing_style": "path"},
            retries={"mode": "adaptive", "max_attempts": 8},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
        self._namespace = os.environ.get("HF_S3_NAMESPACE", "vomebook")
        self._input_namespace = os.environ.get("HF_S3_INPUT_NAMESPACE", self._namespace)
        self._input_bucket = os.environ.get("HF_S3_INPUT_BUCKET", PDF_OCR_INPUT_BUCKET)
        try:
            self._list_workers = max(1, int(os.environ.get("HF_S3_LIST_WORKERS", "16")))
            self._manifest_workers = max(1, int(os.environ.get("HF_S3_MANIFEST_WORKERS", "4")))
        except ValueError as error:
            raise RuntimeError("HF_S3_LIST_WORKERS and HF_S3_MANIFEST_WORKERS must be positive integers") from error
        self._clients = {}

    def _location(self, bucket: str) -> tuple[str, str]:
        if "/" in bucket:
            return bucket.rsplit("/", 1)
        if bucket == PDF_OCR_INPUT_BUCKET:
            return self._input_namespace, self._input_bucket
        return self._namespace, bucket

    def _client(self, namespace: str):
        input_credentials = namespace == self._input_namespace and namespace != self._namespace
        access_key = os.environ.get("HF_S3_INPUT_ACCESS_KEY_ID") if input_credentials else self._access_key
        secret_key = os.environ.get("HF_S3_INPUT_SECRET_ACCESS_KEY") if input_credentials else self._secret_key
        if not access_key or not secret_key:
            raise RuntimeError(
                "HF_S3_INPUT_ACCESS_KEY_ID and HF_S3_INPUT_SECRET_ACCESS_KEY are required "
                f"for namespace {namespace}"
            )
        client_key = (namespace, access_key)
        client = self._clients.get(client_key)
        if client is None:
            client = self._boto3.client(
                "s3",
                endpoint_url=f"https://s3.hf.co/{namespace}",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=self._config,
            )
            self._clients[client_key] = client
        return client

    @staticmethod
    def _list_flat(client, bucket_name: str, prefix: str) -> set[str]:
        files = set()
        for page in client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket_name, Prefix=prefix):
            files.update(item["Key"] for item in page.get("Contents", []))
        return files

    @staticmethod
    def _list_children(client, bucket_name: str, prefix: str) -> tuple[set[str], set[str]]:
        files = set()
        children = set()
        for page in client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket_name, Prefix=prefix, Delimiter="/"):
            files.update(item["Key"] for item in page.get("Contents", []))
            children.update(item["Prefix"] for item in page.get("CommonPrefixes", []))
        return files, children

    def _list_prefix(self, client, bucket_name: str, prefix: str) -> set[str]:
        """Discover only shallow directories, then list hash shards flat."""
        files, children = self._list_children(client, bucket_name, prefix)
        flat_prefixes = set()
        for child in children:
            # ebook-chapters/ has an intermediate objects/ directory. Discover
            # that one level, but never descend into each object hash folder.
            if child.endswith("objects/"):
                nested_files, nested_children = self._list_children(client, bucket_name, child)
                files.update(nested_files)
                flat_prefixes.update(nested_children)
            else:
                flat_prefixes.add(child)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._list_workers, max(1, len(flat_prefixes)))) as executor:
            results = executor.map(
                lambda child: self._list_flat(client, bucket_name, child),
                sorted(flat_prefixes),
            )
            for result in results:
                files.update(result)
        return files

    def list_files(self, bucket: str, prefixes: tuple[str, ...]) -> set[str]:
        namespace, bucket_name = self._location(bucket)
        client = self._client(namespace)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._list_workers, len(prefixes))) as executor:
            results = executor.map(
                lambda prefix: self._list_prefix(client, bucket_name, prefix),
                prefixes,
            )
            files = set()
            for result in results:
                files.update(result)
        return files

    def read_bytes(self, bucket: str, path: str) -> bytes:
        namespace, bucket_name = self._location(bucket)
        try:
            response = self._client(namespace).get_object(Bucket=bucket_name, Key=path)
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                raise S3NotFound(path) from error
            raise
        return response["Body"].read()

    def put_json(self, bucket: str, path: str, payload: dict) -> None:
        namespace, bucket_name = self._location(bucket)
        body = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        self._client(namespace).put_object(
            Bucket=bucket_name, Key=path, Body=body, ContentType="application/json"
        )

    def delete(self, bucket: str, paths: list[str]) -> None:
        namespace, bucket_name = self._location(bucket)
        client = self._client(namespace)
        for start in range(0, len(paths), 1000):
            response = client.delete_objects(
                Bucket=bucket_name,
                Delete={
                    "Objects": [{"Key": path} for path in paths[start:start + 1000]],
                    "Quiet": True,
                },
            )
            errors = response.get("Errors", [])
            if errors:
                raise RuntimeError(f"S3 delete failed for {len(errors)} object(s)")


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


def collect_bucket_paths(value, bucket: str, output: set[str]) -> None:
    """Collect paths explicitly assigned to a bucket by a Reader sidecar."""
    if isinstance(value, dict):
        if value.get("b") == bucket:
            for field in ("p", "path"):
                path = value.get(field)
                if isinstance(path, str):
                    output.add(path)
        if value.get("ob") == bucket:
            path = value.get("f")
            if isinstance(path, str):
                output.add(path)
        if value.get("cb") == bucket:
            path = value.get("c")
            if isinstance(path, str):
                output.add(path)
        for item in value.values():
            collect_bucket_paths(item, bucket, output)
    elif isinstance(value, list):
        for item in value:
            collect_bucket_paths(item, bucket, output)


def bucket_files(store: S3BucketStore, bucket: str, prefixes: tuple[str, ...]) -> set[str]:
    return store.list_files(bucket, prefixes)


def read_index_payloads(store: S3BucketStore, files: set[str]) -> dict[str, dict]:
    required = {f"{INDEX_PREFIX}/reader_assets.json.gz"}
    if not required.issubset(files):
        raise IndexUnavailable("required Reader bucket indexes are missing")
    payloads = {}
    for path in sorted(path for path in files if path.startswith(INDEX_PREFIX + "/")):
        try:
            raw = store.read_bytes(READER_ASSETS_BUCKET, path)
            payload = decode_sidecar(raw) if path.endswith(".json.gz") else json.loads(raw.decode("utf-8"))
            payloads[path] = payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError, gzip.BadGzipFile):
            raise IndexUnavailable(f"unreadable Reader bucket index: {path}")
    return payloads


def current_references(files: set[str], lifecycle: dict, payloads: dict[str, dict]) -> set[str]:
    # The compact sidecar is the published Reader routing index. The full
    # manifest and lifecycle file are introduced by the incremental migrator
    # and may not exist while older published assets are being collected.
    references = {path for path in files if path.startswith(INDEX_PREFIX + "/")}
    active_keys: set[str] = set()
    for path, payload in payloads.items():
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
    return references


def expand_reference_closure(store: S3BucketStore, files: set[str], references: set[str]) -> None:
    """Follow published manifests to their derived page/chapter objects."""
    manifest_suffixes = (
        "/page-manifest.json", "/ocr-manifest.json", "/render-manifest.json",
        "/chapter-manifest.json",
    )
    pending = {path for path in references if path in files and path.endswith(manifest_suffixes)}
    seen = set()
    while pending:
        batch = sorted(pending)
        pending.clear()
        seen.update(batch)

        def load(path: str) -> tuple[str, dict]:
            try:
                raw = store.read_bytes(READER_ASSETS_BUCKET, path)
                payload = json.loads(gzip.decompress(raw).decode("utf-8")
                                     if raw[:2] == b"\x1f\x8b" else raw.decode("utf-8"))
                return path, payload if isinstance(payload, dict) else {}
            except (OSError, ValueError, json.JSONDecodeError, gzip.BadGzipFile):
                raise IndexUnavailable(f"unreadable Reader object manifest: {path}")

        workers = getattr(store, "_manifest_workers", 4)
        if not isinstance(workers, int):
            workers = 16
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(workers, len(batch))) as executor:
            payloads = executor.map(load, batch)
            for path, payload in payloads:
                collect_paths(payload, references)
                if path.endswith("/page-manifest.json"):
                    page_count = payload.get("page_count")
                    if not isinstance(page_count, int) or page_count < 1:
                        raise IndexUnavailable(f"invalid PDF page manifest: {path}")
                    root = posixpath.dirname(path)
                    references.update(
                        f"{root}/pages/page-{number:06d}.webp"
                        for number in range(1, page_count + 1)
                    )
                elif path.endswith("/chapter-manifest.json"):
                    root = posixpath.dirname(path)
                    chapters = payload.get("chapters")
                    if not isinstance(chapters, list):
                        raise IndexUnavailable(f"invalid ebook chapter manifest: {path}")
                    for chapter in chapters:
                        if isinstance(chapter, dict) and isinstance(chapter.get("path"), str):
                            references.add(posixpath.normpath(posixpath.join(root, chapter["path"])))
                    search_index = payload.get("search_index")
                    if isinstance(search_index, dict) and isinstance(search_index.get("path"), str):
                        references.add(posixpath.normpath(posixpath.join(root, search_index["path"])))
        pending.update(path for path in references if path in files
                       and path.endswith(manifest_suffixes) and path not in seen)


def plan_gc(store: S3BucketStore, grace_days: int, limit: int,
            include_input_bucket: bool = False) -> tuple[dict, dict[str, list[str]], dict[str, int]]:
    files = bucket_files(
        store, READER_ASSETS_BUCKET,
        (INDEX_PREFIX + "/", "objects/", "ebook-chapters/", "staging/"),
    )
    print(f"listed {READER_ASSETS_BUCKET}: {len(files)} object(s)")
    payloads = read_index_payloads(store, files)
    lifecycle = payloads.get(f"{INDEX_PREFIX}/{LIFECYCLE_NAME}",
                             {"version": 1, "files": {}, "orphans": {}})
    references = current_references(files, lifecycle, payloads)
    expand_reference_closure(store, files, references)
    asset_candidates = {f"{READER_ASSETS_BUCKET}:{path}" for path in files
                        if path not in references and not path.startswith(INDEX_PREFIX + "/")}
    candidates = set(asset_candidates)
    input_files = (bucket_files(store, PDF_OCR_INPUT_BUCKET, ("objects/",))
                   if include_input_bucket else set())
    if include_input_bucket:
        print(f"listed {PDF_OCR_INPUT_BUCKET}: {len(input_files)} object(s)")
    input_references = {path for path in references if "/ocr-input/" in path or path.endswith(".jxl")}
    input_candidates = {f"{PDF_OCR_INPUT_BUCKET}:{path}" for path in input_files
                        if path not in input_references}
    candidates.update(input_candidates)
    staging_files = bucket_files(store, READER_STAGING_BUCKET, ("objects/", "staging/"))
    print(f"listed {READER_STAGING_BUCKET}: {len(staging_files)} object(s)")
    staging_references: set[str] = set()
    for payload in payloads.values():
        collect_bucket_paths(payload, READER_STAGING_BUCKET, staging_references)
    staging_references.update(path for path in references if path.startswith("staging/pdf/"))
    staging_candidates = {f"{READER_STAGING_BUCKET}:{path}" for path in staging_files
                          if path not in staging_references}
    candidates.update(staging_candidates)
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
    counts = {READER_ASSETS_BUCKET: len(asset_candidates),
              READER_STAGING_BUCKET: len(staging_candidates),
              PDF_OCR_INPUT_BUCKET: len(input_candidates)}
    return updated, expired, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--grace-days", type=int, default=14)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-input-bucket", action="store_true",
                        help="Also scan the OCR PNG/JXL input bucket after its migration is complete")
    parser.add_argument("--show-paths", action="store_true",
                        help="Print every expired object path in report-only output")
    args = parser.parse_args()
    if args.limit < 0 or args.grace_days < 0:
        raise ValueError("limit and grace-days must be non-negative")
    store = S3BucketStore()
    try:
        lifecycle, expired, counts = plan_gc(store, args.grace_days, args.limit, args.include_input_bucket)
    except IndexUnavailable as error:
        print(f"GC skipped: {error}")
        return 0
    expired_count = sum(len(paths) for paths in expired.values())
    print(f"found {sum(counts.values())} unreferenced object(s), {expired_count} past grace period")
    if args.show_paths:
        for bucket, paths in expired.items():
            for path in paths:
                print(f"{bucket}:{path}")
    if args.apply:
        for bucket, paths in expired.items():
            if paths:
                store.delete(bucket, paths)
        expired_keys = {f"{bucket}:{path}" for bucket, paths in expired.items() for path in paths}
        lifecycle["orphans"] = {path: entry for path, entry in lifecycle.get("orphans", {}).items()
                                 if path not in expired_keys}
        store.put_json(READER_ASSETS_BUCKET, f"{INDEX_PREFIX}/{LIFECYCLE_NAME}", lifecycle)
        print(f"deleted {expired_count} unreferenced object(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
