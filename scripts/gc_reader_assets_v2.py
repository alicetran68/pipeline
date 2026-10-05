#!/usr/bin/env python3
"""Conservative GC for the v2 Reader asset bucket."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import boto3
from botocore.config import Config


DEFAULT_BUCKET = "vomebook/reader-assets-v2"
LIFECYCLE_PATH = "reader-index/reader_lifecycle.json"
INDEX_PREFIXES = (
    "pages/image/", "pages/pdf/", "pages/document/",
    "documents/text/", "documents/web/", "documents/spreadsheet/", "documents/office/",
    "chapters/ebook/epub/", "chapters/ebook/chm/", "chapters/", "native/",
)


class IndexUnavailable(RuntimeError):
    pass


def s3_client() -> tuple[object, str]:
    namespace, bucket = os.environ.get("HF_S3_BUCKET", DEFAULT_BUCKET).rsplit("/", 1)
    key = os.environ.get("HF_S3_ACCESS_KEY_ID")
    secret = os.environ.get("HF_S3_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise RuntimeError("HF_S3_ACCESS_KEY_ID and HF_S3_SECRET_ACCESS_KEY are required")
    return boto3.client(
        "s3", endpoint_url=f"https://s3.hf.co/{namespace}",
        aws_access_key_id=key, aws_secret_access_key=secret,
        config=Config(region_name="us-east-1", s3={"addressing_style": "path"},
                      retries={"mode": "adaptive", "max_attempts": 8}),
    ), bucket


def list_files(client, bucket: str) -> set[str]:
    files = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        files.update(item["Key"] for item in page.get("Contents", []))
    return files


def read_json(client, bucket: str, path: str) -> dict:
    try:
        raw = client.get_object(Bucket=bucket, Key=path)["Body"].read()
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            raise IndexUnavailable(f"missing required index: {path}") from error
        raise
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndexUnavailable(f"invalid JSON index: {path}") from error
    if not isinstance(payload, dict):
        raise IndexUnavailable(f"invalid JSON index: {path}")
    return payload


def read_lifecycle(client, bucket: str) -> dict:
    try:
        payload = read_json(client, bucket, LIFECYCLE_PATH)
    except IndexUnavailable as error:
        if "missing required index" not in str(error):
            raise
        return {"version": 2, "orphans": {}}
    if payload.get("version") != 2 or not isinstance(payload.get("orphans"), dict):
        raise IndexUnavailable("invalid v2 lifecycle index")
    return payload


def references_from_indexes(client, bucket: str, files: set[str]) -> set[str]:
    references = {path for path in files if path.endswith("/index.json")}
    found_index = False
    for prefix in INDEX_PREFIXES:
        index_paths = [path for path in files
                       if path.startswith(prefix + "index") and path.endswith(".json")]
        for index_path in index_paths:
            found_index = True
            references.add(index_path)
            payload = read_json(client, bucket, index_path)
            entries = payload.get("files")
            if not isinstance(entries, list):
                raise IndexUnavailable(f"invalid file list: {index_path}")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise IndexUnavailable(f"invalid entry: {index_path}")
                for field in ("object", "manifest", "path", "root"):
                    value = entry.get(field)
                    if isinstance(value, str):
                        root = value if field == "root" else value.rsplit("/", 1)[0]
                        references.update(path for path in files if path == value or path.startswith(root + "/"))
    if not found_index:
        raise IndexUnavailable("no v2 category index is published")
    return references


def mark_orphans(lifecycle: dict, candidates: set[str]) -> dict:
    today = date.today().isoformat()
    orphans = dict(lifecycle.get("orphans", {}))
    for path in candidates:
        if path not in orphans:
            orphans[path] = {"since": today}
    for path in list(orphans):
        if path not in candidates:
            del orphans[path]
    return {"version": 2, "orphans": dict(sorted(orphans.items()))}


def plan(client, bucket: str, grace_days: int, limit: int) -> tuple[dict, list[str], dict[str, int]]:
    files = list_files(client, bucket)
    references = references_from_indexes(client, bucket, files)
    candidates = {path for path in files if path not in references and not path.startswith("reader-index/")}
    lifecycle = mark_orphans(read_lifecycle(client, bucket), candidates)
    cutoff = date.today() - timedelta(days=grace_days)
    expired = []
    for path, entry in lifecycle["orphans"].items():
        try:
            since = date.fromisoformat(entry["since"])
        except (KeyError, TypeError, ValueError):
            continue
        if since <= cutoff:
            expired.append(path)
    return lifecycle, sorted(expired)[:limit] if limit else sorted(expired), {
        "files": len(files), "references": len(references), "candidates": len(candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("HF_S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--grace-days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--show-paths", action="store_true")
    args = parser.parse_args()
    if args.grace_days < 0 or args.limit < 0:
        raise ValueError("grace-days and limit must be non-negative")
    os.environ["HF_S3_BUCKET"] = args.bucket
    client, bucket = s3_client()
    try:
        lifecycle, expired, counts = plan(client, bucket, args.grace_days, args.limit)
    except IndexUnavailable as error:
        print(f"GC skipped: {error}")
        return 0
    client.put_object(Bucket=bucket, Key=LIFECYCLE_PATH,
                      Body=(json.dumps(lifecycle, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
                      ContentType="application/json")
    print(f"bucket={args.bucket} files={counts['files']} references={counts['references']} "
          f"orphans={counts['candidates']} expired={len(expired)}")
    if args.show_paths:
        for path in expired:
            print(path)
    if args.apply and expired:
        for start in range(0, len(expired), 1000):
            client.delete_objects(Bucket=bucket, Delete={
                "Objects": [{"Key": path} for path in expired[start:start + 1000]],
                "Quiet": True,
            })
        lifecycle["orphans"] = {path: entry for path, entry in lifecycle["orphans"].items() if path not in expired}
        client.put_object(Bucket=bucket, Key=LIFECYCLE_PATH,
                          Body=(json.dumps(lifecycle, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
                          ContentType="application/json")
        print(f"deleted={len(expired)}")
    elif args.apply:
        print("deleted=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
