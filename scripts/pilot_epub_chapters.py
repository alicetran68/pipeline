#!/usr/bin/env python3
"""Build and publish incremental EPUB chapter streams to Reader Assets v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

try:
    from .epub_chapters import build_bundle, bundle_version
    from .reader_assets import decode_search_payload, relative_path, source_url
except ImportError:
    from epub_chapters import build_bundle, bundle_version
    from reader_assets import decode_search_payload, relative_path, source_url


TARGET_BUCKET = "vomebook/reader-assets-v2"
ROOT = "chapters/ebook/epub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-source-bytes", type=int, default=0)
    parser.add_argument("--bucket", default=TARGET_BUCKET)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def download(url: str, target: Path, token: str | None) -> None:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def existing_index(bucket: str, token: str | None) -> dict:
    try:
        fs = HfFileSystem(token=token)
        with fs.open(f"hf://buckets/{bucket}/{ROOT}/index.json", "rb") as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("files"), list):
            return payload
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        pass
    return {"version": 1, "kind": "ebook-chapter-stream-index", "files": []}


def select_records(path: Path, revisions: dict, max_bytes: int) -> list[dict]:
    records = decode_search_payload(json.loads(path.read_text(encoding="utf-8")))
    selected = []
    for record in records:
        extension = str(record.get("Extension") or "").lower().lstrip(".")
        size = int(record.get("Size") or 0)
        if extension != "epub" or (max_bytes and size > max_bytes):
            continue
        repo = str(record.get("Repo") or "")
        revision = str(revisions.get(repo) or "")
        if repo and revision:
            selected.append({"repo": repo, "path": relative_path(record), "revision": revision, "source_bytes": size})
    selected.sort(key=lambda item: (item["repo"], item["path"]))
    return selected


def build_one(item: dict, work: Path, token: str | None, bucket: str) -> tuple[dict, dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.epub"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    source_bytes = source.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    bundle = work / "bundle"
    manifest = build_bundle(source, bundle)
    version = bundle_version(bundle)
    root = f"{ROOT}/{source_digest}/{version}"
    uploads = {}
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        uploads[f"{root}/{path.relative_to(bundle).as_posix()}"] = str(path)
    return {
        "key": f"{item['repo']}\0{item['path']}", "repo": item["repo"], "path": item["path"],
        "source_revision": item["revision"], "source_sha256": source_digest,
        "source_bytes": len(source_bytes), "mode": "chapter-stream", "bucket": bucket,
        "root": root, "manifest": f"{root}/chapter-manifest.json",
        "chapter_count": len(manifest["chapters"]),
        "search_index": f"{root}/epub-search-index.json.gz",
    }, uploads


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.max_source_bytes < 0:
        raise ValueError("limit and max-source-bytes must be non-negative")
    token = os.environ.get("HF_TOKEN")
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    previous = existing_index(args.bucket, token)
    completed = {entry.get("key"): entry.get("source_revision") for entry in previous.get("files", []) if isinstance(entry, dict)}
    selected = [item for item in select_records(args.search_data, revisions, args.max_source_bytes)
                if completed.get(f"{item['repo']}\0{item['path']}") != item["revision"]]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        print("no pending EPUB sources")
        return 0
    uploads, entries = {}, []
    with tempfile.TemporaryDirectory(prefix="reader-epub-") as directory:
        root = Path(directory)
        for index, item in enumerate(selected):
            entry, files = build_one(item, root / str(index), token, args.bucket)
            entries.append(entry)
            uploads.update(files)
        merged = {entry.get("key"): entry for entry in previous.get("files", []) if isinstance(entry, dict)}
        merged.update({entry["key"]: entry for entry in entries})
        index_file = root / "index.json"
        index_file.write_text(json.dumps({"version": 1, "kind": "ebook-chapter-stream-index", "files": [merged[key] for key in sorted(merged)]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        uploads[f"{ROOT}/index.json"] = str(index_file)
        print(f"planned {len(entries)} EPUB stream(s), {len(uploads)} object(s)")
        if args.apply:
            batch_bucket_files(args.bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token)
            print(f"published {len(uploads)} object(s) to {args.bucket}")
        else:
            print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
