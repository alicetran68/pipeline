#!/usr/bin/env python3
"""Build and publish incremental native text Reader streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

try:
    from .reader_assets import decode_search_payload, relative_path, source_url
except ImportError:
    from reader_assets import decode_search_payload, relative_path, source_url


TARGET_BUCKET = "vomebook/reader-assets-v2"
ROOT = "documents/text"
SUPPORTED_EXTENSIONS = {"txt", "md", "markdown", "vcf", "ini"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--extension", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bucket", default=TARGET_BUCKET)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def download(url: str, target: Path, token: str | None) -> None:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def decode_text(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    for encoding in ("utf-8", "gb18030", "big5", "shift-jis", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def select_records(path: Path, revisions: dict, extension: str) -> list[dict]:
    records = decode_search_payload(json.loads(path.read_text(encoding="utf-8")))
    selected = []
    for record in records:
        ext = str(record.get("Extension") or "").lower().lstrip(".")
        if ext not in SUPPORTED_EXTENSIONS or (extension != "all" and ext != extension):
            continue
        repo = str(record.get("Repo") or "")
        revision = str(revisions.get(repo) or "")
        if repo and revision:
            selected.append({"repo": repo, "path": relative_path(record), "extension": ext, "revision": revision})
    selected.sort(key=lambda item: (item["extension"], item["repo"], item["path"]))
    return selected


def existing_index(bucket: str, token: str | None) -> dict:
    try:
        fs = HfFileSystem(token=token)
        with fs.open(f"hf://buckets/{bucket}/{ROOT}/index.json", "rb") as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("files"), list):
            return payload
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        pass
    return {"version": 1, "kind": "text-stream-index", "files": []}


def build_one(item: dict, work: Path, token: str | None, bucket: str) -> tuple[dict, dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    raw = source.read_bytes()
    text = decode_text(raw)
    normalized = text.encode("utf-8")
    source_digest = hashlib.sha256(raw).hexdigest()
    output_digest = hashlib.sha256(normalized).hexdigest()
    ext = item["extension"]
    output_ext = "md" if ext == "markdown" else ext
    root = f"{ROOT}/{ext}/{output_digest}"
    object_path = f"{root}/document.{output_ext}"
    manifest_path = f"{root}/document-manifest.json"
    output = work / f"document.{output_ext}"
    output.write_bytes(normalized)
    manifest = {
        "kind": "text-document-stream",
        "version": 1,
        "source_extension": ext,
        "source_sha256": source_digest,
        "bytes": len(normalized),
        "sha256": output_digest,
        "path": f"document.{output_ext}",
    }
    manifest_file = work / "document-manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "key": f"{item['repo']}\0{item['path']}", "repo": item["repo"], "path": item["path"],
        "extension": ext, "source_revision": item["revision"], "source_sha256": source_digest,
        "mode": "document-stream", "bucket": bucket, "object": object_path,
        "manifest": manifest_path, "bytes": len(normalized), "sha256": output_digest,
    }, {object_path: str(output), manifest_path: str(manifest_file)}


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    token = os.environ.get("HF_TOKEN")
    extension = args.extension.lower().lstrip(".")
    if extension != "all" and extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported text extension: {extension}")
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    previous = existing_index(args.bucket, token)
    completed = {entry.get("key"): entry.get("source_revision") for entry in previous.get("files", []) if isinstance(entry, dict)}
    selected = [item for item in select_records(args.search_data, revisions, extension)
                if completed.get(f"{item['repo']}\0{item['path']}") != item["revision"]]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        print("no pending text sources")
        return 0
    uploads, entries = {}, []
    with tempfile.TemporaryDirectory(prefix="reader-text-") as directory:
        root = Path(directory)
        for index, item in enumerate(selected):
            entry, files = build_one(item, root / str(index), token, args.bucket)
            entries.append(entry)
            uploads.update(files)
        merged = {entry.get("key"): entry for entry in previous.get("files", []) if isinstance(entry, dict)}
        merged.update({entry["key"]: entry for entry in entries})
        index_file = root / "index.json"
        index_file.write_text(json.dumps({"version": 1, "kind": "text-stream-index", "files": [merged[key] for key in sorted(merged)]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        uploads[f"{ROOT}/index.json"] = str(index_file)
        print(f"planned {len(entries)} text stream(s), {len(uploads)} object(s)")
        if args.apply:
            batch_bucket_files(args.bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token)
            print(f"published {len(uploads)} object(s) to {args.bucket}")
        else:
            print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
