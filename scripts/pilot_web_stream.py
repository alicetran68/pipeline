#!/usr/bin/env python3
"""Build and publish sanitized HTML/MHTML Reader document streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

try:
    from .convert_reader_assets import inline_html_resources, mhtml_to_html
    from .reader_assets import decode_search_payload, relative_path, source_url
except ImportError:
    from convert_reader_assets import inline_html_resources, mhtml_to_html
    from reader_assets import decode_search_payload, relative_path, source_url


TARGET_BUCKET = "vomebook/reader-assets-v2"
SUPPORTED_EXTENSIONS = {"htm", "html", "mht", "mhtml"}


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
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def existing_index(bucket: str, token: str | None, root: str) -> dict:
    try:
        fs = HfFileSystem(token=token)
        with fs.open(f"hf://buckets/{bucket}/{root}/index.json", "rb") as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("files"), list):
            return payload
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        pass
    return {"version": 1, "kind": "web-document-stream-index", "files": []}


def select_records(path: Path, revisions: dict, extension: str) -> list[dict]:
    records = decode_search_payload(json.loads(path.read_text(encoding="utf-8")))
    result = []
    for record in records:
        ext = str(record.get("Extension") or "").lower().lstrip(".")
        if ext not in SUPPORTED_EXTENSIONS or (extension != "all" and ext != extension):
            continue
        repo = str(record.get("Repo") or "")
        revision = str(revisions.get(repo) or "")
        if repo and revision:
            result.append({"repo": repo, "path": relative_path(record), "extension": ext, "revision": revision})
    result.sort(key=lambda item: (item["extension"], item["repo"], item["path"]))
    return result


def build_one(item: dict, work: Path, token: str | None, bucket: str) -> tuple[dict, dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"source.{item['extension']}"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    raw_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if item["extension"] in {"mht", "mhtml"}:
        html_path = work / "document.html"
        mhtml_to_html(source, html_path)
    else:
        html_path = inline_html_resources(source, source_url(item["repo"], item["revision"], item["path"]), work)
    output = html_path.read_bytes()
    output_digest = hashlib.sha256(output).hexdigest()
    root = f"documents/web/{item['extension']}/{raw_digest}"
    object_path = f"{root}/document.html"
    manifest_path = f"{root}/document-manifest.json"
    manifest_file = work / "document-manifest.json"
    manifest_file.write_text(json.dumps({
        "kind": "web-document-stream", "version": 1, "source_extension": item["extension"],
        "source_sha256": raw_digest, "path": "document.html", "bytes": len(output),
        "sha256": output_digest,
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "key": f"{item['repo']}\0{item['path']}", "repo": item["repo"], "path": item["path"],
        "extension": item["extension"], "source_revision": item["revision"], "source_sha256": raw_digest,
        "mode": "document-stream", "bucket": bucket, "object": object_path, "manifest": manifest_path,
        "bytes": len(output), "sha256": output_digest,
    }, {object_path: str(html_path), manifest_path: str(manifest_file)}


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    extension = args.extension.lower().lstrip(".")
    if extension != "all" and extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported web extension: {extension}")
    token = os.environ.get("HF_TOKEN")
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    selected_all = select_records(args.search_data, revisions, extension)
    uploads, entries = {}, []
    with tempfile.TemporaryDirectory(prefix="reader-web-") as directory:
        root = Path(directory)
        for index, item in enumerate(selected_all):
            if args.limit and len(entries) >= args.limit:
                break
            category = f"documents/web/{item['extension']}"
            previous = existing_index(args.bucket, token, category)
            key = f"{item['repo']}\0{item['path']}"
            if any(e.get("key") == key and e.get("source_revision") == item["revision"] for e in previous.get("files", [])):
                continue
            try:
                entry, files = build_one(item, root / str(index), token, args.bucket)
            except Exception as error:
                print(f"failed: {item['repo']}/{item['path']}: {type(error).__name__}: {error}")
                continue
            entries.append(entry)
            uploads.update(files)
        if not entries:
            print("no pending web sources")
            return 0
        grouped = {}
        for entry in entries:
            grouped.setdefault(f"documents/web/{entry['extension']}", []).append(entry)
        for category, new_entries in grouped.items():
            previous = existing_index(args.bucket, token, category)
            merged = {e.get("key"): e for e in previous.get("files", []) if isinstance(e, dict)}
            merged.update({e["key"]: e for e in new_entries})
            index_file = root / category.replace("/", "_") / "index.json"
            index_file.parent.mkdir(parents=True, exist_ok=True)
            index_file.write_text(json.dumps({"version": 1, "kind": "web-document-stream-index", "files": [merged[k] for k in sorted(merged)]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            uploads[f"{category}/index.json"] = str(index_file)
        print(f"planned {len(entries)} web stream(s), {len(uploads)} object(s)")
        if args.apply:
            batch_bucket_files(args.bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token)
            print(f"published {len(uploads)} object(s) to {args.bucket}")
        else:
            print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
