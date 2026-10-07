#!/usr/bin/env python3
"""Build incremental selectable HTML worksheet streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import msoffcrypto
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

try:
    from .convert_reader_assets import convert_spreadsheet_to_html
    from .reader_assets import decode_search_payload, relative_path, source_url
except ImportError:
    from convert_reader_assets import convert_spreadsheet_to_html
    from reader_assets import decode_search_payload, relative_path, source_url


BUCKET = "vomebook/reader-assets-v2"
EXTENSIONS = {"xls", "xlsx", "csv", "ods"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    p.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    p.add_argument("--extension", default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--retry-failures", action="store_true")
    return p.parse_args()


def download(url, target, token):
    request = urllib.request.Request(url)
    if token: request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=180) as response: target.write_bytes(response.read())


def read_index(bucket, root, token):
    try:
        fs = HfFileSystem(token=token)
        with fs.open(f"hf://buckets/{bucket}/{root}/index.json", "rb") as stream: value = json.loads(stream.read().decode())
        return value if isinstance(value, dict) and isinstance(value.get("files"), list) else {"files": []}
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError): return {"files": []}


def select(path, revisions, extension):
    rows = decode_search_payload(json.loads(path.read_text(encoding="utf-8"))); result = []
    for row in rows:
        ext = str(row.get("Extension") or "").lower().lstrip(".")
        if ext not in EXTENSIONS or (extension != "all" and ext != extension): continue
        repo = str(row.get("Repo") or "")
        if repo and revisions.get(repo): result.append({"repo": repo, "path": relative_path(row), "extension": ext, "revision": revisions[repo]})
    return sorted(result, key=lambda x: (x["extension"], x["repo"], x["path"]))


def build(item, work, token, bucket):
    work.mkdir(parents=True, exist_ok=True); source = work / f"source.{item['extension']}"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest(); output = work / "document.html"
    conversion_source = source
    password = os.environ.get("READER_CONVERSION_PASSWORD")
    with source.open("rb") as stream:
        encrypted = msoffcrypto.OfficeFile(stream)
        if encrypted.is_encrypted():
            if not password:
                raise RuntimeError("encrypted spreadsheet requires READER_CONVERSION_PASSWORD")
            decrypted = work / f"decrypted.{item['extension']}"
            encrypted.load_key(password=password)
            with decrypted.open("wb") as target:
                encrypted.decrypt(target)
            conversion_source = decrypted
    convert_spreadsheet_to_html(conversion_source, output, work, {"extension": item["extension"], "path": item["path"]})
    digest = hashlib.sha256(output.read_bytes()).hexdigest(); root = f"documents/spreadsheet/{item['extension']}/{source_hash}"
    obj, manifest_path = f"{root}/document.html", f"{root}/worksheet-manifest.json"
    manifest = work / "worksheet-manifest.json"
    manifest.write_text(json.dumps({"kind": "spreadsheet-html-stream", "version": 1, "source_extension": item["extension"], "source_sha256": source_hash, "path": "document.html", "bytes": output.stat().st_size, "sha256": digest}, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {"key": f"{item['repo']}\0{item['path']}", "repo": item["repo"], "path": item["path"], "extension": item["extension"], "source_revision": item["revision"], "source_sha256": source_hash, "mode": "document-stream", "bucket": bucket, "object": obj, "manifest": manifest_path, "bytes": output.stat().st_size, "sha256": digest}, {obj: str(output), manifest_path: str(manifest)}


def main():
    a = parse_args()
    if a.limit < 0: raise ValueError("limit must be non-negative")
    ext = a.extension.lower().lstrip(".")
    if ext != "all" and ext not in EXTENSIONS: raise ValueError(f"unsupported spreadsheet extension: {ext}")
    token = os.environ.get("HF_TOKEN"); revisions = json.loads(a.revisions.read_text(encoding="utf-8")); selected = select(a.search_data, revisions, ext)
    entries, uploads, failures = [], {}, {}
    with tempfile.TemporaryDirectory(prefix="reader-spreadsheet-") as directory:
        root = Path(directory)
        for number, item in enumerate(selected):
            if a.limit and len(entries) >= a.limit:
                break
            category = f"documents/spreadsheet/{item['extension']}"
            old = read_index(a.bucket, category, token)
            key = f"{item['repo']}\0{item['path']}"
            handled = list(old.get("files", []))
            if not a.retry_failures:
                handled += list(old.get("failures", []))
            if any(e.get("key") == key and e.get("source_revision") == item["revision"]
                   for e in handled if isinstance(e, dict)):
                continue
            try:
                entry, files = build(item, root / str(number), token, a.bucket)
            except Exception as error:
                failures[key] = {
                    "key": key, "repo": item["repo"], "path": item["path"],
                    "extension": item["extension"], "source_revision": item["revision"],
                    "error": f"{type(error).__name__}: {error}"[:1000],
                }
                print(f"failed: {item['repo']}/{item['path']}: {type(error).__name__}: {error}")
                continue
            entries.append(entry)
            uploads.update(files)
        if not entries and not failures:
            print("no pending spreadsheet sources"); return 0
        by_category = {}
        for entry in entries: by_category.setdefault(f"documents/spreadsheet/{entry['extension']}", []).append(entry)
        for failure in failures.values():
            by_category.setdefault(f"documents/spreadsheet/{failure['extension']}", [])
        for category, added in by_category.items():
            old = read_index(a.bucket, category, token)
            merged = {e.get("key"): e for e in old["files"] if isinstance(e, dict)}
            merged.update({e["key"]: e for e in added})
            category_failures = {e.get("key"): e for e in old.get("failures", []) if isinstance(e, dict)}
            category_failures.update({e["key"]: e for e in failures.values()
                                      if f"documents/spreadsheet/{e['extension']}" == category})
            category_failures = {k: v for k, v in category_failures.items() if k not in merged}
            out = root / category.replace("/", "_") / "index.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"version": 1, "kind": "spreadsheet-html-stream-index", "files": [merged[k] for k in sorted(merged)], "failures": [category_failures[k] for k in sorted(category_failures)]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            uploads[f"{category}/index.json"] = str(out)
        print(f"planned {len(entries)} spreadsheet stream(s), {len(uploads)} object(s)")
        if a.apply: batch_bucket_files(a.bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token); print(f"published {len(uploads)} object(s) to {a.bucket}")
        else: print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__": raise SystemExit(main())
