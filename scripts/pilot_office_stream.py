#!/usr/bin/env python3
"""Build incremental DOC/DOCX/ODT/RTF Reader document streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

try:
    from .convert_reader_assets import decode_html_source, sanitize_html
    from .reader_assets import decode_search_payload, relative_path, source_url
except ImportError:
    from convert_reader_assets import decode_html_source, sanitize_html
    from reader_assets import decode_search_payload, relative_path, source_url


BUCKET = "vomebook/reader-assets-v2"
EXTENSIONS = {"doc", "docx", "odt", "rtf"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    p.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    p.add_argument("--extension", default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--apply", action="store_true")
    return p.parse_args()


def download(url, target, token):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def index(bucket, root, token):
    try:
        fs = HfFileSystem(token=token)
        with fs.open(f"hf://buckets/{bucket}/{root}/index.json", "rb") as stream:
            value = json.loads(stream.read().decode())
        return value if isinstance(value, dict) and isinstance(value.get("files"), list) else {"files": []}
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return {"files": []}


def records(path, revisions, extension):
    rows = decode_search_payload(json.loads(path.read_text(encoding="utf-8")))
    output = []
    for row in rows:
        ext = str(row.get("Extension") or "").lower().lstrip(".")
        if ext not in EXTENSIONS or (extension != "all" and ext != extension):
            continue
        repo = str(row.get("Repo") or "")
        if repo and revisions.get(repo):
            output.append({"repo": repo, "path": relative_path(row), "extension": ext, "revision": revisions[repo]})
    return sorted(output, key=lambda x: (x["extension"], x["repo"], x["path"]))


def build(item, work, token, bucket):
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"source.{item['extension']}"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    ext = item["extension"]
    if ext == "docx":
        output, name = source, "document.docx"
    elif ext == "doc":
        out = work / "docx"
        out.mkdir()
        subprocess.run(["libreoffice", "--headless", "--convert-to", "docx", "--outdir", str(out), str(source)], check=True, timeout=600)
        output, name = out / "source.docx", "document.docx"
    else:
        out = work / "html"
        out.mkdir()
        subprocess.run(["libreoffice", "--headless", "--convert-to", "html", "--outdir", str(out), str(source)], check=True, timeout=600)
        generated = out / f"source.html"
        output = work / "document.html"
        output.write_text(sanitize_html(decode_html_source(generated), allow_relative=False), encoding="utf-8")
        name = "document.html"
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    root = f"documents/office/{ext}/{source_hash}"
    object_path = f"{root}/{name}"
    manifest_path = f"{root}/document-manifest.json"
    manifest = work / "document-manifest.json"
    manifest.write_text(json.dumps({"kind": "office-document-stream", "version": 1, "source_extension": ext, "source_sha256": source_hash, "path": name, "bytes": output.stat().st_size, "sha256": digest}, sort_keys=True, indent=2) + "\n")
    return {"key": f"{item['repo']}\0{item['path']}", "repo": item["repo"], "path": item["path"], "extension": ext, "source_revision": item["revision"], "source_sha256": source_hash, "mode": "document-stream", "bucket": bucket, "object": object_path, "manifest": manifest_path, "bytes": output.stat().st_size, "sha256": digest}, {object_path: str(output), manifest_path: str(manifest)}


def main():
    a = args()
    if a.limit < 0: raise ValueError("limit must be non-negative")
    ext = a.extension.lower().lstrip(".")
    if ext != "all" and ext not in EXTENSIONS: raise ValueError(f"unsupported office extension: {ext}")
    token = os.environ.get("HF_TOKEN")
    revisions = json.loads(a.revisions.read_text(encoding="utf-8"))
    selected = records(a.search_data, revisions, ext)
    entries, uploads = [], {}
    with tempfile.TemporaryDirectory(prefix="reader-office-") as directory:
        root = Path(directory)
        for number, item in enumerate(selected):
            if a.limit and len(entries) >= a.limit: break
            category = f"documents/office/{item['extension']}"
            old = index(a.bucket, category, token)
            key = f"{item['repo']}\0{item['path']}"
            if any(e.get("key") == key and e.get("source_revision") == item["revision"] for e in old["files"]): continue
            try: entry, files = build(item, root / str(number), token, a.bucket)
            except Exception as error:
                print(f"failed: {item['repo']}/{item['path']}: {type(error).__name__}: {error}"); continue
            entries.append(entry); uploads.update(files)
        if not entries: print("no pending office sources"); return 0
        by_category = {}
        for entry in entries: by_category.setdefault(f"documents/office/{entry['extension']}", []).append(entry)
        for category, added in by_category.items():
            old = index(a.bucket, category, token); merged = {e.get("key"): e for e in old["files"] if isinstance(e, dict)}; merged.update({e["key"]: e for e in added})
            out = root / category.replace("/", "_") / "index.json"; out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"version": 1, "kind": "office-document-stream-index", "files": [merged[k] for k in sorted(merged)]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            uploads[f"{category}/index.json"] = str(out)
        print(f"planned {len(entries)} office stream(s), {len(uploads)} object(s)")
        if a.apply:
            batch_bucket_files(a.bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token); print(f"published {len(uploads)} object(s) to {a.bucket}")
        else: print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__": raise SystemExit(main())
