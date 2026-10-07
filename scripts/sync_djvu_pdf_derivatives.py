#!/usr/bin/env python3
"""Incrementally convert current DJVU sources into pdf-pages-v2 derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, batch_bucket_files

try:
    from .convert_reader_assets import convert_file, validate_djvu_pdf
    from .reader_assets import bucket_conversion_contract, decode_search_payload, relative_path, source_url
except ImportError:
    from convert_reader_assets import convert_file, validate_djvu_pdf
    from reader_assets import bucket_conversion_contract, decode_search_payload, relative_path, source_url


BUCKET = "vomebook/pdf-pages-v2"
MANIFEST = "reader-index/derived_pdf_manifest.json"
PROFILE = "djvulibre-pdf-v2"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    p.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--apply", action="store_true")
    return p.parse_args()


def read_manifest(token: str) -> dict:
    fs = HfFileSystem(token=token)
    try:
        with fs.open(f"hf://buckets/{BUCKET}/{MANIFEST}", "rb") as stream:
            return json.loads(stream.read().decode("utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return {"version": 1, "kind": "pdf-derived-source-index", "generation": "weekly-djvu", "files": []}


def download(url: str, target: Path, token: str) -> None:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def main() -> int:
    a = args()
    if a.limit < 0:
        raise ValueError("limit must be non-negative")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    revisions = json.loads(a.revisions.read_text(encoding="utf-8"))
    records = []
    for record in decode_search_payload(json.loads(a.search_data.read_text(encoding="utf-8"))):
        if str(record.get("Extension") or "").lower().lstrip(".") != "djvu":
            continue
        repo = str(record.get("Repo") or "")
        revision = str(revisions.get(repo) or "")
        if repo and revision:
            records.append({"repo": repo, "path": relative_path(record), "revision": revision,
                            "key": f"{repo}\0{relative_path(record)}"})
    records.sort(key=lambda x: (x["repo"], x["path"]))
    if a.limit:
        records = records[:a.limit]
    old = {x.get("key"): x for x in read_manifest(token).get("files", []) if isinstance(x, dict)}
    # Repository revisions are repository-wide and change for unrelated files.
    # Keep an existing derivative stable; source-key additions are the weekly
    # queue. A future explicit force/revision audit can rebuild changed files.
    pending = [x for x in records if x["key"] not in old]
    print(f"djvu_sources={len(records)} pending={len(pending)}")
    if not a.apply:
        print("report-only; pass --apply to convert and publish")
        return 0
    results = dict(old)
    uploads = []
    with tempfile.TemporaryDirectory(prefix="djvu-weekly-") as temp:
        root = Path(temp)
        for number, item in enumerate(pending):
            work = root / str(number); work.mkdir()
            source = work / "source.djvu"
            target = work / "document.pdf"
            try:
                download(source_url(item["repo"], item["revision"], item["path"]), source, token)
                source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                contract = bucket_conversion_contract(item["repo"], item["path"], "djvu", source.stat().st_size)
                convert_file({"extension": "djvu", "profile": contract[0], "reader_mode": "pdf",
                              "output_name": "document.pdf", "path": item["path"]}, source, target, work)
                validate_djvu_pdf(target, work)
                pdf_bytes = target.read_bytes(); pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
                object_id = hashlib.sha256(f"{item['key']}\0{item['revision']}\0{pdf_sha}".encode()).hexdigest()[:32]
                remote = f"derived/weekly-djvu/{object_id}/document.pdf"
                manifest_path = f"derived/weekly-djvu/{object_id}/derived-manifest.json"
                entry = {"key": item["key"], "repo": item["repo"], "source_path": item["path"],
                         "source_extension": "djvu", "source_revision": item["revision"],
                         "source_sha256": source_sha, "derived_sha256": pdf_sha,
                         "derived_bytes": len(pdf_bytes), "profile": PROFILE,
                         "path": remote, "new_path": remote, "manifest_path": manifest_path,
                         "bucket": BUCKET, "generation": "weekly-djvu"}
                uploads.extend([(str(target), remote),
                                ((json.dumps({"version": 1, "kind": "pdf-derived-source", **entry},
                                             ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(), manifest_path)])
                results[item["key"]] = entry
                print(f"ready {item['repo']}/{item['path']}")
            except Exception as error:
                print(f"failed {item['repo']}/{item['path']}: {type(error).__name__}: {error}")
    if uploads:
        batch_bucket_files(BUCKET, add=uploads, token=token)
    payload = {"version": 1, "kind": "pdf-derived-source-index", "generation": "weekly-djvu",
               "files": [results[key] for key in sorted(results)]}
    batch_bucket_files(BUCKET, add=[((json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(), MANIFEST)], token=token)
    print(f"published_objects={len(uploads)} entries={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
