#!/usr/bin/env python3
"""Convert selected non-OCR source formats into static PDF Reader streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files

try:
    from .convert_reader_assets import validate_pdf_content
    from .pilot_office_stream import converted_output, run_libreoffice
    from .reader_assets import decode_search_payload, relative_path, source_url
except ImportError:
    from convert_reader_assets import validate_pdf_content
    from pilot_office_stream import converted_output, run_libreoffice
    from reader_assets import decode_search_payload, relative_path, source_url


BUCKET = "vomebook/reader-assets-v2"
EXTENSIONS = {"ppt", "pptx", "pps", "wps", "ps"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--extension", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
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
            data = json.loads(stream.read().decode("utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("files"), list) else {"files": []}
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return {"files": []}


def records(path: Path, revisions: dict, extension: str) -> list[dict]:
    result = []
    for record in decode_search_payload(json.loads(path.read_text(encoding="utf-8"))):
        ext = str(record.get("Extension") or "").lower().lstrip(".")
        if ext not in EXTENSIONS or (extension != "all" and ext != extension):
            continue
        repo = str(record.get("Repo") or "")
        revision = str(revisions.get(repo) or "")
        if repo and revision:
            result.append({"repo": repo, "path": relative_path(record), "extension": ext,
                           "revision": revision})
    return sorted(result, key=lambda item: (item["extension"], item["repo"], item["path"]))


def build(item: dict, work: Path, token: str | None) -> tuple[dict, dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"source.{item['extension']}"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output_dir = work / "pdf"
    output_dir.mkdir()
    if item["extension"] == "ps":
        output = output_dir / "document.pdf"
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
                        f"-sOutputFile={output}", str(source)], check=True, timeout=600)
    else:
        run_libreoffice(["libreoffice", "--headless", "--convert-to", "pdf",
                         "--outdir", str(output_dir), str(source)], work / "libreoffice-profile")
        output = converted_output(output_dir, "pdf")
    validation = work / "validation"
    validation.mkdir()
    validate_pdf_content(output, validation)
    output_bytes = output.read_bytes()
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    root = f"documents/pdf/{item['extension']}/{source_hash}"
    object_path = f"{root}/document.pdf"
    manifest_path = f"{root}/document-manifest.json"
    manifest = work / "document-manifest.json"
    manifest.write_text(json.dumps({
        "kind": "static-pdf-document-stream", "version": 1,
        "source_extension": item["extension"], "source_sha256": source_hash,
        "path": "document.pdf", "bytes": len(output_bytes), "sha256": output_hash,
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    entry = {"key": f"{item['repo']}\0{item['path']}", "repo": item["repo"],
             "path": item["path"], "extension": item["extension"],
             "source_revision": item["revision"], "source_sha256": source_hash,
             "mode": "pdf", "bucket": BUCKET, "object": object_path,
             "manifest": manifest_path, "bytes": len(output_bytes), "sha256": output_hash}
    return entry, {object_path: str(output), manifest_path: str(manifest)}


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    extension = args.extension.lower().lstrip(".")
    if extension != "all" and extension not in EXTENSIONS:
        raise ValueError(f"unsupported static PDF extension: {extension}")
    token = os.environ.get("HF_TOKEN")
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    selected = records(args.search_data, revisions, extension)
    entries, uploads, failures = [], {}, {}
    with tempfile.TemporaryDirectory(prefix="reader-static-pdf-") as directory:
        root = Path(directory)
        by_category = {}
        for number, item in enumerate(selected):
            category = f"documents/pdf/{item['extension']}"
            old = existing_index(args.bucket, token, category)
            key = f"{item['repo']}\0{item['path']}"
            handled = list(old.get("files", []))
            if not args.retry_failures:
                handled += list(old.get("failures", []))
            if any(e.get("key") == key and e.get("source_revision") == item["revision"]
                   for e in handled if isinstance(e, dict)):
                continue
            try:
                entry, files = build(item, root / str(number), token)
            except Exception as error:
                failures[key] = {"key": key, "repo": item["repo"], "path": item["path"],
                                 "extension": item["extension"], "source_revision": item["revision"],
                                 "error": f"{type(error).__name__}: {error}"[:1000]}
                print(f"failed: {item['repo']}/{item['path']}: {type(error).__name__}: {error}")
                continue
            entries.append(entry); by_category.setdefault(category, []).append(entry); uploads.update(files)
            if args.limit and len(entries) >= args.limit:
                break
        for failure in failures.values():
            by_category.setdefault(f"documents/pdf/{failure['extension']}", [])
        if not entries and not failures:
            print("no pending static PDF sources")
            return 0
        for category, added in by_category.items():
            old = existing_index(args.bucket, token, category)
            merged = {e.get("key"): e for e in old.get("files", []) if isinstance(e, dict)}
            merged.update({e["key"]: e for e in added})
            old_failures = {e.get("key"): e for e in old.get("failures", []) if isinstance(e, dict)}
            old_failures.update({k: v for k, v in failures.items() if f"documents/pdf/{v['extension']}" == category})
            old_failures = {k: v for k, v in old_failures.items() if k not in merged}
            index_file = root / category.replace("/", "_") / "index.json"
            index_file.parent.mkdir(parents=True, exist_ok=True)
            index_file.write_text(json.dumps({"version": 1, "kind": "static-pdf-stream-index",
                                               "files": [merged[k] for k in sorted(merged)],
                                               "failures": [old_failures[k] for k in sorted(old_failures)]},
                                              ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            uploads[f"{category}/index.json"] = str(index_file)
        print(f"processed={len(entries)} failures={len(failures)} objects={len(uploads)}")
        if args.apply:
            batch_bucket_files(args.bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token)
            print(f"published={len(uploads)}")
        else:
            print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
