#!/usr/bin/env python3
"""Build incremental DOC/DOCX/ODT/RTF Reader document streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
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
CHECKPOINT_SIZE = 25
LIBREOFFICE_TIMEOUT_SECONDS = 600


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


def run_libreoffice(command: list[str], profile: Path) -> None:
    """Run one conversion in an isolated profile and clean up hung children."""
    profile.mkdir(parents=True, exist_ok=True)
    full_command = [command[0], f"-env:UserInstallation={profile.as_uri()}", *command[1:]]
    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=LIBREOFFICE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise subprocess.TimeoutExpired(full_command, LIBREOFFICE_TIMEOUT_SECONDS) from error
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, full_command, stdout, stderr)


def publish_checkpoint(bucket: str, category: str, root: Path, merged: dict,
                       failures: dict, uploads: dict[str, str], token: str | None,
                       apply: bool) -> None:
    index_file = root / "index.json"
    index_file.write_text(json.dumps({
        "version": 1,
        "kind": "office-document-stream-index",
        "files": [merged[key] for key in sorted(merged)],
        "failures": [failures[key] for key in sorted(failures) if key not in merged],
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    uploads[f"{category}/index.json"] = str(index_file)
    if apply:
        batch_bucket_files(bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token)
        uploads.clear()


def converted_output(directory: Path, extension: str) -> Path:
    """Find LibreOffice's output even when it normalizes the input basename."""
    candidates = sorted(directory.glob(f"*.{extension}"))
    if len(candidates) == 1:
        return candidates[0]
    expected = directory / f"source.{extension}"
    if expected.is_file():
        return expected
    if not candidates:
        raise FileNotFoundError(f"LibreOffice produced no .{extension} output in {directory}")
    raise RuntimeError(f"LibreOffice produced multiple .{extension} outputs: {candidates}")


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
        run_libreoffice(["libreoffice", "--headless", "--convert-to", "docx", "--outdir", str(out), str(source)], work / "libreoffice-profile")
        output, name = converted_output(out, "docx"), "document.docx"
    else:
        out = work / "html"
        out.mkdir()
        run_libreoffice(["libreoffice", "--headless", "--convert-to", "html", "--outdir", str(out), str(source)], work / "libreoffice-profile")
        generated = converted_output(out, "html")
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
        if ext == "all":
            raise ValueError("office worker must process one extension")
        category = f"documents/office/{ext}"
        old = index(a.bucket, category, token)
        merged = {e.get("key"): e for e in old.get("files", [])
                  if isinstance(e, dict) and e.get("key")}
        failures = {e.get("key"): e for e in old.get("failures", [])
                    if isinstance(e, dict) and e.get("key")}
        checkpoint_count = 0
        for number, item in enumerate(selected):
            if a.limit and len(entries) >= a.limit: break
            key = f"{item['repo']}\0{item['path']}"
            if merged.get(key, {}).get("source_revision") == item["revision"]:
                continue
            try: entry, files = build(item, root / str(number), token, a.bucket)
            except Exception as error:
                failures[key] = {
                    "key": key, "repo": item["repo"], "path": item["path"],
                    "source_revision": item["revision"],
                    "error": f"{type(error).__name__}: {error}"[:1000],
                }
                checkpoint_count += 1
                print(f"failed: {item['repo']}/{item['path']}: {type(error).__name__}: {error}")
            else:
                entries.append(entry)
                merged[key] = entry
                failures.pop(key, None)
                uploads.update(files)
                checkpoint_count += 1
            if checkpoint_count >= CHECKPOINT_SIZE:
                publish_checkpoint(a.bucket, category, root, merged, failures, uploads, token, a.apply)
                checkpoint_count = 0
        if checkpoint_count or entries or failures:
            publish_checkpoint(a.bucket, category, root, merged, failures, uploads, token, a.apply)
        if not entries and not failures:
            print("no pending office sources")
            return 0
        print(f"processed {len(entries)} office stream(s), failures={len(failures)}")
        if not a.apply:
            print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__": raise SystemExit(main())
