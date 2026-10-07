#!/usr/bin/env python3
"""Build the new-only compact Reader sidecar from v2 canonical indexes."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files


ASSETS_BUCKET = "vomebook/reader-assets-v2"
PDF_BUCKET = "vomebook/pdf-pages-v2"
SIDECAR_PATH = "reader-index/reader_assets.json.gz"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-bucket", default=ASSETS_BUCKET)
    parser.add_argument("--pdf-bucket", default=PDF_BUCKET)
    parser.add_argument("--derived-manifest", default="reader-index/derived_pdf_manifest.json")
    parser.add_argument("--output", type=Path, default=Path("output/reader_assets_v2.json.gz"))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def read_json(fs: HfFileSystem, bucket: str, path: str) -> dict:
    with fs.open(f"hf://buckets/{bucket}/{path}", "rb") as stream:
        value = json.loads(stream.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid index: {bucket}/{path}")
    return value


def add_index(files: dict, payload: dict, mode_for: dict[str, str], bucket: str,
              *, chapters: bool = False) -> None:
    for entry in payload.get("files", []):
        if not isinstance(entry, dict) or not entry.get("key"):
            continue
        extension = str(entry.get("extension") or entry.get("source_extension") or "").lower()
        mode = mode_for.get(extension)
        if not mode:
            continue
        path = entry.get("manifest") if chapters else entry.get("object")
        if not isinstance(path, str):
            continue
        compact = {"s": 2, "m": mode, "p": path, "b": bucket}
        if chapters:
            compact.update({"c": path, "cb": bucket})
            if entry.get("fallback"):
                compact["f"] = entry["fallback"]
        files[entry["key"]] = compact


def build(fs: HfFileSystem, assets_bucket: str, pdf_bucket: str, derived_path: str) -> dict:
    files = {}
    for extension, mode in (("txt", "t"), ("md", "k"), ("markdown", "k"), ("vcf", "t"), ("ini", "t")):
        payload = read_json(fs, assets_bucket, "documents/text/index.json")
        if extension == "txt":
            selected = [x for x in payload.get("files", []) if x.get("extension") == extension]
            add_index(files, {"files": selected}, {extension: mode}, assets_bucket)
        elif extension == "md":
            selected = [x for x in payload.get("files", []) if x.get("extension") in {"md", "markdown"}]
            add_index(files, {"files": selected}, {"md": mode, "markdown": mode}, assets_bucket)
        else:
            selected = [x for x in payload.get("files", []) if x.get("extension") == extension]
            add_index(files, {"files": selected}, {extension: mode}, assets_bucket)
    for extension in ("html", "htm", "mht", "mhtml"):
        try:
            payload = read_json(fs, assets_bucket, f"documents/web/{extension}/index.json")
        except FileNotFoundError:
            continue
        add_index(files, payload, {extension: "h"}, assets_bucket)
    for extension in ("doc", "docx", "odt", "rtf"):
        payload = read_json(fs, assets_bucket, f"documents/office/{extension}/index.json")
        add_index(files, payload, {extension: "d" if extension in {"doc", "docx"} else "h"}, assets_bucket)
    for extension in ("xls", "xlsx", "csv", "ods"):
        try:
            payload = read_json(fs, assets_bucket, f"documents/spreadsheet/{extension}/index.json")
        except FileNotFoundError:
            continue
        add_index(files, payload, {extension: "h"}, assets_bucket)
    for extension in ("ppt", "pptx", "pps", "wps", "ps"):
        payload = read_json(fs, assets_bucket, f"documents/pdf/{extension}/index.json")
        add_index(files, payload, {extension: "p"}, assets_bucket)
    for extension in ("epub", "mobi", "azw3", "fb2", "chm"):
        payload = read_json(fs, assets_bucket, f"chapters/ebook/{extension}/index.json")
        add_index(files, payload, {extension: "e"}, assets_bucket, chapters=True)
    for kind, mode in (("audio", "a"), ("video", "v"), ("swf", "f")):
        payload = read_json(fs, assets_bucket, f"media/{kind}/index.json")
        add_index(files, payload, {kind: mode}, assets_bucket)

    derived = read_json(fs, pdf_bucket, derived_path)
    for entry in derived.get("files", []):
        if not isinstance(entry, dict) or not entry.get("key") or not entry.get("new_path"):
            continue
        files[entry["key"]] = {"s": 2, "m": "p", "p": entry["new_path"], "b": pdf_bucket}
    return {"v": 1, "f": dict(sorted(files.items()))}


def main() -> int:
    args = arguments()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    fs = HfFileSystem(token=token)
    index = build(fs, args.assets_bucket, args.pdf_bucket, args.derived_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(json.dumps(index, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode(), compresslevel=9, mtime=0)
    args.output.write_bytes(payload)
    print(f"mappings={len(index['f'])} bytes={len(payload)} output={args.output}")
    if args.apply:
        batch_bucket_files(args.assets_bucket, add=[(str(args.output), SIDECAR_PATH)], token=token)
        print(f"published={args.assets_bucket}/{SIDECAR_PATH}")
    else:
        print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
