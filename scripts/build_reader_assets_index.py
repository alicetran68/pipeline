#!/usr/bin/env python3
"""Build the compact search sidecar from a Reader Assets manifest."""

import argparse
import gzip
import json
from pathlib import Path

try:
    from .reader_assets import SPREADSHEET_EXTENSIONS, load_json, validate_manifest
    from .pdf_assets import PDF_DECISION_PROFILE, PDF_PROFILE
    from . import shared
except ImportError:
    from reader_assets import SPREADSHEET_EXTENSIONS, load_json, validate_manifest
    from pdf_assets import PDF_DECISION_PROFILE, PDF_PROFILE
    import shared

STATUS = {"ready": 2, "failed": 4}
MODE = {"pdf": "p", "epub": "e", "foliate": "e", "docx": "d", "html": "h", "text": "t", "markdown": "k", "image": "i", "audio": "a", "video": "v"}


def build_index(manifest: dict, pdf_manifest: dict | None = None,
                ocr_manifest: dict | None = None) -> dict:
    files = {}
    spreadsheet_keys = {
        key for key, entry in manifest["files"].items()
        if entry.get("status") == "ready"
        and entry.get("source_extension") in SPREADSHEET_EXTENSIONS
        and entry.get("path", "").endswith("/page-manifest.json")
    }
    for key, entry in manifest["files"].items():
        status = entry.get("status")
        if status not in STATUS:
            continue
        compact = {"s": STATUS[status]}
        if status == "ready":
            compact.update({"m": MODE[entry["reader_mode"]], "p": entry["path"]})
            if entry.get("bucket") and not (entry["reader_mode"] == "pdf" and entry["path"].endswith("/document.pdf")):
                compact["b"] = entry["bucket"]
            if entry.get("chapter_manifest"):
                compact["c"] = entry["chapter_manifest"]
                if entry.get("chapter_bucket"):
                    compact["cb"] = entry["chapter_bucket"]
            if entry.get("fallback_path"):
                compact["f"] = entry["fallback_path"]
        files[key] = compact
    for key, entry in (pdf_manifest or {}).get("files", {}).items():
        if key in spreadsheet_keys:
            continue
        if entry.get("status") != "ready":
            continue
        if (entry.get("strategy") != "sampled-webp"
                or entry.get("render_profile") != PDF_PROFILE
                or entry.get("decision_profile") != PDF_DECISION_PROFILE):
            continue
        path = entry.get("path") or entry.get("page_manifest", {}).get("path")
        if path:
            files[key] = {**files.get(key, {}), **shared.pdf_pages_sidecar_entry(path)}
    for key, entry in (ocr_manifest or {}).get("files", {}).items():
        if key in spreadsheet_keys:
            continue
        # Rendering may finish before recognition (or recognition may fail).
        # Preserve its complete Reader stream during unrelated sidecar rebuilds.
        page_path = (entry.get("page_manifest") or {}).get("path")
        if (entry.get("status") in {"rendered", "failed"} and page_path
                and entry.get("render_manifest")):
            files[key] = {**files.get(key, {}), **shared.pdf_pages_sidecar_entry(page_path)}
        if entry.get("status") != "ready" or not isinstance(entry.get("ocr_manifest"), str):
            continue
        merged = shared.merge_pdf_ocr_sidecar_entry(files.get(key), entry)
        if merged:
            files[key] = merged
    return {"v": 1, "f": dict(sorted(files.items()))}


def encode_index(manifest: dict, pdf_manifest: dict | None = None,
                 ocr_manifest: dict | None = None) -> bytes:
    payload = json.dumps(build_index(manifest, pdf_manifest, ocr_manifest), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return gzip.compress(payload, compresslevel=9, mtime=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pdf-manifest", type=Path)
    parser.add_argument("--ocr-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/reader_assets.json.gz"))
    args = parser.parse_args()
    manifest = validate_manifest(load_json(args.manifest))
    pdf_manifest = load_json(args.pdf_manifest) if args.pdf_manifest else None
    ocr_manifest = load_json(args.ocr_manifest) if args.ocr_manifest else None
    index = build_index(manifest, pdf_manifest, ocr_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encode_index(manifest, pdf_manifest, ocr_manifest))
    print(f"wrote {len(index['f'])} reader asset mapping(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
