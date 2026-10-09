#!/usr/bin/env python3
"""Attach native text to an existing Lin Yizhang page stream without rerendering it.

Run for one verified source at a time; immutable objects are uploaded before the
OCR registry and Reader sidecar are updated. No image recognition is involved.
"""

import argparse
import gzip
import json
from pathlib import Path
import tempfile

import pymupdf
from huggingface_hub import HfApi

try:
    from . import lin_pdf_text, ocr_layout, pdf_ocr, pdf_ocr_stages as stages
    from . import publish_pdf_ocr_assets as publication, reader_assets, shared
    from .run_pdf_ocr import source_path
except ImportError:
    import lin_pdf_text, ocr_layout, pdf_ocr, pdf_ocr_stages as stages
    import publish_pdf_ocr_assets as publication, reader_assets, shared
    from run_pdf_ocr import source_path


def build(entry: dict, manifest: dict, source: Path, bundle: Path) -> dict:
    if (not lin_pdf_text.applies(entry) or entry.get("status") != "ready" or not entry.get("page_manifest")
            or entry.get("source_kind") != "generated"
            or not str(entry.get("reader_assets_path") or "").endswith("/gbk-font-repair-v1/document.pdf")):
        raise ValueError("source has no eligible complete Lin Yizhang page stream")
    stages.validate_render(entry, manifest)
    digest, size = shared.hash_file(source)
    if digest != entry["source_sha256"] or size != entry["source_bytes"]:
        raise ValueError("PDF source changed since page rendering")
    language = pdf_ocr.detect_language(entry["key"])
    backend = "rapidocr_onnxruntime" if language in {"ch", "en"} else "paddle_onnxruntime"
    options = stages.layout_options({}, entry["key"])
    profile = stages.recognition_identity({**entry, "ocr_language": language, "ocr_backend": backend}, options)
    root = stages.root_for(digest, entry["key"], entry["render_manifest"]["sha256"] + profile)
    pages, texts, nonempty = [], [], 0
    with pymupdf.open(source) as document:
        if len(document) != entry["page_count"]:
            raise ValueError("PDF page count changed since rendering")
        for number, rendered in enumerate(manifest["pages"], 1):
            text = lin_pdf_text.extract(document, number)
            payload = pdf_ocr.page_payload(number, text["width"], text["height"], text["blocks"], "native")
            payload.update(ocr_layout.arrange(payload["blocks"], payload["width"], payload["height"], {}))
            if payload["text"].strip():
                nonempty += 1
            out = bundle / root / "ocr" / f"page-{number:06d}.json.gz"
            pdf_ocr.write_gzip_json(out, payload)
            page = {**rendered, "source": "native", "width": text["width"], "height": text["height"],
                    "chars": len(payload["text"]), "text": payload["text"],
                    "text_spans": payload["text_spans"], "layout": payload["layout"]}
            stages.set_page_meta(page, "o", stages.metadata(out, bundle))
            pages.append(page)
            texts.append({"page": number, "text": payload["text"],
                          "layout": payload["layout"], "text_spans": payload["text_spans"]})
    if nonempty < entry["page_count"] * .8:
        raise ValueError(f"native text coverage too low: {nonempty}/{entry['page_count']} pages")
    book_path = bundle / root / "ocr" / "book-text.json.gz"
    pdf_ocr.write_gzip_json(book_path, {"version": 2, "kind": "pdf-book-text", "complete": True,
                                        "source_sha256": digest, "page_count": entry["page_count"],
                                        "offset_unit": "unicode-codepoint", "profile": profile,
                                        "language": language, "ocr_version": "native-mupdf-1.28.2",
                                        "pages": texts})
    manifest_path = bundle / root / "ocr-manifest.json"
    pdf_ocr.write_json(manifest_path, {"version": 1, "kind": "pdf-ocr", "complete": True,
                                       "profile": profile, "engine": "PyMuPDF native text 1.28.2",
                                       "language": language, "ocr_version": "native-mupdf-1.28.2",
                                       "backend": "native", "source_sha256": digest, "source_bytes": size,
                                       "source_revision": entry.get("source_revision", ""),
                                       "classification": "native-text", "page_count": entry["page_count"],
                                       "pages": pages, "book_text": stages.metadata(book_path, bundle),
                                       "page_manifest": entry["page_manifest"]})
    meta = stages.metadata(manifest_path, bundle)
    return {**stages.public_item(entry), "status": "ready", "profile": profile,
            "language": language, "ocr_version": "native-mupdf-1.28.2", "backend": "native",
            "stream": True, "ocr_manifest": meta["path"], "ocr_manifest_sha256": meta["sha256"],
            "ocr_manifest_bytes": meta["bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="full Teachers source path ending in .pdf")
    parser.add_argument("--assets-repo", default="vomebook/Reader-Assets")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without uploading")
    args = parser.parse_args()
    key = reader_assets.asset_key("VoiceOfML/Teachers", args.path)
    api = HfApi()
    revision = stages.retry(lambda: api.repo_info(repo_id=args.assets_repo, repo_type="dataset")).sha
    rendered = stages.load_registry(api, args.assets_repo, stages.RENDER_REGISTRY, revision)["files"]
    entry = rendered.get(key)
    if not isinstance(entry, dict) or not lin_pdf_text.applies(entry):
        raise ValueError("no rendered Lin Yizhang PDF for this path")
    previous = stages.load_registry(api, args.assets_repo, publication.OCR_MANIFEST_NAME, revision)["files"].get(key, {})
    if (previous.get("status") == "ready" and previous.get("ocr_version") == "native-mupdf-1.28.2"
            and stages.same_source(previous, entry) and previous.get("page_manifest") == entry.get("page_manifest")):
        print("current complete text layer already published")
        return
    manifest = json.loads(stages.read_object(entry["render_manifest"], "/render-manifest.json"))
    source = source_path(entry)
    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp)
        result = build(entry, manifest, source, bundle)
        if not args.dry_run:
            result["processing_roots"] = stages.upload_objects(bundle)
            publication.publish(api, args.assets_repo, [result])
        print(json.dumps({"path": args.path, "pages": result["page_count"],
                          "status": "validated" if args.dry_run else "published"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
