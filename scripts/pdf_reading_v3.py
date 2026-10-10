#!/usr/bin/env python3
"""Offline v3 bundle construction; no uploads, pointer changes or model calls."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

try:
    from . import pdf_ocr, pdf_text_layer as text, shared
except ImportError:
    import pdf_ocr
    import pdf_text_layer as text
    import shared

BUCKETS = {shared.READER_ASSETS_BUCKET, shared.PDF_PAGES_BUCKET, shared.PDF_OCR_INPUT_BUCKET}
PARTITION_SIZE = 128
SEARCH_PARTITION_PAGES = 32
SEARCH_PARTITION_BYTES = 512 * 1024


def resource(value):
    if not isinstance(value, dict) or value.get("bucket") not in BUCKETS:
        raise ValueError("invalid qualified resource bucket")
    path = value.get("path")
    if (not isinstance(path, str) or not (path.startswith(("objects/", "derived/")) or
            re.fullmatch(r"documents/pdf/[a-z0-9_-]+/[0-9a-f]{64}/document\.pdf", path))
            or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(ord(c) < 32 for c in path) or "?" in path or "#" in path):
        raise ValueError("invalid qualified resource path")
    text.sha(value.get("sha256"))
    if type(value.get("bytes")) is not int or value["bytes"] < 1:
        raise ValueError("invalid qualified resource size")
    if value.get("role") not in {"runtime", "processing", "review", "provenance"}:
        raise ValueError("invalid qualified resource role")
    return value


def verified_read(ref, read):
    resource(ref)
    raw = read(ref)
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("v3 resource checksum mismatch")
    return raw


def decode(raw):
    return json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)


def stored(path, bundle, payload, *, compressed=False, role="runtime"):
    writer = pdf_ocr.write_gzip_json if compressed else pdf_ocr.write_json
    digest, size = writer(path, payload)
    return {"bucket": shared.PDF_PAGES_BUCKET, "path": path.relative_to(bundle).as_posix(),
            "sha256": digest, "bytes": size, "role": role}


def partition_for(manifest, page):
    """Locate one demanded page without fetching any preceding partitions."""
    if type(page) is not int or not 1 <= page <= manifest["page_count"]:
        raise ValueError("page outside manifest")
    partitions = manifest["partitions"]
    index = bisect.bisect_right([p["start"] for p in partitions], page) - 1
    if index < 0 or page > partitions[index]["end"]:
        return None
    return partitions[index]


def backfill_text(manifest, read, bundle, options=None):
    """Upgrade verified existing raw page objects, without rendering or recognition."""
    try:
        from . import ocr_layout
    except ImportError:
        import ocr_layout
    source = text.sha(manifest["source_sha256"])
    count = manifest.get("page_count")
    pages = manifest.get("pages", [])
    if (manifest.get("kind") != "pdf-ocr" or manifest.get("complete") is not True
            or type(count) is not int or count < 1
            or [p.get("p") for p in pages] != list(range(1, count + 1))):
        raise ValueError("backfill requires complete OCR page coverage")
    options = ocr_layout.validate_options(options)
    if options.get("rotation", 0):
        raise ValueError("backfill cannot rotate already positioned raw text")
    layers = []
    for page in pages:
        ref = resource({"bucket": shared.PDF_PAGES_BUCKET, "path": page["o"],
                        "sha256": page["os"], "bytes": page["ob"], "role": "provenance"})
        payload = decode(verified_read(ref, read))
        if (payload.get("kind") != "pdf-ocr-page" or payload.get("page") != page["p"]
                or payload.get("source") != page["source"]):
            raise ValueError("backfill raw page identity mismatch")
        config = manifest.get("layout_options", {})
        page_options = {**config.get("default", {}), **config.get("pages", {}).get(str(page["p"]), {}), **options}
        # Existing blocks have already been transformed to original coordinates.
        arrange_options = {k: v for k, v in page_options.items() if k != "rotation"}
        if (options or "text_spans" not in payload or payload["source"] == "native") and payload["blocks"]:
            payload.update(ocr_layout.arrange(payload["blocks"], payload["width"], payload["height"], arrange_options,
                                              include_writing_modes=True))
        layers.append(text.from_page(payload, source, manifest.get("language", "und"), ref,
                                     layout_options=page_options))
    identity = text.digest({"raw_manifest": manifest, "options": options, "text_layer": 2})
    root = Path("objects") / source[:2] / source / identity[:16]
    return build_text_bundle(layers, pages, root, bundle)


def validate_text_manifest(manifest):
    if (manifest.get("version") != 1 or manifest.get("kind") != "pdf-text-layer-index"
            or manifest.get("complete") is not True or manifest.get("revision") not in {"raw", "effective"}):
        raise ValueError("invalid text layer index")
    text.sha(manifest["source_sha256"])
    text.sha(manifest["generation"])
    count = manifest.get("page_count")
    if type(count) is not int or count < 1:
        raise ValueError("invalid text layer page count")
    cursor = 1
    for partition in manifest["partitions"]:
        if (type(partition["start"]) is not int or type(partition["end"]) is not int
                or partition["start"] != cursor or not cursor <= partition["end"] <= count):
            raise ValueError("noncontiguous text partitions")
        resource(partition["resource"])
        cursor = partition["end"] + 1
    if cursor != count + 1:
        raise ValueError("incomplete text partition coverage")
    resource(manifest["review"])
    if manifest.get("book_text"):
        resource(manifest["book_text"])
    if "search_partitions" in manifest:
        cursor = 1
        for part in manifest["search_partitions"]:
            if (type(part["start"]) is not int or type(part["end"]) is not int
                    or part["start"] != cursor or not cursor <= part["end"] <= count
                    or part["end"] - part["start"] >= SEARCH_PARTITION_PAGES):
                raise ValueError("invalid search partition coverage")
            resource(part["resource"])
            cursor = part["end"] + 1
        if cursor != count + 1:
            raise ValueError("incomplete search partition coverage")
    if manifest.get("parent"):
        resource(manifest["parent"])
    return manifest


def build_text_bundle(layers, raw_pages, root, bundle, *, parent=None):
    if not layers or [p["page"] for p in layers] != list(range(1, len(layers) + 1)):
        raise ValueError("text layers must cover every page")
    if len(raw_pages) != len(layers):
        raise ValueError("raw page coverage mismatch")
    source = layers[0]["source_sha256"]
    entries, tasks = [], []
    for layer, page in zip(layers, raw_pages):
        text.validate(layer)
        if layer["source_sha256"] != source or page["p"] != layer["page"]:
            raise ValueError("text layer source mismatch")
        out = bundle / root / "text" / f"page-{layer['page']:06d}.json.gz"
        ref = stored(out, bundle, layer, compressed=True)
        entries.append({"page": layer["page"], "page_identity": layer["page_identity"],
                        "generation": layer["generation"], "resource": ref,
                        "processing": layer["processing"], "quality": layer["quality"]})
        if layer["provenance"].get("raw_resource"):
            entries[-1]["raw_resource"] = resource(layer["provenance"]["raw_resource"])
        evidence = []
        if page.get("i"):
            evidence.append(resource({"bucket": page.get("ibucket", shared.PDF_OCR_INPUT_BUCKET),
                                      "path": page["i"], "sha256": page["is"], "bytes": page["ib"],
                                      "role": "review"}))
        task = text.review_task(layer, ref, evidence)
        if task:
            tasks.append(task)
    generation = text.digest([p["generation"] for p in entries])
    corrected = any(layer["revision"] == "accepted" for layer in layers)
    revision = "effective" if corrected else "raw"
    quality = "partially-reviewed" if corrected else "unreviewed"
    search_partitions, chunk, chunk_bytes = [], [], 0

    def flush_search():
        start, end = chunk[0]["page"], chunk[-1]["page"]
        ref = stored(bundle / root / "text" / f"search-{start:06d}-{end:06d}.json.gz", bundle,
                     {"version": 1, "kind": "pdf-search-text-partition", "source_sha256": source,
                      "generation": generation, "offset_unit": "unicode-codepoint",
                      "start": start, "end": end, "pages": chunk}, compressed=True)
        search_partitions.append({"start": start, "end": end, "resource": ref})

    for layer in layers:
        page = {"page": layer["page"], "text": layer["text"], "text_generation": layer["generation"]}
        size = len(json.dumps(page, ensure_ascii=False).encode("utf-8"))
        if chunk and (len(chunk) >= SEARCH_PARTITION_PAGES or chunk_bytes + size > SEARCH_PARTITION_BYTES):
            flush_search()
            chunk, chunk_bytes = [], 0
        chunk.append(page)
        chunk_bytes += size
    flush_search()
    book_text = stored(bundle / root / "text" / "book-text.json.gz", bundle,
                       {"version": 2, "kind": "pdf-book-text", "complete": True,
                        "source_sha256": source, "page_count": len(layers),
                        "generation": generation, "revision": revision, "quality": quality,
                        "offset_unit": "unicode-codepoint",
                        "pages": [{"page": p["page"], "text": p["text"],
                                    "text_generation": p["generation"], "quality": p["quality"],
                                    "layout": {"version": "text-layer-v1", "offset_unit": "unicode-codepoint",
                                               "writing_mode": "auto", "mapping_precision": "block"},
                                   "text_spans": [{"start": r["start"], "end": r["end"], "box": r["box"],
                                                   "region_id": r["id"], "precision": r["mapping_precision"]}
                                                  for r in p["regions"]]} for p in layers]}, compressed=True)
    review = stored(bundle / root / "text-review-manifest.json", bundle,
                    {"version": 1, "kind": "pdf-text-review", "source_sha256": source,
                     "generation": generation, "tasks": tasks}, role="review")
    partitions = []
    for offset in range(0, len(entries), PARTITION_SIZE):
        chunk = entries[offset:offset + PARTITION_SIZE]
        start, end = chunk[0]["page"], chunk[-1]["page"]
        ref = stored(bundle / root / "text" / f"partition-{start:06d}-{end:06d}-manifest.json", bundle,
                     {"version": 1, "kind": "pdf-text-partition", "source_sha256": source,
                      "generation": generation, "start": start, "end": end, "pages": chunk})
        partitions.append({"start": start, "end": end, "resource": ref})
    manifest = {"version": 1, "kind": "pdf-text-layer-index", "complete": True,
                "source_sha256": source, "generation": generation, "page_count": len(layers),
                "revision": revision, "quality": quality, "offset_unit": "unicode-codepoint",
                "partitions": partitions, "review": review, "review_count": len(tasks), "book_text": book_text,
                "search_partitions": search_partitions}
    if parent is not None:
        manifest["parent"] = resource(parent)
    validate_text_manifest(manifest)
    return stored(bundle / root / "text-layer-manifest.json", bundle, manifest)


def verify_text_bundle(ref, read, source_sha256, page_count, *, verify_evidence=False, page_geometries=None):
    manifest = validate_text_manifest(decode(verified_read(ref, read)))
    if manifest["source_sha256"] != source_sha256 or manifest["page_count"] != page_count:
        raise ValueError("text layer index source mismatch")
    generations = []
    layers = []
    page_resources = {}
    for partition in manifest["partitions"]:
        data = decode(verified_read(partition["resource"], read))
        if (data.get("version") != 1 or data.get("kind") != "pdf-text-partition"
                or data.get("source_sha256") != source_sha256 or data.get("generation") != manifest["generation"]
                or data.get("start") != partition["start"] or data.get("end") != partition["end"]
                or [p["page"] for p in data["pages"]] != list(range(partition["start"], partition["end"] + 1))):
            raise ValueError("invalid text partition identity")
        for entry in data["pages"]:
            layer = text.validate(decode(verified_read(entry["resource"], read)))
            if (layer["page"] != entry["page"] or layer["source_sha256"] != source_sha256
                    or layer["generation"] != entry["generation"] or layer["page_identity"] != entry["page_identity"]
                    or layer["processing"] != entry["processing"] or layer["quality"] != entry["quality"]):
                raise ValueError("text page identity mismatch")
            raw_ref = layer["provenance"].get("raw_resource")
            if entry.get("raw_resource") != raw_ref:
                raise ValueError("raw page provenance missing from text partition")
            if raw_ref:
                resource(raw_ref)
            generations.append(layer["generation"])
            layers.append(layer)
            page_resources[layer["page"]] = entry["resource"]
            if page_geometries is not None:
                geometry = page_geometries[layer["page"] - 1]
                aspect = layer["geometry"]["width"] / layer["geometry"]["height"]
                if abs(aspect / (geometry["width"] / geometry["height"]) - 1) > .01:
                    raise ValueError("text layer/PDF page geometry mismatch")
    if text.digest(generations) != manifest["generation"]:
        raise ValueError("text index generation mismatch")
    for part in manifest.get("search_partitions", []):
        data = decode(verified_read(part["resource"], read))
        expected = [{"page": p["page"], "text": p["text"], "text_generation": p["generation"]}
                    for p in layers[part["start"] - 1:part["end"]]]
        if (data.get("version") != 1 or data.get("kind") != "pdf-search-text-partition"
                or data.get("source_sha256") != source_sha256 or data.get("generation") != manifest["generation"]
                or data.get("offset_unit") != "unicode-codepoint" or data.get("start") != part["start"]
                or data.get("end") != part["end"] or data.get("pages") != expected):
            raise ValueError("search partition differs from effective page text")
    corrected = any(p["revision"] == "accepted" for p in layers)
    if (manifest["revision"] != ("effective" if corrected else "raw")
            or manifest["quality"] != ("partially-reviewed" if corrected else "unreviewed")):
        raise ValueError("text index quality claim mismatch")
    if manifest.get("book_text"):
        index = decode(verified_read(manifest["book_text"], read))
        expected_spans = [[{"start": r["start"], "end": r["end"], "box": r["box"],
                            "region_id": r["id"], "precision": r["mapping_precision"]}
                           for r in layer["regions"]] for layer in layers]
        if (index.get("kind") != "pdf-book-text" or index.get("version") != 2
                or index.get("complete") is not True or index.get("source_sha256") != source_sha256
                or index.get("generation") != manifest["generation"]
                or index.get("page_count") != page_count or index.get("revision") != manifest["revision"]
                or [p.get("page") for p in index.get("pages", [])] != list(range(1, page_count + 1))
                or any(p["text"] != layer["text"] or p["text_generation"] != layer["generation"]
                       or p["quality"] != layer["quality"] or p["text_spans"] != spans
                       for p, layer, spans in zip(index["pages"], layers, expected_spans))):
            raise ValueError("search index differs from effective page text")
    review = decode(verified_read(manifest["review"], read))
    if (review.get("kind") != "pdf-text-review" or review.get("version") != 1
            or review.get("source_sha256") != source_sha256 or review.get("generation") != manifest["generation"]
            or len(review.get("tasks", [])) != manifest["review_count"]):
        raise ValueError("invalid review manifest")
    known = {p["page"]: p for p in layers}
    ids = set()
    for task in review["tasks"]:
        layer = known.get(task.get("page"))
        if (not layer or task.get("base_generation") != layer["generation"]
                or task.get("raw_sha256") != layer["raw_sha256"]
                or task.get("page_identity") != layer["page_identity"]
                or task.get("id") in ids or text.digest({k: v for k, v in task.items() if k != "id"}) != task.get("id")):
            raise ValueError("invalid review task identity")
        if (task.get("source_sha256") != source_sha256 or task.get("issues") != layer["review_flags"]
                or task.get("region_ids") != [r["id"] for r in layer["regions"]]):
            raise ValueError("review task scope mismatch")
        ids.add(task["id"])
        resource(task["text_layer"])
        if task["text_layer"] != page_resources[task["page"]]:
            raise ValueError("review task references a different text page")
        for evidence in task["evidence"]:
            resource(evidence)
            if verify_evidence:
                verified_read(evidence, read)
    return manifest


def repartition_text_bundle(ref, read, bundle):
    """Refresh search packaging without changing any raw or accepted page text."""
    original = decode(verified_read(ref, read))
    original = verify_text_bundle(ref, read, original["source_sha256"], original["page_count"])
    review = decode(verified_read(original["review"], read))
    evidence = {task["page"]: task["evidence"] for task in review["tasks"]}
    layers, pages = [], []
    for part in original["partitions"]:
        for entry in decode(verified_read(part["resource"], read))["pages"]:
            layer = decode(verified_read(entry["resource"], read))
            layers.append(layer)
            page = {"p": layer["page"]}
            for item in evidence.get(layer["page"], []):
                if item["bucket"] == shared.PDF_OCR_INPUT_BUCKET and item["path"].endswith(".png"):
                    page.update(i=item["path"], ibucket=item["bucket"], ib=item["bytes"], **{"is": item["sha256"]})
                    break
            pages.append(page)
    source = original["source_sha256"]
    identity = text.digest({"parent": ref, "search_packaging": 1})
    root = Path("objects") / source[:2] / source / identity[:16]
    return build_text_bundle(layers, pages, root, bundle, parent=ref)


def accept_text_bundle(ref, proposals, read, bundle, *, actor):
    """Offline trusted acceptance producing a new complete text/search generation."""
    source_manifest = decode(verified_read(ref, read))
    source = source_manifest["source_sha256"]
    verified = verify_text_bundle(ref, read, source, source_manifest["page_count"])
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("correction proposals required")
    by_page = {}
    for proposal in proposals:
        page = proposal.get("page")
        if (type(page) is not int or not 1 <= page <= verified["page_count"] or page in by_page):
            raise ValueError("duplicate or invalid correction page")
        by_page[page] = proposal
    review = decode(verified_read(verified["review"], read))
    evidence_by_page = {t["page"]: t["evidence"] for t in review["tasks"]}
    layers, pages = [], []
    for partition in verified["partitions"]:
        data = decode(verified_read(partition["resource"], read))
        for entry in data["pages"]:
            layer = decode(verified_read(entry["resource"], read))
            if layer["page"] in by_page:
                layer = text.accept_proposal(layer, by_page[layer["page"]], actor=actor)
            layers.append(layer)
            page = {"p": layer["page"]}
            for evidence in evidence_by_page.get(layer["page"], []):
                if evidence["bucket"] == shared.PDF_OCR_INPUT_BUCKET and evidence["path"].endswith(".png"):
                    page.update(i=evidence["path"], ibucket=evidence["bucket"],
                                ib=evidence["bytes"])
                    page["is"] = evidence["sha256"]
                    break
            pages.append(page)
    identity = text.digest({"parent": ref, "proposals": proposals, "actor": actor, "acceptance": 2})
    root = Path("objects") / source[:2] / source / identity[:16]
    return build_text_bundle(layers, pages, root, bundle, parent=ref)


def page_map(pdf_bytes, source_sha256):
    """Use the PDF engine's real page geometry, not an invented crop/rotation."""
    import pymupdf
    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.needs_pass or len(document) == 0:
            raise ValueError("unreadable primary PDF")
        for index, page in enumerate(document, 1):
            geometry = {"page": index, "media_box": list(page.mediabox), "crop_box": list(page.cropbox),
                        "rotation": page.rotation, "width": page.rect.width, "height": page.rect.height}
            try:
                images = page.get_image_info()
                geometry["classification"] = ("bitonal-image" if any(i.get("bpc") == 1 for i in images) else
                                              "native-vector" if page.get_text().strip() else
                                              "raster-or-mixed" if images else "vector-or-empty")
            except Exception:
                geometry["classification"] = "unknown"
            geometry["identity"] = text.digest({"source_sha256": source_sha256, **geometry})
            pages.append(geometry)
    return {"version": 1, "kind": "pdf-page-map", "source_sha256": source_sha256, "pages": pages}


def generate_previews(spec, read, bundle, *, start=1, end=None, dpi=150, max_pixels=50000000):
    """Generate resumable page ranges, choosing lossless PNG for protected pages."""
    import pymupdf
    import io
    from PIL import Image
    if type(dpi) is not int or not 72 <= dpi <= 600 or type(max_pixels) is not int or max_pixels < 4096:
        raise ValueError("invalid preview render budget")
    source = text.sha(spec["source_sha256"])
    raw = verified_read(spec["primary"], read)
    mapping = page_map(raw, source)
    count = len(mapping["pages"])
    end = count if end is None else end
    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= count:
        raise ValueError("invalid preview generation range")
    recipe = text.digest({"source": source, "primary": spec["primary"], "dpi": dpi,
                          "max_pixels": max_pixels, "webp_quality": 85, "policy": "mixed-preview-v1"})
    root = Path("objects") / source[:2] / source / recipe[:16]
    generated = []
    with pymupdf.open(stream=raw, filetype="pdf") as document:
        for number in range(start, end + 1):
            geometry = mapping["pages"][number - 1]
            if geometry["classification"] == "unknown":
                raise ValueError("unknown page inspection requires retry before stream generation")
            protected = geometry["classification"] == "bitonal-image"
            requested_dpi = max(dpi, 300) if protected else dpi
            scale = min(requested_dpi / 72, math.sqrt(max_pixels / (geometry["width"] * geometry["height"])))
            # Pixmap dimensions are rounded upwards by the PDF engine.
            while True:
                pixmap = document[number - 1].get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                                       colorspace=pymupdf.csRGB, alpha=False)
                if pixmap.width * pixmap.height <= max_pixels:
                    break
                scale *= .99
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            stream = io.BytesIO()
            codec = "png" if protected else "webp"
            image.save(stream, "PNG" if protected else "WEBP", **({} if protected else {"quality": 85, "method": 6}))
            path = bundle / root / "preview" / f"page-{number:06d}.{codec}"
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = stream.getvalue()
            path.write_bytes(encoded)
            ref = {"bucket": shared.PDF_PAGES_BUCKET, "path": path.relative_to(bundle).as_posix(),
                   "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "role": "runtime"}
            generated.append({"page": number, "page_identity": geometry["identity"], "codec": codec,
                              "width": image.width, "height": image.height, "resource": ref,
                              "recipe": {"requested_dpi": requested_dpi, "actual_scale": scale,
                                         "max_pixels": max_pixels, "role": "preview-not-archival"}})
            image.close()
    existing = {entry["page"]: entry for entry in spec.get("previews", [])}
    for entry in generated:
        existing[entry["page"]] = entry
    return {**spec, "previews": [existing[p] for p in sorted(existing)]}


def validate_reading_manifest(manifest):
    if (manifest.get("version") != 3 or manifest.get("kind") != "pdf-reading"
            or not isinstance(manifest.get("source_key"), str) or not manifest["source_key"]
            or manifest.get("primary", {}).get("kind") != "pdf"
            or manifest["primary"].get("status") != "ready"):
        raise ValueError("invalid reading manifest")
    text.sha(manifest["source_sha256"])
    text.sha(manifest["generation"])
    count = manifest.get("page_count")
    if type(count) is not int or count < 1:
        raise ValueError("invalid reading page count")
    resource(manifest["page_map"])
    primary = resource(manifest["primary"]["resource"])
    if (primary["bucket"] == shared.PDF_OCR_INPUT_BUCKET or primary["role"] != "runtime"
            or not primary["path"].endswith(".pdf")):
        raise ValueError("invalid primary reading document")
    preview = manifest["preview"]
    if (preview.get("page_count") != count or type(preview.get("complete")) is not bool
            or type(preview.get("ready_pages")) is not int or not 0 <= preview["ready_pages"] <= count):
        raise ValueError("invalid preview state")
    previous = 0
    covered = 0
    for partition in preview["partitions"]:
        start, end = partition["start"], partition["end"]
        if (type(start) is not int or type(end) is not int or not previous < start <= end <= count
                or (start - 1) % PARTITION_SIZE or end != min(count, start + PARTITION_SIZE - 1)
                or type(partition.get("complete")) is not bool):
            raise ValueError("invalid reading partition")
        resource(partition["resource"])
        if preview["complete"] and (start != previous + 1 or not partition["complete"]):
            raise ValueError("incomplete advertised preview")
        previous = end
        covered += end - start + 1
    if preview["complete"] and (covered != count or preview["ready_pages"] != count):
        raise ValueError("incomplete advertised preview")
    if manifest.get("text_layer"):
        resource(manifest["text_layer"])
    expected = {"document": "ready", "preview": "ready" if preview["complete"] else "pending",
                "text": "ready" if manifest.get("text_layer") else "pending", "searchable_pdf": "skipped"}
    if any(manifest["components"].get(k) != v for k, v in expected.items()):
        raise ValueError("invalid reading component state")
    return manifest


def build_reading(spec, read, bundle):
    """Build a candidate with a full PDF and explicitly complete/partial previews."""
    source = text.sha(spec["source_sha256"])
    primary = resource(spec["primary"])
    if (primary["bucket"] == shared.PDF_OCR_INPUT_BUCKET or primary["role"] != "runtime"
            or not primary["path"].endswith(".pdf")):
        raise ValueError("primary must be a reading-bucket PDF")
    mapping = page_map(verified_read(primary, read), source)
    count = len(mapping["pages"])
    previews = spec.get("previews", [])
    seen = set()
    for entry in previews:
        number = entry.get("page")
        if type(number) is not int or not 1 <= number <= count or number in seen:
            raise ValueError("duplicate or invalid preview page")
        seen.add(number)
        geometry = mapping["pages"][number - 1]
        if entry.get("page_identity") != geometry["identity"]:
            raise ValueError("preview page map mismatch")
        ref = resource(entry["resource"])
        if ref["bucket"] == shared.PDF_OCR_INPUT_BUCKET or ref["role"] != "runtime":
            raise ValueError("processing inputs are not published previews")
        codec = entry.get("codec")
        if codec not in {"png", "webp", "jpeg"}:
            raise ValueError("unsupported preview codec")
        if geometry["classification"] in {"bitonal-image", "unknown"} and codec != "png":
            raise ValueError("protected pages require lossless preview encoding")
        from PIL import Image
        import io
        with Image.open(io.BytesIO(verified_read(ref, read))) as image:
            image.load()
            if image.format.lower() != codec or list(image.size) != [entry["width"], entry["height"]]:
                raise ValueError("preview codec or dimensions mismatch")
            if abs((image.width / image.height) / (geometry["width"] / geometry["height"]) - 1) > .01:
                raise ValueError("preview aspect ratio mismatch")
    complete = seen == set(range(1, count + 1))
    if spec.get("require_complete_preview", False) and not complete:
        raise ValueError("incomplete whole-book preview")
    text_ref = spec.get("text_layer")
    if text_ref:
        verify_text_bundle(text_ref, read, source, count, page_geometries=mapping["pages"])
    recipe = text.digest({"version": 3, "spec": spec, "page_map": mapping})
    root = Path("objects") / source[:2] / source / recipe[:16]
    map_ref = stored(bundle / root / "page-map.json.gz", bundle, mapping, compressed=True)
    partitions = []
    # Fixed logical ranges permit direct access even when a range is incomplete.
    by_page = {p["page"]: p for p in previews}
    for start in range(1, count + 1, PARTITION_SIZE):
        end = min(count, start + PARTITION_SIZE - 1)
        entries = [by_page[p] for p in range(start, end + 1) if p in by_page]
        if not entries:
            continue
        ref = stored(bundle / root / "preview" / f"partition-{start:06d}-{end:06d}-manifest.json", bundle,
                     {"version": 3, "kind": "pdf-preview-partition", "source_sha256": source,
                      "page_map_sha256": map_ref["sha256"], "start": start, "end": end, "pages": entries})
        partitions.append({"start": start, "end": end, "complete": len(entries) == end - start + 1,
                           "resource": ref})
    manifest = {"version": 3, "kind": "pdf-reading", "source_sha256": source,
                "source_key": spec["source_key"], "generation": recipe,
                "page_count": count, "page_map": map_ref,
                "primary": {"kind": "pdf", "status": "ready", "resource": primary},
                "preview": {"page_count": count, "complete": complete, "ready_pages": len(seen),
                            "partitions": partitions},
                "text_layer": text_ref,
                "components": {"document": "ready", "preview": "ready" if complete else "pending",
                               "text": "ready" if text_ref else "pending", "correction": "pending",
                               "searchable_pdf": "skipped"},
                "provenance": {"builder": "pdf-reading-v3-v1", "recipe_sha256": recipe}}
    validate_reading_manifest(manifest)
    ref = stored(bundle / root / "reading-manifest.json", bundle, manifest)
    return ref, manifest


def verify_reading(ref, read):
    manifest = validate_reading_manifest(decode(verified_read(ref, read)))
    source = text.sha(manifest["source_sha256"])
    text.sha(manifest["generation"])
    primary = resource(manifest["primary"]["resource"])
    if (primary["bucket"] == shared.PDF_OCR_INPUT_BUCKET or primary["role"] != "runtime"
            or not primary["path"].endswith(".pdf")):
        raise ValueError("invalid primary reading document")
    expected = page_map(verified_read(primary, read), source)
    mapping = decode(verified_read(manifest["page_map"], read))
    if mapping != expected or len(mapping["pages"]) != manifest["page_count"]:
        raise ValueError("reading page map differs from PDF")
    preview = manifest["preview"]
    count = manifest["page_count"]
    if type(preview.get("complete")) is not bool or preview.get("page_count") != count:
        raise ValueError("invalid preview completeness")
    seen, previous = set(), 0
    from PIL import Image
    import io
    for partition in preview["partitions"]:
        start, end = partition["start"], partition["end"]
        if (type(start) is not int or type(end) is not int or not previous < start <= end <= count
                or (start - 1) % PARTITION_SIZE or end != min(count, start + PARTITION_SIZE - 1)):
            raise ValueError("invalid preview partition range")
        previous = end
        data = decode(verified_read(partition["resource"], read))
        if (data.get("version") != 3 or data.get("kind") != "pdf-preview-partition"
                or data.get("source_sha256") != source or data.get("page_map_sha256") != manifest["page_map"]["sha256"]
                or data.get("start") != start or data.get("end") != end):
            raise ValueError("preview partition identity mismatch")
        numbers = [p["page"] for p in data["pages"]]
        if (numbers != sorted(set(numbers)) or any(type(p) is not int or not start <= p <= end for p in numbers)
                or type(partition.get("complete")) is not bool
                or partition["complete"] != (numbers == list(range(start, end + 1)))):
            raise ValueError("preview partition page coverage mismatch")
        for page in data["pages"]:
            geometry = mapping["pages"][page["page"] - 1]
            if page["page_identity"] != geometry["identity"]:
                raise ValueError("preview/PDF page identity mismatch")
            media = resource(page["resource"])
            if media["bucket"] == shared.PDF_OCR_INPUT_BUCKET or media["role"] != "runtime":
                raise ValueError("invalid runtime preview")
            if page["codec"] not in {"png", "webp", "jpeg"}:
                raise ValueError("unsupported preview codec")
            if geometry["classification"] in {"bitonal-image", "unknown"} and page["codec"] != "png":
                raise ValueError("protected page has lossy preview")
            with Image.open(io.BytesIO(verified_read(media, read))) as image:
                image.load()
                if image.format.lower() != page["codec"] or list(image.size) != [page["width"], page["height"]]:
                    raise ValueError("preview format mismatch")
                if abs((image.width / image.height) / (geometry["width"] / geometry["height"]) - 1) > .01:
                    raise ValueError("preview geometry mismatch")
            seen.add(page["page"])
    if (preview["ready_pages"] != len(seen)
            or preview["complete"] != (seen == set(range(1, count + 1)))):
        raise ValueError("reading stream completeness mismatch")
    if manifest.get("text_layer"):
        verify_text_bundle(manifest["text_layer"], read, source, count, page_geometries=mapping["pages"])
    components = manifest["components"]
    if (components.get("document") != "ready"
            or components.get("preview") != ("ready" if preview["complete"] else "pending")
            or components.get("text") != ("ready" if manifest.get("text_layer") else "pending")
            or components.get("searchable_pdf") != "skipped"):
        raise ValueError("reading component state mismatch")
    return manifest


def local_reader(directory):
    directory = directory.resolve()
    def read(ref):
        resource(ref)
        target = (directory / ref["bucket"] / PurePosixPath(ref["path"])).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("local resource escapes bucket mirror")
        return target.read_bytes()
    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "backfill-text", "generate-stream", "accept-text"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--manifest", type=Path, help="OCR manifest or qualified reading resource JSON")
    parser.add_argument("--mirror", type=Path, required=True,
                        help="Verified inputs under <mirror>/<bucket>/<path>")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pixels", type=int, default=50000000)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--actor")
    args = parser.parse_args()
    read = local_reader(args.mirror)
    if args.command == "accept-text":
        if args.manifest is None or args.proposals is None or args.output is None or not args.actor:
            parser.error("accept-text requires --manifest, --proposals, --output and --actor")
        ref = accept_text_bundle(json.loads(args.manifest.read_text()), json.loads(args.proposals.read_text()),
                                 read, args.output, actor=args.actor)
        print(json.dumps({"text_layer": ref}, sort_keys=True))
    elif args.command == "generate-stream":
        if args.spec is None or args.output is None:
            parser.error("generate-stream requires --spec and --output")
        spec = generate_previews(json.loads(args.spec.read_text()), read, args.output,
                                 start=args.start, end=args.end, dpi=args.dpi, max_pixels=args.max_pixels)
        pdf_ocr.write_json(args.output / "reading-spec.json", spec)
        print(json.dumps({"spec": str(args.output / "reading-spec.json"), "ready_pages": len(spec["previews"])}))
    elif args.command == "build":
        if args.spec is None or args.output is None:
            parser.error("build requires --spec and --output")
        ref, manifest = build_reading(json.loads(args.spec.read_text()), read, args.output)
        print(json.dumps({"resource": ref, "components": manifest["components"]}, sort_keys=True))
    elif args.command == "backfill-text":
        if args.manifest is None or args.output is None:
            parser.error("backfill-text requires --manifest and --output")
        ref = backfill_text(json.loads(args.manifest.read_text()), read, args.output)
        print(json.dumps({"text_layer": ref}, sort_keys=True))
    else:
        if args.manifest is None:
            parser.error("verify requires --manifest")
        manifest = verify_reading(json.loads(args.manifest.read_text()), read)
        print(json.dumps({"generation": manifest["generation"], "components": manifest["components"]}, sort_keys=True))


if __name__ == "__main__":
    main()
