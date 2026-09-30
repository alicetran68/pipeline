#!/usr/bin/env python3
"""Durable PDF rendering followed by independent, resumable image-only OCR.

Only the rendering stage downloads PDFs or uses Poppler/cjxl. The OCR stage
uses checksummed PNG objects; neither Reader WebP nor a PDF is an OCR input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import quote

import httpx
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import get_session, hf_raise_for_status

try:
    from . import pdf_ocr, pdf_assets, lin_pdf_text, pdf_render_schedule, plan_pdf_ocr, publish_pdf_ocr_assets as publication, shared, ocr_layout
    from .run_pdf_ocr import source_path, _bucket_retry_delay
    from .reader_bucket import index_path, publish_bytes, publish_json, read_json as read_bucket_json, update_lifecycle_consumer
except ImportError:
    import pdf_ocr, pdf_assets, lin_pdf_text, plan_pdf_ocr, publish_pdf_ocr_assets as publication, shared, ocr_layout
    import pdf_render_schedule
    from run_pdf_ocr import source_path, _bucket_retry_delay
    from reader_bucket import index_path, publish_bytes, publish_json, read_json as read_bucket_json, update_lifecycle_consumer

RENDER_REGISTRY = "pdf_render_manifest.json"
PROGRESS_REGISTRY = "pdf_ocr_progress.json"
RENDER_PROGRESS_REGISTRY = "pdf_render_progress.json"
RENDER_RANGE_PAGES = 250
RENDER_RANGE_THRESHOLD = 500
BUCKET = f"hf://buckets/{shared.PDF_PAGES_BUCKET}"
OCR_INPUT_BUCKET = f"hf://buckets/{shared.PDF_OCR_INPUT_BUCKET}"
SMALL_RENDER_MAX_SOURCE_BYTES = 100 * 1024 * 1024
RENDER_SAMPLE_TIMEOUT = 60


def render_profile() -> str:
    return (f"pdf-render-v2-text-png-dpi-{pdf_ocr.OCR_DPI}-maxpix-{pdf_ocr.MAX_PAGE_PIXELS}"
            f"-webp-{pdf_ocr.WEBP_QUALITY}-{pdf_ocr.WEBP_MAX_DIMENSION}"
            f"-native-{pdf_ocr.MIN_NATIVE_PAGE_CHARS}-jxl-{int(pdf_ocr.JXL_ENABLED)}"
            f"-{pdf_ocr.JXL_DISTANCE:g}-{pdf_ocr.JXL_EFFORT}-reader-source-pixels-v2-clean-webp-80-native-stream-v1")


def root_for(source_sha: str, key: str, identity: str) -> Path:
    suffix = hashlib.sha256(f"{key}\0{identity}".encode()).hexdigest()[:16]
    return Path("objects") / source_sha[:2] / source_sha / suffix


def metadata(path: Path, bundle: Path) -> dict:
    sha, size = shared.hash_file(path)
    return {"path": path.relative_to(bundle).as_posix(), "sha256": sha, "bytes": size}


def public_item(item: dict) -> dict:
    return {k: v for k, v in item.items()
            if not k.startswith("_") and k not in {"probe", "page_chars", "bundle_root"}}


def retry(operation):
    for attempt in range(8):
        try:
            return operation()
        except HfHubHTTPError as exc:
            if shared.hf_status_code(exc) not in {408, 429, 500, 502, 503, 504} or attempt == 7:
                raise
            delay = _bucket_retry_delay(exc, attempt)
        except (httpx.TransportError, ConnectionError, OSError):
            if attempt == 7:
                raise
            delay = min(300, 5 * 2 ** attempt)
        print(f"temporary object transfer failure; retry in {delay}s", flush=True)
        time.sleep(delay)


def upload_objects(bundle: Path) -> None:
    """Sync only one book's immutable prefix, never recursively list the whole bucket."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for root in sorted((bundle / "objects").glob("*/*/*")):
        if root.is_dir():
            relative = root.relative_to(bundle).as_posix()
            retry(lambda: api.sync_bucket(
                str(root), f"{BUCKET}/{relative}",
                include=["pages/*.webp", "ocr/**", "page-manifest.json", "render-manifest.json",
                         "render-range-*.json"], quiet=True))
            retry(lambda: api.sync_bucket(
                str(root), f"{OCR_INPUT_BUCKET}/{relative}",
                include=["ocr-input/**", "pages/*.jxl"], quiet=True))


def read_object(meta: dict, suffix: str | None = None) -> bytes:
    path = pdf_ocr.validate_ocr_object_path(meta["path"], suffix)
    bucket = meta.get("bucket") or (
        shared.PDF_OCR_INPUT_BUCKET if "/ocr-input/" in path or path.endswith(".jxl")
        else shared.PDF_PAGES_BUCKET)
    def download():
        # Public object resolve avoids per-page bucket_info/paths-info API calls.
        response = get_session().get(
            f"https://huggingface.co/buckets/{bucket}/resolve/{quote(path, safe='/')}",
            follow_redirects=True, timeout=120)
        hf_raise_for_status(response)
        return response.content
    data = retry(download)
    if len(data) != meta["bytes"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise ValueError(f"object checksum mismatch: {path}")
    return data


def page_meta(page: dict, field: str) -> dict:
    return {"path": page[field], "sha256": page[field + "s"], "bytes": page[field + "b"],
            **({"bucket": page[field + "bucket"]} if page.get(field + "bucket") else {})}


def set_page_meta(page: dict, field: str, meta: dict) -> None:
    page.update({field: meta["path"], field + "s": meta["sha256"], field + "b": meta["bytes"]})


def load_registry(api, repo, name, revision=None):
    if type(api) is HfApi:
        try:
            return read_bucket_json(index_path(name), os.environ.get("HF_TOKEN"))
        except (FileNotFoundError, OSError, ValueError):
            return {"version": 1, "files": {}}
    try:
        path = retry(lambda: api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=name,
                                                revision=revision))
    except HfHubHTTPError as exc:
        if shared.hf_status_code(exc) == 404:
            return {"version": 1, "files": {}}
        raise
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError(f"invalid {name}")
    return data


def save_registry(api, repo, name, updates, merge=None, publish_streams=False):
    if not updates:
        return
    for attempt in range(20):
        info = retry(lambda: api.repo_info(repo_id=repo, repo_type="dataset"))
        data = load_registry(api, repo, name, info.sha)
        for key, value in updates.items():
            data["files"][key] = merge(data["files"].get(key), value) if merge else value
        content = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=content)]
        if publish_streams:
            ocr_state = load_registry(api, repo, publication.OCR_MANIFEST_NAME, info.sha)
            try:
                sidecar_path = retry(lambda: api.hf_hub_download(
                    repo_id=repo, repo_type="dataset", filename=publication.SIDECAR_NAME, revision=info.sha))
                with gzip.open(sidecar_path, "rt", encoding="utf-8") as stream:
                    sidecar = json.load(stream)
            except HfHubHTTPError as exc:
                if shared.hf_status_code(exc) != 404:
                    raise
                sidecar = {"v": 1, "f": {}}
            if sidecar.get("v") != 1 or not isinstance(sidecar.get("f"), dict):
                raise ValueError("invalid Reader sidecar")
            for key, value in updates.items():
                if value.get("status") == "ready":
                    previous = ocr_state["files"].get(key, {})
                    if previous.get("status") != "ready":
                        ocr_state["files"][key] = {**value, "status": "rendered"}
                    elif (value.get("range_status") == "failed" and value.get("classification") == "native-text"
                          and value.get("page_manifest") and same_source(previous, value)
                          and previous.get("source_sha256") == value.get("source_sha256")):
                        ocr_state["files"][key] = {**previous, "page_manifest": value["page_manifest"],
                                                   "render_manifest": value["render_manifest"],
                                                   "range_status": "failed", "classification": "native-text"}
                    entry = dict(sidecar["f"].get(key) or {})
                    if value.get("page_manifest"):
                        entry.update(shared.pdf_pages_sidecar_entry(value["page_manifest"]["path"]))
                        sidecar["f"][key] = entry
            operations.append(CommitOperationAdd(path_in_repo=publication.SIDECAR_NAME,
                                                  path_or_fileobj=publication.encode_sidecar(sidecar)))
            operations.append(CommitOperationAdd(
                path_in_repo=publication.OCR_MANIFEST_NAME,
                path_or_fileobj=(json.dumps(ocr_state, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()))
        try:
            api.create_commit(repo_id=repo, repo_type="dataset", parent_commit=info.sha,
                              commit_message=f"Publish {name}",
                              operations=operations)
            token = os.environ.get("HF_TOKEN")
            if token and type(api) is HfApi:
                publish_json(index_path(name), data, token)
                if publish_streams:
                    publish_json(index_path(publication.OCR_MANIFEST_NAME), ocr_state, token)
                    publish_bytes(index_path(publication.SIDECAR_NAME), publication.encode_sidecar(sidecar), token)
                if name == RENDER_REGISTRY:
                    for key, value in updates.items():
                        update_lifecycle_consumer(
                            key, "render", "done" if value.get("status") in {"ready", "skipped"} else "failed", token)
            return
        except HfHubHTTPError as exc:
            if not shared.is_retryable_hf_status(shared.hf_status_code(exc)) or attempt == 19:
                raise
            time.sleep(shared.hf_retry_delay(attempt))


def same_source(entry, item):
    return bool(entry and entry.get("source_revision") == item.get("source_revision")
                and entry.get("source_kind") == item.get("source_kind")
                and entry.get("reader_assets_path") == item.get("reader_assets_path"))


def render_partition_matches(item, partition):
    if partition == "all":
        return True
    size = int(item.get("source_bytes") or 0)
    if not size:
        return partition == "small"
    is_small = size < SMALL_RENDER_MAX_SOURCE_BYTES
    return is_small if partition == "small" else not is_small


def pending_render(records, rendered, ocr, retry_failed=False, partition="all"):
    pending = []
    for item in records:
        if not render_partition_matches(item, partition):
            continue
        previous = rendered.get(item["key"])
        if same_source(previous, item) and previous.get("render_profile") == render_profile():
            if previous.get("status") in {"ready", "skipped"}:
                if not item.get("force_image_render") or previous.get("image_rendered"):
                    continue
            if previous.get("status") == "failed" and not retry_failed:
                continue
        pending.append(item)
    # Repeatedly failed PDFs must not monopolize the first checkpoint and
    # prevent untouched books from entering later batches.
    return sorted(pending, key=lambda item: (
        rendered.get(item["key"], {}).get("status") == "failed",
        item["repo"], item["path"], item["source_kind"],
    ))


def range_id(start, end):
    return f"{start:06d}-{end:06d}"


def render_ranges(book):
    count = book["page_count"]
    size = RENDER_RANGE_PAGES if count > RENDER_RANGE_THRESHOLD else count
    return [(start, min(start + size - 1, count)) for start in range(1, count + 1, size)]


def range_identity(book):
    return {field: book[field] for field in ("key", "source_sha256", "source_revision",
                                           "source_kind", "render_profile", "page_count")}


def validate_range(book, start, end, descriptor):
    name = f"/render-range-{range_id(start, end)}.json"
    root = root_for(book["source_sha256"], book["key"], book["render_profile"])
    if descriptor.get("path") != (root / name.lstrip("/")).as_posix():
        raise ValueError("render range path mismatch")
    payload = json.loads(read_object(descriptor, name))
    if (payload.get("version") != 1 or payload.get("kind") != "pdf-render-range"
            or any(payload.get(field) != value for field, value in range_identity(book).items())
            or payload.get("start") != start or payload.get("end") != end):
        raise ValueError("render range identity mismatch")
    pages = payload.get("pages")
    if not isinstance(pages, list) or [p.get("p") for p in pages] != list(range(start, end + 1)):
        raise ValueError("incomplete render range")
    if (book.get("force_image_render") and book["probe"]["classification"] == "native-text"
            and (not payload.get("image_rendered") or any("w" not in page for page in pages))):
        raise ValueError("native render range lacks required page images")
    validate_render({**book, "page_manifest": None}, {**book, "version": 1, "kind": "pdf-render",
                                                    "complete": True, "page_manifest": None,
                                                    "image_rendered": payload.get("image_rendered", False), "pages": pages,
                                                    "classification": book["probe"]["classification"]},
                    page_numbers=range(start, end + 1))
    return pages


def estimate_render_cost(item, source, progress):
    book = {**item, "render_profile": render_profile()}
    previous = progress.get(book["key"], {})
    image_rendered = bool(book.get("force_image_render") or book.get("probe", {}).get("classification") != "native-text")
    cost = pdf_render_schedule.history_cost(range_identity(book), previous, image_rendered)
    if cost is not None:
        return cost
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        queue = root / "book.json"
        output = root / "cost"
        pdf_ocr.write_json(queue, book)
        command = [sys.executable, str(Path(__file__).resolve()), "measure-render", "--queue", str(queue),
                   "--source", str(source.resolve()), "--output", str(output)]
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True) as process:
            try:
                process.wait(timeout=RENDER_SAMPLE_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Terminate Poppler children too; sampling never owns a production worker.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        result = output / "render-cost.json"
        if result.is_file():
            return json.loads(result.read_text(encoding="utf-8"))
    return {"seconds_per_page": pdf_render_schedule.DEFAULT_PAGE_SECONDS,
            "setup_seconds": 0, "source": "fallback", "samples": 0}


def sample_render_cost(book, source, output=None):
    count = book["page_count"]
    samples = []
    for number in sorted({1, (count + 1) // 2, count}):
        try:
            with tempfile.TemporaryDirectory() as temp:
                result = render_book({**book, "start": number, "end": number}, source, Path(temp))
                timing = result["timing"]
                if (pdf_render_schedule.finite_seconds(timing.get("page_seconds"), positive=True) is not None
                        and pdf_render_schedule.finite_seconds(timing.get("setup_seconds")) is not None):
                    samples.append(timing)
        except Exception as exc:
            print(f"render timing sample unavailable: {type(exc).__name__}", flush=True)
        cost = ({"seconds_per_page": max(s["page_seconds"] for s in samples),
                 "setup_seconds": max(s["setup_seconds"] for s in samples),
                 "source": "sample", "samples": len(samples)} if samples else {
                     "seconds_per_page": pdf_render_schedule.DEFAULT_PAGE_SECONDS,
                     "setup_seconds": 0, "source": "fallback", "samples": 0})
        if output is not None:
            pending = output / "render-cost.pending.json"
            pdf_ocr.write_json(pending, cost)
            os.replace(pending, output / "render-cost.json")
    return cost


def plan_render_ranges(queue, progress, target_seconds=None):
    if target_seconds is not None and pdf_render_schedule.finite_seconds(target_seconds, positive=True) is None:
        raise ValueError("render target seconds must be finite and positive")
    books = [{**item, "render_profile": render_profile()}
              for shard in queue["shards"] for item in shard["records"]]
    tasks, saved = [], {}
    for book in books:
        prior = progress.get(book["key"], {})
        existing = prior.get("ranges", {}) if all(prior.get(k) == v for k, v in range_identity(book).items()) else {}
        completed = []
        for key, descriptor in existing.items():
            bounds = pdf_render_schedule.parse_range(key, book["page_count"])
            if bounds is None:
                continue
            start, end = bounds
            try:
                validate_range(book, start, end, descriptor)
                saved.setdefault(book["key"], {})[key] = descriptor
                completed.append(bounds)
            except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
                pass
        size = RENDER_RANGE_PAGES if book["page_count"] > RENDER_RANGE_THRESHOLD else book["page_count"]
        if target_seconds is not None:
            size = pdf_render_schedule.range_size(book.get("_render_cost", {}), target_seconds, size)
        for start, end in pdf_render_schedule.missing_ranges(book["page_count"], completed, size):
            tasks.append({**book, "start": start, "end": end})
    if target_seconds is None:
        count = min(256, len(tasks), max(1, math.ceil(sum(t["end"] - t["start"] + 1 for t in tasks) / 500)))
        shards = shared.weighted_shards(tasks, count, weight=lambda t: t["end"] - t["start"] + 1,
                                        order=lambda t: (-(t["end"] - t["start"] + 1), t["key"], t["start"])) if tasks else []
    else:
        shards = pdf_render_schedule.balance(tasks, target_seconds)
    return {**queue, "books": books, "saved_ranges": saved,
             **({"target_render_seconds": target_seconds} if target_seconds is not None else {}),
             "shard_count": len(shards), "shard_ids": list(range(len(shards))),
             "shards": [{"index": i, "page_count": sum(t["end"] - t["start"] + 1 for t in shard),
                         **({"estimated_seconds": round(sum(map(pdf_render_schedule.task_seconds, shard)), 2)}
                            if target_seconds is not None else {}),
                         "records": ([{key: task[key] for key in (*range_identity(task), "start", "end")}
                                      | {"_render_cost": task.get("_render_cost", {})} for task in shard]
                                     if target_seconds is not None else shard)} for i, shard in enumerate(shards)]}


def expand_render_tasks(queue, records):
    books = {book["key"]: book for book in queue.get("books", [])}
    tasks = []
    for record in records:
        book = books.get(record["key"])
        if book is None:
            raise ValueError("render task refers to an unknown book")
        if any(record.get(key) != value for key, value in range_identity(book).items()):
            raise ValueError("render task identity differs from book")
        tasks.append({**book, **record})
    return tasks


def render_book(item: dict, source: Path, bundle: Path) -> dict:
    started = time.perf_counter()
    source_sha, source_bytes = shared.hash_file(source)
    if item.get("source_sha256") and item["source_sha256"] != source_sha:
        raise ValueError("PDF source changed after planning")
    lin_native = lin_pdf_text.applies(item)
    probe = item.get("probe") or (lin_pdf_text.probe(source) if lin_native else pdf_ocr.probe_pdf(source))
    if "start" in item and not 1 <= item["start"] <= item["end"] <= probe["page_count"]:
        raise ValueError("invalid render page range")
    base = {**public_item(item), "source_sha256": source_sha, "source_bytes": source_bytes,
            "render_profile": render_profile(), "profile": pdf_ocr.asset_profile(),
            "page_count": probe["page_count"], "classification": probe["classification"]}
    root = root_for(source_sha, item["key"], render_profile())
    bundle.mkdir(parents=True, exist_ok=True)
    pages = []
    force_image_render = bool(item.get("force_image_render"))
    first, last = item.get("start", 1), item.get("end", probe["page_count"])
    scan_images = (pdf_ocr.scan_reader_images(source, first, last)
                   if any(chars == 0 for chars in probe["page_chars"][first - 1:last])
                   else {})
    native_text = {}
    if not lin_native:
        native_numbers = [number for number in range(first, last + 1)
                          if probe["classification"] == "native-text"
                          or probe["page_chars"][number - 1] >= pdf_ocr.MIN_NATIVE_PAGE_CHARS]
        native_text = pdf_ocr.native_pages(source, native_numbers)
    page_started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=bundle) as temp, (
            lin_pdf_text.pymupdf.open(source) if lin_native else nullcontext(None)) as document:
        for number in range(item.get("start", 1), item.get("end", probe["page_count"]) + 1):
            if probe["classification"] == "native-text" and not force_image_render:
                text = lin_pdf_text.extract(document, number) if lin_native else native_text[number]
                payload = pdf_ocr.page_payload(number, text["width"], text["height"], text["blocks"], "native")
                payload.update(ocr_layout.arrange(payload["blocks"], payload["width"], payload["height"], {}))
                if not lin_native:
                    payload["text"] = text["text"]
                out = bundle / root / "ocr" / f"page-{number:06d}.json.gz"
                pdf_ocr.write_gzip_json(out, payload)
                page = {"p": number, "source": "native", "width": text["width"], "height": text["height"],
                        "chars": len(payload["text"]), "text": payload["text"],
                        "text_spans": payload["text_spans"], "layout": payload["layout"]}
                set_page_meta(page, "o", metadata(out, bundle))
                pages.append(page)
                continue
            native = probe["classification"] == "native-text" or probe["page_chars"][number - 1] >= pdf_ocr.MIN_NATIVE_PAGE_CHARS
            reader_pixels = scan_images.get(number) if not native and probe["page_chars"][number - 1] == 0 else None
            png, width, height = pdf_ocr.render_page(source, number, Path(temp), reader_pixels,
                                                     reader_jxl=pdf_ocr.JXL_ENABLED)
            page = {"p": number, "source": "native" if native else "ocr", "width": width, "height": height}
            for field, local, folder in (("i", png, "ocr-input"),
                                         ("w", png.with_suffix(".webp"), "pages")):
                destination = bundle / root / folder / local.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(local), destination)
                set_page_meta(page, field, metadata(destination, bundle))
            if pdf_ocr.JXL_ENABLED:
                jxl = bundle / root / "pages" / f"page-{number:06d}.jxl"
                reader_png = Path(temp) / f"page-{number:06d}-reader.png"
                pdf_ocr.encode_jxl(reader_png if reader_png.is_file() else bundle / page["i"], jxl)
                set_page_meta(page, "j", metadata(jxl, bundle))
            if native:
                text = lin_pdf_text.extract(document, number) if lin_native else native_text[number]
                payload = pdf_ocr.page_payload(number, text["width"], text["height"], text["blocks"], "native")
                payload.update(ocr_layout.arrange(payload["blocks"], payload["width"], payload["height"], {}))
                if not lin_native:
                    payload["text"] = text["text"]
                out = bundle / root / "ocr" / f"page-{number:06d}.json.gz"
                pdf_ocr.write_gzip_json(out, payload)
                set_page_meta(page, "o", metadata(out, bundle))
                page.update({"chars": len(payload["text"]), "text": payload["text"],
                             "text_spans": payload["text_spans"], "layout": payload["layout"]})
            pages.append(page)
            print(f"rendered {number}/{probe['page_count']}: {item['key']}", flush=True)
    timing = {"page_count": len(pages), "image_rendered": any("w" in page for page in pages),
              "page_seconds": time.perf_counter() - page_started,
              "setup_seconds": page_started - started}
    if "start" in item:
        start, end = item["start"], item["end"]
        if not 1 <= start <= end <= probe["page_count"] or len(pages) != end - start + 1:
            raise ValueError("invalid render page range")
        descriptor = bundle / root / f"render-range-{range_id(start, end)}.json"
        pdf_ocr.write_json(descriptor, {**range_identity(base), "version": 1, "kind": "pdf-render-range",
                                        "image_rendered": any("w" in page for page in pages),
                                        "start": start, "end": end, "pages": pages})
        return {**range_identity(base), "status": "range", "start": start, "end": end,
                "timing": timing,
                "descriptor": metadata(descriptor, bundle)}
    image_pages = [{"page": p["p"], **page_meta(p, "w")} for p in pages if "w" in p]
    page_manifest_meta = None
    if image_pages:
        page_manifest = bundle / root / "page-manifest.json"
        pdf_ocr.write_json(page_manifest, pdf_assets.compact_page_manifest(
            source_sha, render_profile(), image_pages, manifest_dir=root))
        page_manifest_meta = {**metadata(page_manifest, bundle), "version": pdf_assets.PAGE_MANIFEST_VERSION}
    manifest = {**base, "version": 1, "kind": "pdf-render", "complete": True,
                "image_rendered": bool(image_pages), "pages": pages, "page_manifest": page_manifest_meta}
    manifest_path = bundle / root / "render-manifest.json"
    pdf_ocr.write_json(manifest_path, manifest)
    return {**base, "status": "ready", "render_manifest": metadata(manifest_path, bundle),
            "page_manifest": page_manifest_meta, "image_rendered": bool(image_pages),
            "ocr_pages": sum(p["source"] == "ocr" for p in pages)}


def validate_render(item, manifest, page_numbers=None):
    if (manifest.get("version") != 1 or manifest.get("kind") != "pdf-render"
            or manifest.get("complete") is not True):
        raise ValueError("invalid render manifest")
    for field in ("key", "source_sha256", "source_revision", "render_profile", "page_count", "page_manifest"):
        if manifest.get(field) != item.get(field):
            raise ValueError(f"render identity mismatch: {field}")
    pages = manifest.get("pages", [])
    expected = list(page_numbers) if page_numbers is not None else list(range(1, item["page_count"] + 1))
    if len(pages) != len(expected) or [p["p"] for p in pages] != expected:
        raise ValueError("incomplete render page sequence")
    if (item.get("force_image_render") and manifest.get("classification") == "native-text"
            and not manifest.get("image_rendered")):
        raise ValueError("native render lacks required page images")
    root = root_for(item["source_sha256"], item["key"], item["render_profile"])
    for p in pages:
        if p.get("source") not in {"native", "ocr"} or p["width"] <= 0 or p["height"] <= 0:
            raise ValueError("invalid rendered page")
        for field, suffix in (("i", ".png"), ("w", ".webp"), ("j", ".jxl"), ("o", ".json.gz")):
            if field in {"i", "w"} and manifest["classification"] == "native-text" and not manifest.get("image_rendered"):
                continue
            if field == "j" and manifest["classification"] == "native-text" and not manifest.get("image_rendered"):
                continue
            if field == "j" and "-jxl-1-" not in item["render_profile"]:
                continue
            if field == "o" and p["source"] != "native":
                continue
            meta = page_meta(p, field)
            pdf_ocr.validate_ocr_object_path(meta["path"], f"page-{p['p']:06d}{suffix}")
            folder = {"i": "ocr-input", "w": "pages", "j": "pages", "o": "ocr"}[field]
            if meta["path"] != (root / folder / f"page-{p['p']:06d}{suffix}").as_posix():
                raise ValueError("rendered page object outside book profile")
            if meta["bytes"] < 1 or len(meta["sha256"]) != 64:
                raise ValueError("invalid rendered object metadata")
    return manifest


def merge_render_ranges(old, new):
    identity = range_identity(new)
    ranges = dict(old.get("ranges", {})) if isinstance(old, dict) and all(
        old.get(key) == value for key, value in identity.items()) else {}
    ranges.update(new["ranges"])
    timings = dict(old.get("range_timings", {})) if isinstance(old, dict) and all(
        old.get(key) == value for key, value in identity.items()) and isinstance(old.get("range_timings", {}), dict) else {}
    timings.update(new.get("range_timings", {}))
    return {**identity, "ranges": ranges, **({"range_timings": timings} if timings else {})}


def assemble_render_book(book, descriptors, bundle):
    pages = []
    ranges = []
    for key, descriptor in descriptors.items():
        bounds = pdf_render_schedule.parse_range(key, book["page_count"])
        if bounds is None:
            raise ValueError("invalid render range key")
        ranges.append((*bounds, descriptor))
    for start, end, descriptor in sorted(ranges, key=lambda item: (item[0], item[1])):
        if start > len(pages) + 1:
            return None
        incoming = validate_range(book, start, end, descriptor)
        overlap = min(len(incoming), max(0, len(pages) - start + 1))
        if pages[start - 1:start - 1 + overlap] != incoming[:overlap]:
            raise ValueError("conflicting overlapping render pages")
        pages.extend(incoming[overlap:])
    if len(pages) != book["page_count"]:
        return None
    root = root_for(book["source_sha256"], book["key"], book["render_profile"])
    image_pages = [{"page": p["p"], **page_meta(p, "w")} for p in pages if "w" in p]
    page_manifest_meta = None
    if image_pages:
        page_manifest = bundle / root / "page-manifest.json"
        page_manifest.parent.mkdir(parents=True, exist_ok=True)
        pdf_ocr.write_json(page_manifest, pdf_assets.compact_page_manifest(
            book["source_sha256"], book["render_profile"], image_pages, manifest_dir=root))
        page_manifest_meta = {**metadata(page_manifest, bundle), "version": pdf_assets.PAGE_MANIFEST_VERSION}
    base = {**public_item(book), "classification": book["probe"]["classification"],
            "render_profile": book["render_profile"], "profile": book["profile"]}
    manifest = {**base, "version": 1, "kind": "pdf-render", "complete": True,
                "image_rendered": bool(image_pages), "pages": pages, "page_manifest": page_manifest_meta}
    output = bundle / root / "render-manifest.json"
    pdf_ocr.write_json(output, manifest)
    result = {**base, "status": "ready", "render_manifest": metadata(output, bundle),
              "page_manifest": page_manifest_meta, "image_rendered": bool(image_pages),
              "ocr_pages": sum(p["source"] == "ocr" for p in pages)}
    validate_render(result, manifest)
    return result


def recognition_identity(entry, options):
    digest = hashlib.sha256(json.dumps(options, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    # Recognition language/model belongs to the OCR worker. A language change
    # must reuse the published PNG render instead of forcing PDF rendering.
    language, backend = pdf_ocr.book_ocr_config(entry)
    base = pdf_ocr.asset_profile(language, backend)
    return f"{base}-layout-{ocr_layout.VERSION}-{digest}"


def generation_for(book):
    return hashlib.sha256((book["render_manifest"]["sha256"] + book["profile"]).encode()).hexdigest()


def layout_options(overrides, key):
    config = overrides.get(key, {})
    default = ocr_layout.validate_options(config.get("default", {}))
    pages = config.get("pages", {})
    for page, value in pages.items():
        if not page.isdigit() or int(page) < 1:
            raise ValueError("layout page numbers must be positive integers")
        ocr_layout.validate_options({**default, **value})
    return {"default": default, "pages": pages}


def reuse_recognized_pages(old, entry, pages):
    """Reuse verified OCR when only Reader image objects changed."""
    old_profile, old_separator, old_layout = str(old.get("profile") or "").partition("-layout-")
    new_profile, new_separator, new_layout = str(entry.get("profile") or "").partition("-layout-")
    same_recognition = (bool(old_separator and new_separator) and old_layout == new_layout
                        and pdf_ocr.ocr_profile_without_jxl(old_profile) ==
                        pdf_ocr.ocr_profile_without_jxl(new_profile))
    if (old.get("status") != "ready" or not same_source(old, entry)
            or old.get("source_sha256") != entry.get("source_sha256")
            or not same_recognition):
        return {}
    try:
        metadata = {"path": old["ocr_manifest"], "sha256": old["ocr_manifest_sha256"],
                    "bytes": old["ocr_manifest_bytes"]}
        previous = json.loads(read_object(metadata, "/ocr-manifest.json"))
        if (previous.get("kind") != "pdf-ocr" or previous.get("complete") is not True
                or previous.get("source_sha256") != entry["source_sha256"]
                or previous.get("profile") != old["profile"]
                or previous.get("page_count") != len(pages)):
            return {}
        old_pages = previous.get("pages")
        if not isinstance(old_pages, list) or len(old_pages) != len(pages):
            return {}
        saved = {}
        for page, prior in zip(pages, old_pages):
            if prior.get("p") != page["p"] or prior.get("source") != page["source"]:
                return {}
            if page["source"] != "ocr":
                continue
            if (page_meta(prior, "i")["sha256"] != page_meta(page, "i")["sha256"]
                    or page_meta(prior, "i")["bytes"] != page_meta(page, "i")["bytes"]):
                return {}
            output = {**page, **{field: prior[field] for field in
                                 ("o", "os", "ob", "text", "text_spans", "layout", "chars")}}
            pdf_ocr.validate_ocr_object_path(output["o"], f"page-{page['p']:06d}.json.gz")
            saved[str(page["p"])] = output
        return saved
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return {}


def plan_images(rendered, current, progress, limit=20, target=500, overrides=None,
                retry_failed_only=False):
    if limit < 1 or target < 1:
        raise ValueError("limit and target must be positive")
    books, tasks = [], []
    for key, entry in sorted(rendered.items(), key=lambda pair: (
            current.get(pair[0], {}).get("status") == "failed", pair[0])):
        if retry_failed_only and current.get(key, {}).get("status") != "failed":
            continue
        if entry.get("status") not in {"ready", "skipped"}:
            continue
        # Older page streams were classified as scans by Poppler even though
        # their repaired GBK source contains Unicode text. Backfill them from
        # the PDF; never send all of their pages to image recognition.
        if lin_pdf_text.applies(entry) and entry.get("ocr_pages", 0) > 0:
            continue
        options = layout_options(overrides or {}, key)
        manifest = None
        if entry["status"] != "skipped":
            manifest = validate_render(entry, json.loads(read_object(entry["render_manifest"], "/render-manifest.json")))
        native_text = "\n".join(str(page.get("text") or "") for page in (manifest or {}).get("pages", []))
        if pdf_ocr.OCR_LANG != "auto":
            language, _, backend = pdf_ocr.resolve_ocr_config(pdf_ocr.OCR_LANG, pdf_ocr.OCR_BACKEND)
        else:
            language = pdf_ocr.detect_language(native_text) if native_text.strip() else pdf_ocr.book_ocr_config(entry)[0]
            backend = (pdf_ocr.OCR_BACKEND if pdf_ocr.OCR_BACKEND != "rapidocr_onnxruntime"
                       else ("rapidocr_onnxruntime" if language in {"ch", "en"} else "paddle_onnxruntime"))
        entry = {**entry, "ocr_language": language, "ocr_backend": backend,
                 "profile": recognition_identity({**entry, "ocr_language": language, "ocr_backend": backend}, options),
                 "layout_options": options}
        old = current.get(key, {})
        if (old.get("status") in {"ready", "skipped"} and same_source(old, entry)
                and old.get("profile") == entry.get("profile")
                and old.get("source_sha256") == entry.get("source_sha256")
                and old.get("page_manifest") == entry.get("page_manifest")):
            continue
        if len(books) >= limit:
            break
        if entry["status"] == "skipped":
            books.append(entry)
            continue
        generation = generation_for(entry)
        previous = progress.get(key, {})
        saved = previous.get("pages", {}) if previous.get("generation") == generation else reuse_recognized_pages(
            old, entry, manifest["pages"])
        book = {**entry, "pages": manifest["pages"], "saved": saved}
        books.append(book)
        pending = [p for p in manifest["pages"] if p["source"] == "ocr" and str(p["p"]) not in saved]
        # Fast ONNX recognition benefits from fewer model startups. Keep the
        # slower multilingual Paddle backend below a conservative task size so
        # one shard cannot approach the runner timeout.
        task_target = min(target, 1200) if entry["ocr_backend"] == "paddle_onnxruntime" else target
        for start in range(0, len(pending), task_target):
            tasks.append({"key": key, "generation": generation, "profile": entry["profile"],
                          "ocr_language": entry["ocr_language"], "ocr_backend": entry["ocr_backend"],
                          "source_sha256": entry["source_sha256"], "layout_options": options,
                          "pages": pending[start:start + task_target]})
    # Pack small books together, while large books can span several workers.
    count = min(256, len(tasks), max(1, math.ceil(sum(len(t["pages"]) for t in tasks) / target)))
    shards = shared.weighted_shards(tasks, count, weight=lambda t: len(t["pages"]),
                                   order=lambda t: (-len(t["pages"]), t["key"], t["pages"][0]["p"])) if tasks else []
    return {"version": 1, "kind": "pdf-image-ocr-queue", "language": pdf_ocr.OCR_LANG,
            "ocr_version": pdf_ocr.OCR_VERSION, "backend": pdf_ocr.OCR_BACKEND, "books": books,
            "target_pages_per_shard": target, "total_ocr_pages": sum(len(t["pages"]) for t in tasks),
            "shard_count": len(shards), "shard_ids": list(range(len(shards))), "shards": shards}


def recognize_task(task, bundle):
    root = root_for(task["source_sha256"], task["key"], task["generation"] + task["profile"])
    done, errors = [], []
    bundle.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=bundle) as temp:
        png = Path(temp) / "input.png"
        for page in task["pages"]:
            try:
                png.write_bytes(read_object(page_meta(page, "i"), ".png"))
                from PIL import Image
                config = task.get("layout_options", {})
                options = {**config.get("default", {}), **config.get("pages", {}).get(str(page["p"]), {})}
                ocr_layout.validate_options(options)
                rotation = options.get("rotation", 0)
                with Image.open(png) as image:
                    if image.size != (page["width"], page["height"]):
                        raise ValueError("OCR input dimensions mismatch")
                    width, height = image.size
                    if rotation:
                        rotated = image.rotate(-rotation, expand=True)
                        width, height = rotated.size
                        rotated.save(png)
                        rotated.close()
                if pdf_ocr.resolve_ocr_config(task["ocr_language"], task["ocr_backend"]) == pdf_ocr.resolve_ocr_config():
                    blocks = pdf_ocr.ocr_page(png, width, height)
                else:
                    blocks = pdf_ocr.ocr_page(png, width, height, task["ocr_language"], task["ocr_backend"])
                payload = pdf_ocr.page_payload(page["p"], page["width"], page["height"], blocks, "ocr")
                arranged = ocr_layout.arrange(blocks, width, height, options)
                payload.update(ocr_layout.restore_coordinates(arranged, rotation))
                out = bundle / root / "ocr" / f"page-{page['p']:06d}.json.gz"
                pdf_ocr.write_gzip_json(out, payload)
                result = {**page, "chars": len(payload["text"]), "text": payload["text"],
                          "text_spans": payload["text_spans"], "layout": payload["layout"]}
                set_page_meta(result, "o", metadata(out, bundle))
                done.append(result)
                print(f"OCR page {page['p']}: {task['key']}", flush=True)
            except Exception as exc:
                errors.append({"page": page["p"], "error": f"{type(exc).__name__}: {exc}"[:1000]})
            finally:
                png.unlink(missing_ok=True)
    return {"key": task["key"], "generation": task["generation"], "pages": done, "errors": errors}


def merge_progress(old, new):
    pages = dict((old or {}).get("pages", {})) if (old or {}).get("generation") == new["generation"] else {}
    pages.update(new["pages"])
    return {"generation": new["generation"], "pages": pages}


def collect_progress(queue, results):
    books = {b["key"]: b for b in queue["books"] if b["status"] == "ready"}
    updates = {}
    for result in results:
        book = books.get(result["key"])
        if not book or result["generation"] != generation_for(book):
            raise ValueError("OCR result generation mismatch")
        source_pages = {p["p"]: p for p in book["pages"]}
        value = updates.setdefault(book["key"], {"generation": result["generation"], "pages": dict(book["saved"])})
        for page in result["pages"]:
            original = source_pages.get(page["p"], {})
            if original.get("source") != "ocr" or page.get("is") != original.get("is"):
                raise ValueError("OCR result input mismatch")
            pdf_ocr.validate_ocr_object_path(page["o"], f"page-{page['p']:06d}.json.gz")
            key = str(page["p"])
            if key in value["pages"] and value["pages"][key] != page:
                raise ValueError("conflicting OCR page results")
            value["pages"][key] = page
    return updates


def assemble_book(book, saved, bundle):
    base = {k: v for k, v in book.items() if k not in {"pages", "saved"}}
    if book["status"] == "skipped":
        return base
    pages = [p if p["source"] == "native" else saved.get(str(p["p"])) for p in book["pages"]]
    if any(p is None for p in pages):
        return {**base, "status": "failed", "error": "OCR pages incomplete; uploaded pages retained for retry"}
    texts = []
    for page in pages:
        payload = json.loads(gzip.decompress(read_object(page_meta(page, "o"), ".json.gz")))
        if (payload.get("kind") != "pdf-ocr-page" or payload.get("page") != page["p"]
                or payload.get("source") != page["source"]):
            raise ValueError("OCR page payload mismatch")
        if page["source"] == "native":
            config = book.get("layout_options", {})
            options = {**config.get("default", {}), **config.get("pages", {}).get(str(page["p"]), {})}
            # Native extraction is already in original page coordinates.
            options.pop("rotation", None)
            if payload["blocks"]:
                payload.update(ocr_layout.arrange(payload["blocks"], payload["width"], payload["height"], options))
            else:
                payload["text_spans"] = []
                payload["layout"] = {"version": ocr_layout.VERSION, "writing_mode": "auto",
                                     "review": ["native-text-without-positioned-blocks"] if payload["text"] else [],
                                     "mapping_precision": "block", "offset_unit": "unicode-codepoint"}
        texts.append({"page": page["p"], "text": payload["text"],
                      "layout": payload["layout"], "text_spans": payload["text_spans"]})
    language, backend = pdf_ocr.book_ocr_config(book)
    root = root_for(book["source_sha256"], book["key"], book["render_manifest"]["sha256"] + book["profile"])
    text_path = bundle / root / "ocr" / "book-text.json.gz"
    pdf_ocr.write_gzip_json(text_path, {"version": 2, "kind": "pdf-book-text", "complete": True,
                                      "source_sha256": book["source_sha256"], "page_count": book["page_count"],
                                      "offset_unit": "unicode-codepoint", "profile": book["profile"],
                                      "language": language, "ocr_version": pdf_ocr.resolve_ocr_config(language, backend)[1],
                                      "pages": texts})
    manifest_path = bundle / root / "ocr-manifest.json"
    pdf_ocr.write_json(manifest_path, {
        "version": 1, "kind": "pdf-ocr", "complete": True, "profile": book["profile"],
        "engine": f"{backend} / {pdf_ocr.resolve_ocr_config(language, backend)[1]} / CPU / {language}",
        "language": language, "ocr_version": pdf_ocr.resolve_ocr_config(language, backend)[1],
        "backend": backend, "source_sha256": book["source_sha256"],
        "source_bytes": book["source_bytes"], "source_revision": book.get("source_revision", ""),
        "classification": book["classification"], "page_count": book["page_count"],
        "dpi": pdf_ocr.OCR_DPI, "pages": pages, "book_text": metadata(text_path, bundle),
        **({"page_manifest": book["page_manifest"]} if book.get("page_manifest") else {}),
    })
    meta = metadata(manifest_path, bundle)
    return {**base, "status": "ready", "language": language, "ocr_version": pdf_ocr.resolve_ocr_config(language, backend)[1],
            "backend": backend, "stream": bool(book.get("page_manifest")), "ocr_manifest": meta["path"],
            "ocr_manifest_sha256": meta["sha256"], "ocr_manifest_bytes": meta["bytes"]}


def read_results(paths):
    results = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("results"), list):
            raise ValueError(f"invalid results: {path}")
        results.extend(data["results"])
    return results


def result_paths(paths, directory, expected=False):
    found = set(paths)
    if directory and directory.is_dir():
        found.update(directory.rglob("results*.json"))
    if expected and not found:
        raise ValueError("planned workers but no result artifacts found; refusing empty publication")
    return sorted(found)


def publish_legacy_render(queue, results, api, repo):
    planned = {item["key"]: item for shard in queue["shards"] for item in shard["records"]}
    for result in results:
        item = planned.get(result["key"])
        if not item or result.get("source_sha256") != item.get("source_sha256"):
            raise ValueError("render result does not match source queue")
        if result.get("status") == "ready":
            manifest = validate_render(result, json.loads(read_object(result["render_manifest"], "/render-manifest.json")))
            if result.get("page_manifest"):
                images = json.loads(read_object(result["page_manifest"], "/page-manifest.json"))
                if images.get("page_count") != manifest["page_count"]:
                    raise ValueError("render page manifest count mismatch")
    results.extend({**public_item(x), "render_profile": render_profile()} for x in queue.get("failed", []))
    seen = {r["key"] for r in results}
    for item in planned.values():
        if item["key"] not in seen:
            results.append({**public_item(item), "render_profile": render_profile(),
                            "status": "failed", "error": "render worker result missing"})
    save_registry(api, repo, RENDER_REGISTRY, {r["key"]: r for r in results}, publish_streams=True)
    print(f"render publication: {len(results)} results, "
          f"{sum(r.get('status') == 'ready' for r in results)} ready", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan-render", "measure-render", "render", "publish-render", "plan-ocr", "ocr", "publish-ocr"))
    parser.add_argument("--queue", type=Path, default=Path("output/pdf-ocr/queue.json"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf-ocr/bundle"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--partition", choices=("all", "small", "large"), default="all")
    parser.add_argument("--native-text-stream", action="store_true")
    parser.add_argument("--target-render-seconds", type=int)
    parser.add_argument("--retry-failed-only", action="store_true")
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--layout-overrides", type=Path, default=Path("state/pdf_ocr_layout.json"))
    parser.add_argument("--assets-repo", default="vomebook/Reader-Assets")
    args = parser.parse_args()
    if args.target_render_seconds is not None and (args.target_render_seconds < 1 or args.stage != "plan-render"):
        parser.error("--target-render-seconds requires plan-render and a positive value")
    if args.stage == "measure-render":
        if args.source is None:
            parser.error("measure-render requires --source")
        sample_render_cost(json.loads(args.queue.read_text(encoding="utf-8")), args.source, args.output)
        return 0
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = args.assets_repo
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage.startswith("plan-"):
        revision = retry(lambda: api.repo_info(repo_id=repo, repo_type="dataset")).sha
        rendered = load_registry(api, repo, RENDER_REGISTRY, revision)["files"]
        current = load_registry(api, repo, publication.OCR_MANIFEST_NAME, revision)["files"]
        if args.stage == "plan-render":
            assets = load_registry(api, repo, "manifest.json", revision)
            assets["revision"] = revision
            range_state = load_registry(api, repo, "pdf_range_manifest.json", revision)
            records = pdf_ocr.source_records(args.search_data, args.revisions, assets, range_manifest=range_state)
            records = pending_render(records, rendered, current, args.retry_failed, args.partition)
            selected = pdf_ocr.queue(records, args.limit, args.checkpoint)
            render_progress = load_registry(api, repo, RENDER_PROGRESS_REGISTRY, revision)["files"]
            estimator = (lambda item, source: estimate_render_cost(item, source, render_progress)) \
                if args.target_render_seconds is not None else None
            queue = plan_pdf_ocr.plan(selected, native_text_stream=args.native_text_stream,
                                      render_estimator=estimator)
            queue["kind"] = "pdf-render-queue"
            queue = plan_render_ranges(queue, render_progress, args.target_render_seconds)
        else:
            progress = load_registry(api, repo, PROGRESS_REGISTRY, revision)["files"]
            overrides = json.loads(args.layout_overrides.read_text(encoding="utf-8")) if args.layout_overrides.is_file() else {}
            queue = plan_images(rendered, current, progress, args.limit,
                                plan_pdf_ocr.ocr_target_pages_per_shard(), overrides,
                                retry_failed_only=args.retry_failed_only)
        args.queue.parent.mkdir(parents=True, exist_ok=True)
        pdf_ocr.write_json(args.queue, queue)
        print(f"{args.stage}: {queue['shard_count']} shards", flush=True)
        return 0
    if args.stage == "publish-render":
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
        results = read_results(result_paths(args.results, args.results_dir, False))
        if "books" not in queue:
            if queue["shards"] and not results:
                raise ValueError("planned workers but no result artifacts found; refusing empty publication")
            publish_legacy_render(queue, results, api, repo)
            return 0
        planned = {(item["key"], item["start"], item["end"]): item for item in expand_render_tasks(
            queue, [item for shard in queue["shards"] for item in shard["records"]])}
        books = {item["key"]: item for item in queue["books"]}
        completed_ranges = {key: dict(value) for key, value in queue.get("saved_ranges", {}).items()}
        range_timings = {}
        for result in results:
            item = planned.get((result["key"], result.get("start"), result.get("end")))
            if not item or any(result.get(field) != value for field, value in range_identity(item).items()):
                raise ValueError("render result does not match source queue")
            if result.get("status") == "range":
                start, end = item["start"], item["end"]
                validate_range(item, start, end, result["descriptor"])
                saved = completed_ranges.setdefault(item["key"], {})
                key = range_id(start, end)
                if key in saved and saved[key] != result["descriptor"]:
                    raise ValueError("conflicting render range results")
                saved[key] = result["descriptor"]
                timing = result.get("timing", {})
                if (isinstance(timing, dict) and timing.get("page_count") == end - start + 1
                        and pdf_render_schedule.finite_seconds(timing.get("page_seconds"), positive=True) is not None
                        and pdf_render_schedule.finite_seconds(timing.get("setup_seconds")) is not None):
                    range_timings.setdefault(item["key"], {})[key] = timing
        updates = [{**range_identity(book), "ranges": completed_ranges[key]}
                   for key, book in books.items() if key in completed_ranges]
        for update in updates:
            if update["key"] in range_timings:
                update["range_timings"] = range_timings[update["key"]]
        if updates:
            save_registry(api, repo, RENDER_PROGRESS_REGISTRY, {r["key"]: r for r in updates},
                          merge=merge_render_ranges)
        published = [{**public_item(x), "render_profile": render_profile()} for x in queue.get("failed", [])]
        for book in books.values():
            with tempfile.TemporaryDirectory(dir=args.output) as temp:
                try:
                    result = assemble_render_book(book, completed_ranges.get(book["key"], {}), Path(temp))
                    if result:
                        upload_objects(Path(temp))
                        validate_render(result, json.loads(read_object(result["render_manifest"], "/render-manifest.json")))
                        if result.get("page_manifest"):
                            images = json.loads(read_object(result["page_manifest"], "/page-manifest.json"))
                            if images.get("page_count") != result["page_count"]:
                                raise ValueError("render page manifest count mismatch")
                except Exception as exc:
                    result = {**public_item(book), "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:1000]}
                published.append(result or {**public_item(book), "status": "failed",
                                            "error": "render ranges incomplete; uploaded ranges retained for retry"})
        save_registry(api, repo, RENDER_REGISTRY, {r["key"]: r for r in published}, publish_streams=True)
        print(f"render publication: {len(published)} results, "
              f"{sum(r.get('status') == 'ready' for r in published)} ready", flush=True)
        return 0
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    expected = "pdf-render-queue" if args.stage == "render" else "pdf-image-ocr-queue"
    if queue.get("version") != 1 or queue.get("kind") != expected:
        raise ValueError("invalid stage queue")
    if args.stage == "publish-ocr":
        results = read_results(result_paths(args.results, args.results_dir, bool(queue["shards"])))
        updates = collect_progress(queue, results)
        save_registry(api, repo, PROGRESS_REGISTRY, updates, merge_progress)
        completed = []
        for book in queue["books"]:
            try:
                saved = updates.get(book["key"], {}).get("pages", book.get("saved", {}))
                with tempfile.TemporaryDirectory(dir=args.output) as temp:
                    result = assemble_book(book, saved, Path(temp))
                    if result["status"] == "ready":
                        upload_objects(Path(temp))
                completed.append(result)
            except Exception as exc:
                completed.append({**public_item({k: v for k, v in book.items() if k not in {"pages", "saved"}}),
                                  "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:1000]})
        if completed:
            publication.publish(api, repo, completed)
        print(f"published {len(completed)} books; incomplete books retain page progress", flush=True)
        return 0
    if not 0 <= args.shard < len(queue["shards"]):
        raise ValueError("invalid shard")
    tasks = queue["shards"][args.shard]
    if args.stage == "render":
        tasks = expand_render_tasks(queue, tasks["records"]) if "books" in queue else tasks["records"]
    else:
        # Same worker/model, small durable checkpoints. A timeout near page 500
        # must not discard all of that worker's completed recognition.
        tasks = [{**task, "pages": task["pages"][start:start + 25]}
                 for task in tasks for start in range(0, len(task["pages"]), 25)]
    results = []
    for task in tasks:
        try:
            with tempfile.TemporaryDirectory(dir=args.output) as temp:
                if args.stage == "render":
                    result = render_book(task, source_path(task), Path(temp))
                else:
                    result = recognize_task(task, Path(temp))
                upload_objects(Path(temp))
            results.append(result)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:1000]
            if args.stage == "render":
                result = {**range_identity(task), "start": task["start"], "end": task["end"],
                          "status": "failed", "error": error}
            else:
                result = {"key": task["key"], "generation": task["generation"], "pages": [], "errors": [{"error": error}]}
            results.append(result)
        # Keep completed book/range metadata even if a later task times out.
        pdf_ocr.write_json(args.output / f"results-{args.shard}.json", {"version": 1, "results": results})
    return int(any(r.get("status") == "failed" or r.get("errors") for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
