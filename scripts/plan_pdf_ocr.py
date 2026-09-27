#!/usr/bin/env python3
"""Plan deterministic, page-weighted PP-OCRv6 PDF work."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import pdf_assets, pdf_ocr, lin_pdf_text, shared
except ImportError:
    import pdf_assets
    import pdf_ocr
    import lin_pdf_text
    import shared


MAX_OCR_SHARDS = 20
DEFAULT_OCR_TARGET_PAGES_PER_SHARD = 2000


def ocr_target_pages_per_shard() -> int:
    value = os.environ.get("PDF_OCR_TARGET_PAGES_PER_SHARD", str(DEFAULT_OCR_TARGET_PAGES_PER_SHARD))
    try:
        return max(1, int(value))
    except ValueError as exc:
        raise ValueError("PDF_OCR_TARGET_PAGES_PER_SHARD must be a positive integer") from exc


def recommended_ocr_shard_count(records: list[dict]) -> int:
    """Choose enough book-level shards to amortize runner/model startup overhead."""
    if not records:
        return 0
    total_pages = sum(max(1, int(item.get("page_count", 0))) for item in records)
    target = ocr_target_pages_per_shard()
    return max(1, min(MAX_OCR_SHARDS, len(records), math.ceil(total_pages / target)))


def retry(operation, label: str):
    for attempt in range(6):
        try:
            return operation()
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            delay = min(60, 2 ** attempt)
            print(f"transient Hub error while {label} ({status}); retrying in {delay}s", flush=True)
            time.sleep(delay)


def download_source(item: dict) -> Path:
    if item.get("source_kind") == "generated":
        return Path(retry(lambda: hf_hub_download(
            item["reader_assets_repo"], item["reader_assets_path"], repo_type="dataset",
            revision=item["reader_assets_revision"], token=os.environ.get("HF_TOKEN")),
            "downloading generated PDF"))
    return Path(retry(lambda: hf_hub_download(
        item["repo"], item["path"], repo_type="dataset", revision=item["source_revision"],
        token=os.environ.get("HF_TOKEN")), "downloading source PDF"))


def plan(records: list[dict], workers: int = 4, current: dict | None = None,
         retry_failed: bool = False, native_text_stream: bool = False, render_estimator=None) -> dict:
    current_files = (current or {}).get("files", {})
    terminal = {"ready", "failed", "skipped"} if not retry_failed else {"ready", "skipped"}
    records = [item for item in records if not (
        isinstance(current_files.get(item["key"]), dict)
        and current_files[item["key"]].get("status") in terminal
        and current_files[item["key"]].get("profile") == pdf_ocr.asset_profile()
        and current_files[item["key"]].get("source_revision") == item.get("source_revision")
    )]
    def inspect(item: dict) -> dict:
        try:
            source = download_source(item)
            digest, size = shared.hash_file(source)
            probe = lin_pdf_text.probe(source) if lin_pdf_text.applies(item) else pdf_ocr.probe_pdf(source)
            inspected = {**item, "source_sha256": digest, "source_bytes": size, "probe": probe,
                    "page_count": probe["page_count"], "status": "planned", "profile": pdf_ocr.asset_profile(),
                    **({"native_extractor": "pymupdf-v1"} if lin_pdf_text.applies(item)
                       and probe["native_page_ratio"] >= .8 else {}),
                    **({"force_image_render": True} if native_text_stream
                       and probe["classification"] == "native-text" else {})}
            if render_estimator is not None:
                inspected["_render_cost"] = render_estimator(inspected, source)
            return inspected
        except Exception as exc:
            return {**item, "status": "failed", "profile": pdf_ocr.asset_profile(),
                    "error": f"{type(exc).__name__}: {exc}"[:1000]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        selected = list(executor.map(inspect, records))
    ready = [item for item in selected if item.get("status") == "planned"]
    shards = pdf_ocr_shards(ready, recommended_ocr_shard_count(ready)) if ready else []
    return {
        "version": 1, "kind": "pdf-ocr-queue", "profile": pdf_ocr.asset_profile(),
        "total_records": len(selected), "total_pages": sum(item["page_count"] for item in ready),
        "target_pages_per_shard": ocr_target_pages_per_shard(),
        "shard_count": len(shards), "shard_ids": list(range(len(shards))),
        "failed": [item for item in selected if item.get("status") == "failed"],
        "shards": [{"index": index, "page_count": sum(x["page_count"] for x in shard),
                    "records": shard} for index, shard in enumerate(shards)],
    }


def pdf_ocr_shards(records: list[dict], shard_count: int) -> list[list[dict]]:
    shards = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for item in sorted(records, key=lambda value: (-value["page_count"], value["key"])):
        index = min(range(shard_count), key=lambda value: (loads[value], value))
        shards[index].append(item)
        loads[index] += item["page_count"]
    return [shard for shard in shards if shard]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--assets-manifest", type=Path)
    parser.add_argument("--ocr-manifest", type=Path)
    parser.add_argument("--range-manifest", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--repo", default="")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("output/pdf-ocr/queue.json"))
    args = parser.parse_args()
    assets = None
    if args.assets_manifest and args.assets_manifest.is_file():
        assets = json.loads(args.assets_manifest.read_text(encoding="utf-8"))
    range_state = None
    if args.range_manifest and args.range_manifest.is_file():
        range_state = json.loads(args.range_manifest.read_text(encoding="utf-8"))
    records = pdf_ocr.source_records(args.search_data, args.revisions, assets, args.repo, range_state)
    current = None
    if args.ocr_manifest and args.ocr_manifest.is_file():
        current = json.loads(args.ocr_manifest.read_text(encoding="utf-8"))
    if current:
        files = current.get("files", {})
        pending = []
        for item in records:
            entry = files.get(item["key"])
            if not isinstance(entry, dict) or entry.get("source_revision") != item.get("source_revision"):
                pending.append(item)
                continue
            terminal = {"ready", "failed", "skipped"} if not args.retry_failed else {"ready", "skipped"}
            if entry.get("status") in terminal:
                if entry.get("profile") == pdf_ocr.asset_profile():
                    continue
                # A JXL-only profile upgrade can reuse the old OCR/page objects.
                if entry.get("status") == "ready":
                    pending.append({**item, "_previous_ocr": entry})
                    continue
            pending.append(item)
        records = pending
    selected = pdf_ocr.queue(records, args.limit, args.checkpoint)
    result = plan(selected, args.workers, retry_failed=args.retry_failed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"planned {result['total_records']} PDF OCR book(s), {result['total_pages']} page(s), {result['shard_count']} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
