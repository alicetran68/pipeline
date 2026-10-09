#!/usr/bin/env python3
"""Build one OCR shard; upload only immutable objects and a small result file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, sync_bucket
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import pdf_ocr, shared
    from .reader_bucket import materialize as materialize_bucket
except ImportError:
    import pdf_ocr
    import shared
    from reader_bucket import materialize as materialize_bucket


def _bucket_retry_delay(error: HfHubHTTPError, attempt: int) -> int:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return max(0, min(300, int(float(retry_after))))
        except (TypeError, ValueError):
            pass
    return min(300, 5 * (2 ** attempt))


def _sync_bucket_with_retry(local_dir: str, bucket: str, token: str | None,
                            max_attempts: int = 8, include: list[str] | None = None) -> None:
    """Upload immutable OCR objects without turning temporary Hub throttling into a failed book."""
    for attempt in range(max_attempts):
        try:
            sync_bucket(local_dir, bucket, include=include, token=token, quiet=True)
            return
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            retryable = status is None or status == 429 or 500 <= status < 600
            if not retryable or attempt + 1 == max_attempts:
                raise
            delay = _bucket_retry_delay(exc, attempt)
        except (httpx.TransportError, ConnectionError, OSError):
            if attempt + 1 == max_attempts:
                raise
            delay = min(300, 5 * (2 ** attempt))
        print(f"transient bucket sync error; retrying in {delay}s "
              f"(attempt {attempt + 1}/{max_attempts})", flush=True)
        time.sleep(delay)
    raise RuntimeError("bucket sync retry limit reached")


def upload_ocr_objects(bundle: Path) -> None:
    """Scope remote listings to each local book/profile, not the whole Bucket."""
    for root in sorted((bundle / "objects").glob("*/*/*")):
        if root.is_dir():
            destination = f"hf://buckets/{shared.PDF_PAGES_BUCKET}/{root.relative_to(bundle).as_posix()}"
            _sync_bucket_with_retry(str(root), destination, os.environ.get("HF_TOKEN"))


def source_path(item: dict) -> Path:
    for attempt in range(6):
        try:
            if item.get("source_kind") == "generated":
                if item.get("reader_assets_bucket") and item.get("reader_assets_path"):
                    return materialize_bucket(item["reader_assets_path"], os.environ.get("HF_TOKEN"), ".pdf",
                                              bucket=item["reader_assets_bucket"])
                return Path(hf_hub_download(
                    item["reader_assets_repo"], item["reader_assets_path"], repo_type="dataset",
                    revision=item["reader_assets_revision"], token=os.environ.get("HF_TOKEN")))
            return Path(hf_hub_download(
                item["repo"], item["path"], repo_type="dataset", revision=item["source_revision"],
                token=os.environ.get("HF_TOKEN")))
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("source download retry limit reached")


def build_queue(queue_path: Path, shard: int, output: Path, sync_objects: bool = False) -> list[dict]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("version") != 1 or queue.get("kind") != "pdf-ocr-queue":
        raise ValueError("invalid PDF OCR queue")
    shards = queue.get("shards")
    if not isinstance(shards, list) or not 0 <= shard < len(shards):
        raise ValueError("invalid PDF OCR shard")
    results = []
    for item in shards[shard].get("records", []):
        book = output / f"book-{len(results):04d}"
        book.mkdir(parents=True, exist_ok=True)
        try:
            result = pdf_ocr.build_item(item, source_path(item), book)
            result["bundle_root"] = book.name
            if sync_objects and result.get("status") == "ready":
                upload_ocr_objects(book)
        except Exception as exc:
            result = {**item, "status": "failed", "profile": pdf_ocr.asset_profile(),
                      "error": f"{type(exc).__name__}: {exc}"[:1000]}
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/pdf-ocr/bundle"))
    parser.add_argument("--sync-bucket", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = build_queue(args.queue, args.shard, args.output, args.sync_bucket)
    (args.output / "results.json").write_text(
        json.dumps({"version": 1, "profile": pdf_ocr.asset_profile(), "results": results},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"built {len(results)} PDF OCR result(s)")
    return 0 if all(result.get("status") != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
