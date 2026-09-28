#!/usr/bin/env python3
"""Report or delete unreferenced roots in the PDF page Bucket.

The Bucket has no manifest-aware garbage collector. Keep whole immutable roots
when any current Reader, render, OCR, sidecar, or progress manifest references
an object below them; delete only old roots that are completely unreferenced.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

try:
    from .reader_assets import READER_ASSETS_REPO
except ImportError:
    from reader_assets import READER_ASSETS_REPO

BUCKET = "vomebook/pdf-pages"
OBJECT_PREFIX = "objects"
REGISTRY_FILES = (
    "manifest.json",
    "pdf_manifest.json",
    "pdf_render_manifest.json",
    "pdf_ocr_manifest.json",
    "pdf_render_progress.json",
    "pdf_ocr_progress.json",
    "pdf_range_manifest.json",
)
SIDECAR_NAME = "reader_assets.json.gz"


def object_root(path: str) -> str | None:
    parts = str(path or "").split("/")
    if len(parts) < 4 or parts[0] != OBJECT_PREFIX:
        return None
    if len(parts[1]) != 2 or len(parts[2]) != 64 or len(parts[3]) != 16:
        return None
    if any(not part for part in parts[:4]):
        return None
    return "/".join(parts[:4])


def referenced_roots(*values) -> set[str]:
    roots: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            root = object_root(value)
            if root:
                roots.add(root)

    for value in values:
        visit(value)
    return roots


def load_optional_json(api: HfApi, repo: str, filename: str, revision: str) -> dict:
    try:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename,
                               revision=revision, token=api.token)
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return {}
        raise
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_optional_sidecar(api: HfApi, repo: str, revision: str) -> dict:
    try:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=SIDECAR_NAME,
                               revision=revision, token=api.token)
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return {}
        raise
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    return data if isinstance(data, dict) else {}


def load_protection(api: HfApi, repo: str, revision: str) -> set[str]:
    values = [load_optional_json(api, repo, name, revision) for name in REGISTRY_FILES]
    values.append(load_optional_sidecar(api, repo, revision))
    return referenced_roots(*values)


def _list_directories(prefix: str, token: str, attempts: int = 6) -> list[tuple[str, object]]:
    for attempt in range(attempts):
        try:
            api = HfApi(token=token)
            return [
                (str(item.path), getattr(item, "uploaded_at", None))
                for item in api.list_bucket_tree(BUCKET, prefix=prefix, recursive=False, token=token)
                if getattr(item, "type", None) == "directory"
            ]
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {408, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
            time.sleep(min(120, 2 ** attempt))
    raise RuntimeError("bucket directory listing retry limit reached")


def _parallel_directories(prefixes: list[str], token: str, workers: int) -> list[tuple[str, object]]:
    output = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_list_directories, prefix, token) for prefix in prefixes]
        for future in as_completed(futures):
            output.extend(future.result())
    return output


def list_object_roots(token: str, workers: int = 4) -> list[tuple[str, object]]:
    shards = [path for path, _ in _list_directories(OBJECT_PREFIX, token)]
    source_dirs = _parallel_directories(shards, token, workers)
    roots = _parallel_directories([path for path, _ in source_dirs], token, workers)
    return sorted(roots, key=lambda item: item[0])


def _uploaded_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def eligible_roots(roots: list[tuple[str, object]], protected: set[str], today: date,
                   grace_days: int, limit: int) -> list[str]:
    cutoff = today - timedelta(days=grace_days)
    candidates = []
    for path, uploaded_at in roots:
        if path in protected:
            continue
        uploaded = _uploaded_date(uploaded_at)
        if uploaded is None or uploaded > cutoff:
            continue
        candidates.append(path)
    return candidates[:limit]


def delete_roots(roots: list[str], token: str) -> None:
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="pdf-bucket-gc-") as empty:
        for root in roots:
            api.sync_bucket(empty, f"hf://buckets/{BUCKET}/{root}", delete=True,
                            token=token, quiet=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    parser.add_argument("--grace-days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-unreferenced", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grace_days < 0 or args.limit < 1 or args.workers < 1:
        raise ValueError("grace-days, limit, and workers must be positive")
    if args.apply and not args.allow_unreferenced:
        raise ValueError("bucket deletion requires --allow-unreferenced")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    revision = api.repo_info(repo_id=args.assets_repo, repo_type="dataset").sha
    protected = load_protection(api, args.assets_repo, revision)
    roots = list_object_roots(token, args.workers)
    paths = eligible_roots(roots, protected, date.today(), args.grace_days, args.limit)
    print(f"bucket roots: {len(roots)}; protected: {len(protected)}; eligible: {len(paths)}")
    for path in paths:
        print(path)
    if args.apply and paths:
        delete_roots(paths, token)
        print(f"deleted {len(paths)} unreferenced PDF bucket root(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
