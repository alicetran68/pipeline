#!/usr/bin/env python3
"""Publish OCR metadata after immutable page objects reached the Bucket."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import pdf_ocr, shared, reader_lifecycle
    from .reader_assets import READER_ASSETS_REPO
    from .reader_bucket import (INDEX_FILES, publish_catalog, publish_indexes,
                                publish_json, read_json as read_bucket_json,
                                read_bytes as read_bucket_bytes)
except ImportError:
    import pdf_ocr
    import shared
    import reader_lifecycle
    from reader_assets import READER_ASSETS_REPO
    from reader_bucket import (INDEX_FILES, publish_catalog, publish_indexes,
                               publish_json, read_json as read_bucket_json,
                               read_bytes as read_bucket_bytes)


OCR_MANIFEST_NAME = "pdf_ocr_manifest.json"
SIDECAR_NAME = "reader_assets.json.gz"


def load_remote(api: HfApi, repo: str, filename: str, fallback):
    bucket_name = {"manifest.json": "manifest", OCR_MANIFEST_NAME: "ocr"}.get(filename)
    if bucket_name and type(api) is HfApi:
        try:
            return read_bucket_json(INDEX_FILES[bucket_name], os.environ.get("HF_TOKEN"))
        except FileNotFoundError:
            return fallback
    try:
        path = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename)
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return fallback
        raise
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sidecar(api: HfApi, repo: str) -> dict:
    if type(api) is HfApi:
        try:
            data = json.loads(gzip.decompress(read_bucket_bytes(
                INDEX_FILES["sidecar"], os.environ.get("HF_TOKEN"))).decode("utf-8"))
        except FileNotFoundError:
            return {"v": 1, "f": {}}
        if data.get("v") != 1 or not isinstance(data.get("f"), dict):
            raise ValueError("invalid Reader bucket sidecar")
        return data
    try:
        path = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=SIDECAR_NAME)
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return {"v": 1, "f": {}}
        raise
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("v") != 1 or not isinstance(data.get("f"), dict):
        raise ValueError("invalid Reader Assets sidecar")
    return data


def encode_sidecar(data: dict) -> bytes:
    return gzip.compress(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9, mtime=0,
    )


def update_sidecar(sidecar: dict, results: list[dict]) -> dict:
    updated = {"v": 1, "f": dict(sidecar.get("f", {}))}
    for result in results:
        key = result.get("key")
        if not key:
            continue
        current = dict(updated["f"].get(key) or {})
        merged = shared.merge_pdf_ocr_sidecar_entry(current, result)
        if merged:
            updated["f"][key] = merged
        else:
            updated["f"].pop(key, None)
    return updated


def release_processing_roots(results: list[dict], generation: str, token: str | None) -> None:
    for result in results:
        for path in result.get("processing_roots", []):
            if not isinstance(path, str) or not path.startswith("reader-index/processing/"):
                raise ValueError("invalid processing root path")
            record = read_bucket_json(path, token)
            reader_lifecycle.validate_processing_handoff(record, result)
            released = reader_lifecycle.release_processing(record, generation)
            publish_json(path, released, token)


def published(current: dict, results: list[dict]) -> bool:
    files = current.get("files", {})
    for result in results:
        entry = files.get(result.get("key"), {})
        if result.get("status") == "ready":
            if (entry.get("status") != "ready" or entry.get("profile") != result.get("profile")
                    or entry.get("source_sha256") != result.get("source_sha256")
                    or entry.get("ocr_manifest") != result.get("ocr_manifest")
                    or entry.get("page_manifest") != result.get("page_manifest")):
                return False
        elif result.get("status") in {"skipped", "failed"}:
            if entry.get("status") != result.get("status"):
                return False
        elif entry.get("status") != "failed" or entry.get("error") != result.get("error"):
            return False
    return bool(results)


def publish(api: HfApi, repo: str, results: list[dict], attempts: int = 20) -> None:
    for attempt in range(attempts):
        bucket_publication = type(api) is HfApi
        info = None if bucket_publication else api.repo_info(repo_id=repo, repo_type="dataset")
        manifest = load_remote(api, repo, OCR_MANIFEST_NAME, pdf_ocr.empty_manifest())
        if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
            raise ValueError("invalid remote PDF OCR manifest")
        files = dict(manifest["files"])
        for result in results:
            entry = {key: value for key, value in result.items()
                     if key not in {"bundle_root", "probe", "page_chars"}}
            files[result["key"]] = entry
        updated = {"version": 1, "profile": pdf_ocr.asset_profile(),
                   "files": dict(sorted(files.items()))}
        old_sidecar = load_sidecar(api, repo)
        sidecar = update_sidecar(old_sidecar, results)
        if published(manifest, results) and sidecar == old_sidecar:
            if bucket_publication and os.environ.get("HF_TOKEN"):
                generation = publish_catalog(sidecar, os.environ.get("HF_TOKEN"))
                release_processing_roots(results, generation, os.environ.get("HF_TOKEN"))
            return
        operations = [
            CommitOperationAdd(path_in_repo=OCR_MANIFEST_NAME,
                               path_or_fileobj=(json.dumps(updated, ensure_ascii=False,
                                                           sort_keys=True, indent=2) + "\n").encode()),
            CommitOperationAdd(path_in_repo=SIDECAR_NAME, path_or_fileobj=encode_sidecar(sidecar)),
        ]
        try:
            if bucket_publication:
                publish_indexes({
                    INDEX_FILES["ocr"]: (json.dumps(updated, ensure_ascii=False,
                                                   sort_keys=True, indent=2) + "\n").encode(),
                    INDEX_FILES["sidecar"]: encode_sidecar(sidecar),
                }, os.environ.get("HF_TOKEN"))
                token = os.environ.get("HF_TOKEN")
                if token:
                    generation = publish_catalog(sidecar, token)
                    release_processing_roots(results, generation, token)
                return
            api.create_commit(repo_id=repo, repo_type="dataset", operations=operations,
                               commit_message="Publish PDF OCR metadata", parent_commit=info.sha)
            return
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {409, 412, 429} and not (status and 500 <= status < 600):
                raise
            if attempt + 1 == attempts:
                raise
            time.sleep(min(60, 2 ** min(attempt, 5)))
    raise RuntimeError("PDF OCR publication retry limit reached")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = []
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("results"), list):
            raise ValueError(f"invalid OCR result bundle: {path}")
        results.extend(data["results"])
    if args.dry_run:
        print(json.dumps({"results": len(results), "profile": pdf_ocr.asset_profile()}, ensure_ascii=False))
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    publish(HfApi(token=token), args.assets_repo, results)
    print(f"published {len(results)} PDF OCR metadata result(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
