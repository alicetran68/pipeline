#!/usr/bin/env python3
"""Build a deterministic queue for changed files requiring reader conversion."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

try:
    from .reader_assets import (
        EPUB_CHAPTER_PROFILE, MANIFEST_NAME, READER_ASSETS_BUCKET, READER_ASSETS_REPO,
        SPREADSHEET_PAGE_PROFILE, asset_key, bucket_conversion_contract, canonical_json,
        conversion_dependencies, decode_search_payload,
        empty_manifest, load_json, needs_epub_chapters, relative_path, reusable_object_key,
        source_conversion_contract, source_url, validate_manifest,
    )
    from .reader_bucket import INDEX_FILES, read_json as read_bucket_json
except ImportError:
    from reader_assets import (
        EPUB_CHAPTER_PROFILE, MANIFEST_NAME, READER_ASSETS_BUCKET, READER_ASSETS_REPO,
        SPREADSHEET_PAGE_PROFILE, asset_key, bucket_conversion_contract, canonical_json,
        conversion_dependencies, decode_search_payload,
        empty_manifest, load_json, needs_epub_chapters, relative_path, reusable_object_key,
        source_conversion_contract, source_url, validate_manifest,
    )
    from reader_bucket import INDEX_FILES, read_json as read_bucket_json

try:
    from . import shared
except ImportError:
    import shared


def remote_manifest(api: HfApi, repo_id: str) -> dict:
    if type(api).__name__ == "HfApi":
        try:
            return validate_manifest(read_bucket_json(INDEX_FILES["manifest"], os.environ.get("HF_TOKEN")))
        except (FileNotFoundError, OSError, ValueError):
            return empty_manifest()
    try:
        if not api.file_exists(repo_id=repo_id, repo_type="dataset", filename=MANIFEST_NAME):
            return empty_manifest()
    except RepositoryNotFoundError:
        return empty_manifest()
    path = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=MANIFEST_NAME)
    return validate_manifest(load_json(Path(path)))


def shard_for_key(key: str, shard_count: int) -> int:
    return shared.hash_for_key(key, shard_count)


def build_queue(records, revisions, manifest, *, repo="", extension="", exact_path="", limit=0,
                  retry_failed=False, force=False, bucket_migrate=False,
                  shard_count=1, shard_index=0) -> list[dict]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid reader asset shard")
    selected = []
    extension = extension.lower().lstrip(".")
    files = manifest.get("files", {})
    for record in records:
        source_repo = str(record.get("Repo") or "")
        ext = str(record.get("Extension") or "").lower().lstrip(".")
        if (repo and source_repo != repo) or (extension and ext != extension):
            continue
        path = relative_path(record)
        if ext in {"htm", "html"} and any(
                part.lower() == ".files" or part.lower().endswith(".files") or part.lower().endswith("_files")
                for part in path.split("/")):
            continue
        contract = (bucket_conversion_contract(source_repo, path, ext, int(record.get("Size") or 0))
                    if bucket_migrate else
                    source_conversion_contract(source_repo, path, ext, int(record.get("Size") or 0)))
        if contract is None:
            continue
        spreadsheet_pages = contract[0] == SPREADSHEET_PAGE_PROFILE
        revision = str(revisions.get(source_repo) or "")
        if not revision:
            continue
        if exact_path and path != exact_path:
            continue
        key = asset_key(source_repo, path)
        if shard_for_key(key, shard_count) != shard_index:
            continue
        profile, reader_mode, output_name = contract
        existing = files.get(key, {})
        chapter_path = existing.get("chapter_manifest") if isinstance(existing, dict) else None
        chapter_bucket_path_missing_prefix = (
            existing.get("chapter_bucket") == READER_ASSETS_BUCKET
            and isinstance(chapter_path, str)
            and not chapter_path.startswith("ebook-chapters/")
        )
        if (bucket_migrate and not force and existing.get("status") == "ready"
                and existing.get("bucket") == READER_ASSETS_BUCKET
                and not chapter_bucket_path_missing_prefix):
            continue
        manual = str(existing.get("profile") or "").startswith("manual-")
        if not force and existing.get("status") == "ready" and manual:
            continue
        # Source revisions are repository-wide. A new commit does not mean this
        # particular file changed, and the metadata feed has no file digest.
        # Reuse a ready artifact unless an explicit rebuild was requested.
        if not bucket_migrate and not force and existing.get("status") == "ready" and existing.get("profile") == profile:
            retryable_update = (
                retry_failed and existing.get("failed_source_revision") == revision
                and existing.get("failed_profile") == profile
            )
            missing_chapters = (
                needs_epub_chapters(ext, reader_mode, int(record.get("Size") or 0))
                and (
                    existing.get("chapter_bundle_profile") != EPUB_CHAPTER_PROFILE
                    or not existing.get("chapter_manifest")
                    or existing.get("chapter_bundle_error")
                )
            )
            if not retryable_update and not missing_chapters:
                continue
        if (bucket_migrate and contract[1] == "pdf" and not spreadsheet_pages
                and not force and existing.get("status") == "ready"
                and existing.get("profile") == profile
                and existing.get("bucket") in {None, ""}):
            continue
        if not bucket_migrate and not force and existing.get("status") == "failed" and existing.get("profile") == profile and not retry_failed:
            continue
        failed_current = (existing.get("failed_source_revision") == revision
                          and existing.get("failed_profile") == profile)
        if not bucket_migrate and not force and failed_current and not retry_failed:
            continue
        item = {
            "key": key, "repo": source_repo, "path": path, "extension": ext,
            "source_revision": revision, "source_bytes": record.get("Size") or 0,
            "source_url": source_url(source_repo, revision, path), "profile": profile,
            "reader_mode": reader_mode, "output_name": output_name,
            "conversion_dependencies": conversion_dependencies(ext, reader_mode),
        }
        if bucket_migrate and reader_mode == "pdf":
            if spreadsheet_pages:
                item["page_stream"] = True
        selected.append(item)
    priority = {"pdf": 9, "tif": 0, "tiff": 0, "epub": 1, "mobi": 1, "azw3": 1, "fb2": 1, "odt": 1, "rtf": 1, "chm": 1, "djvu": 2,
                  "doc": 3, "docx": 3, "htm": 3, "html": 3, "txt": 3, "md": 3, "markdown": 3,
                  "jpg": 3, "jpeg": 3, "png": 3, "gif": 3, "bmp": 3, "webp": 3, "caj": 3, "kdh": 3,
                 "ppt": 3, "pptx": 3, "pps": 3, "odp": 3, "xls": 3, "xlsx": 3, "csv": 3, "ods": 3, "wps": 3,
                 "mht": 3, "mhtml": 3, "ps": 3,
                   "ape": 3, "wma": 3, "amr": 3, "asx": 4,
                   "flac": 3, "m4a": 3, "mpga": 3, "wav": 3,
                   "flv": 4, "f4v": 4, "rm": 4, "rmvb": 4, "mkv": 4, "avi": 4,
                   "mpg": 4, "mpeg": 4, "mts": 4, "ts": 4, "wmv": 4, "mp4": 4, "mov": 4, "swf": 4}
    selected.sort(key=lambda item: (priority[item["extension"]], item["repo"], item["path"]))
    return selected[:limit] if limit > 0 else selected


def active_keys(records) -> list[str]:
    keys = []
    for record in records:
        repo = str(record.get("Repo") or "")
        extension = str(record.get("Extension") or "").lower().lstrip(".")
        path = relative_path(record)
        if extension in {"htm", "html"} and any(
                part.lower() == ".files" or part.lower().endswith(".files") or part.lower().endswith("_files")
                for part in path.split("/")):
            continue
        if source_conversion_contract(repo, path, extension, int(record.get("Size") or 0)) is not None:
            keys.append(asset_key(repo, path))
    return sorted(set(keys))


def reusable_objects(manifest: dict) -> dict:
    objects = {}
    entries = list(manifest.get("files", {}).items())
    entries.extend(("", entry) for entry in manifest.get("orphans", {}).values())
    for key, entry in entries:
        if (entry.get("status", "ready") != "ready" or not entry.get("source_sha256")
                or not entry.get("path") or not entry.get("sha256")
                or not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0
                or not entry.get("reader_mode")):
            continue
        extension = entry.get("source_extension", "")
        if not key and extension in {"htm", "html"}:
            continue
        identity = reusable_object_key(
            entry["source_sha256"], entry.get("profile", ""), extension=extension,
            source_revision=entry.get("source_revision", ""), key=key,
        )
        objects[identity] = {
            field: entry[field]
            for field in ("path", "bytes", "sha256", "reader_mode") if field in entry
        }
    return dict(sorted(objects.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--output", type=Path, default=Path("output/reader-assets/queue.json"))
    parser.add_argument("--repo", default="")
    parser.add_argument("--extension", default="")
    parser.add_argument("--path", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bucket-migrate", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid reader asset shard")
    records = decode_search_payload(load_json(args.search_data))
    revisions = load_json(args.revisions)
    if args.manifest:
        manifest = validate_manifest(load_json(args.manifest))
    else:
        manifest = remote_manifest(HfApi(token=os.environ.get("HF_TOKEN") or None), args.assets_repo)
    queue = build_queue(records, revisions, manifest, repo=args.repo, extension=args.extension, exact_path=args.path,
                        limit=args.limit, retry_failed=args.retry_failed, force=args.force,
                         bucket_migrate=args.bucket_migrate,
                        shard_count=args.shard_count, shard_index=args.shard_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    current_keys = active_keys(records)
    args.output.write_bytes(canonical_json({
        "version": 1,
        "items": queue,
        "active_keys": current_keys,
        "stale_keys": sorted(set(manifest.get("files", {})) - set(current_keys)),
        "objects": reusable_objects(manifest),
        "force_rebuild": bool(args.force),
        "authoritative_snapshot": not bool(args.repo or args.extension or args.path or args.bucket_migrate),
        "bucket_migration": bool(args.bucket_migrate),
    }, pretty=True))
    print(f"queued {len(queue)} reader asset conversion(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
