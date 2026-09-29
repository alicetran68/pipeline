#!/usr/bin/env python3
"""Migrate existing CHM-derived EPUB assets into v8 bucket chapter bundles.

This command is designed for GitHub Actions shards. It never runs the CHM
converter: the already published EPUB asset is the chapterization input.
"""

import argparse
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import epub_chapters, publish_reader_assets, shared
    from .reader_assets import (
        EPUB_CHAPTER_BUNDLE_DIR, EPUB_CHAPTER_PROFILE, MANIFEST_NAME,
        READER_ASSETS_REPO, READER_EBOOK_BUCKET, canonical_json, load_json,
        validate_manifest,
    )
except ImportError:
    import epub_chapters
    import publish_reader_assets
    import shared
    from reader_assets import (
        EPUB_CHAPTER_BUNDLE_DIR, EPUB_CHAPTER_PROFILE, MANIFEST_NAME,
        READER_ASSETS_REPO, READER_EBOOK_BUCKET, canonical_json, load_json,
        validate_manifest,
    )


def migration_candidates(manifest: dict, *, shard_count: int, shard_index: int,
                         limit: int = 0) -> list[tuple[str, dict]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid migration shard")
    selected = []
    for key, entry in manifest.get("files", {}).items():
        if (entry.get("status") != "ready" or entry.get("source_extension") != "chm"
                or entry.get("reader_mode") != "epub"
                or (entry.get("chapter_manifest")
                    and entry.get("chapter_bucket") == READER_EBOOK_BUCKET)):
            continue
        if shared.hash_for_key(key, shard_count) != shard_index:
            continue
        selected.append((key, entry))
    selected.sort(key=lambda item: item[0])
    return selected[:limit] if limit > 0 else selected


def build_bundle(api: HfApi, repo_id: str, output: Path, *, revision: str,
                 shard_count: int, shard_index: int, limit: int = 0) -> int:
    manifest_path = api.hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=MANIFEST_NAME, revision=revision,
    )
    manifest = validate_manifest(load_json(Path(manifest_path)))
    candidates = migration_candidates(
        manifest, shard_count=shard_count, shard_index=shard_index, limit=limit,
    )
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, (key, entry) in enumerate(candidates, 1):
        print(f"[{number}/{len(candidates)}] {key}", flush=True)
        try:
            epub_path = Path(api.hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=entry["path"], revision=revision,
            ))
            with tempfile.TemporaryDirectory(prefix="chm-chapters-") as temporary:
                staged = Path(temporary) / "chapter-bundle"
                epub_chapters.build_bundle(epub_path, staged)
                bundle_version = epub_chapters.bundle_version(staged)
                object_root = Path(*Path(entry["path"]).parts[:3])
                chapter_parent = object_root / bundle_version / (
                    f"{Path(entry['path']).parent.name}-{EPUB_CHAPTER_PROFILE}"
                )
                chapter_root = output / chapter_parent / EPUB_CHAPTER_BUNDLE_DIR
                chapter_root.parent.mkdir(parents=True, exist_ok=True)
                if chapter_root.exists():
                    shutil.rmtree(chapter_root)
                shutil.move(staged, chapter_root)
            result = {"key": key, **entry}
            result.pop("chapter_bundle_error", None)
            result.pop("chapter_bucket", None)
            result["chapter_manifest"] = (
                chapter_parent / EPUB_CHAPTER_BUNDLE_DIR / "chapter-manifest.json"
            ).as_posix()
            result["chapter_bundle_profile"] = EPUB_CHAPTER_PROFILE
        except Exception as exc:
            result = {"key": key, **entry}
            result.pop("chapter_manifest", None)
            result.pop("chapter_bucket", None)
            result["chapter_bundle_profile"] = EPUB_CHAPTER_PROFILE
            result["chapter_bundle_error"] = f"{type(exc).__name__}: {exc}"
            print(f"warning: {key}: {result['chapter_bundle_error']}", flush=True)
        results.append(result)
    (output / "bundle.json").write_bytes(canonical_json({
        "version": 1,
        "results": results,
        "skip_object_upload": True,
        "skip_bucket_sync": True,
    }, pretty=True))
    print(f"prepared {len(results)} CHM chapter migration(s)", flush=True)
    return len(results)


def upload_bucket_files(api: HfApi, bundle: Path, token: str, *, max_attempts: int = 10) -> int:
    files = sorted(path for path in bundle.rglob("*")
                   if path.is_file() and path.name != "bundle.json")
    additions = [
        (str(path), "ebook-chapters/" + path.relative_to(bundle).as_posix())
        for path in files
    ]
    if not additions:
        return 0
    for attempt in range(max_attempts):
        try:
            api.batch_bucket_files(READER_EBOOK_BUCKET, add=additions, token=token)
            return len(additions)
        except HfHubHTTPError as exc:
            status = shared.hf_status_code(exc)
            if status not in {429, 500, 502, 503, 504} or attempt + 1 == max_attempts:
                raise
            delay = shared.hf_retry_delay(attempt, cap=180) + random.uniform(0, 5)
            print(f"transient bucket upload error ({status}); retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("bucket upload retry limit reached")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    parser.add_argument("--bundle", type=Path, default=Path("output/chm-chapters"))
    parser.add_argument("--shard-count", type=int, default=24)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    revision = api.repo_info(repo_id=args.assets_repo, repo_type="dataset").sha
    manifest_path = api.hf_hub_download(
        repo_id=args.assets_repo, repo_type="dataset", filename=MANIFEST_NAME, revision=revision,
    )
    manifest = validate_manifest(load_json(Path(manifest_path)))
    candidates = migration_candidates(
        manifest, shard_count=args.shard_count, shard_index=args.shard_index, limit=args.limit,
    )
    if args.dry_run:
        print(f"would migrate {len(candidates)} CHM EPUB asset(s) from {revision}")
        return 0
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not candidates:
        print("no CHM EPUB assets require bucket chapter migration")
        return 0
    if args.bundle.exists():
        shutil.rmtree(args.bundle)
    build_bundle(
        api, args.assets_repo, args.bundle, revision=revision,
        shard_count=args.shard_count, shard_index=args.shard_index, limit=args.limit,
    )
    uploaded = upload_bucket_files(api, args.bundle, token)
    print(f"uploaded {uploaded} chapter files", flush=True)
    publish_reader_assets.publish_bundle(api, args.assets_repo, args.bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
