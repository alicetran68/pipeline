#!/usr/bin/env python3
"""Report or delete Reader Assets that have remained unreferenced past a grace period."""

import argparse
import os
from datetime import date, timedelta

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

try:
    from .build_reader_assets_index import encode_index
    from .publish_reader_assets import remote_manifest, remote_pdf_manifest, remote_pdf_ocr_manifest
    from .reader_assets import MANIFEST_NAME, READER_ASSETS_REPO, canonical_json
except ImportError:
    from build_reader_assets_index import encode_index
    from publish_reader_assets import remote_manifest, remote_pdf_manifest, remote_pdf_ocr_manifest
    from reader_assets import MANIFEST_NAME, READER_ASSETS_REPO, canonical_json

SIDECAR_NAME = "reader_assets.json.gz"


def expired_orphans(manifest: dict, today: date, grace_days: int, limit: int) -> list[str]:
    referenced = {entry.get("path") for entry in manifest["files"].values() if entry.get("status") == "ready"}
    cutoff = today - timedelta(days=grace_days)
    expired = []
    for path, entry in sorted(manifest.get("orphans", {}).items()):
        if path in referenced:
            continue
        try:
            since = date.fromisoformat(entry["since"])
        except (KeyError, TypeError, ValueError):
            continue
        if since <= cutoff:
            expired.append(path)
    return expired[:limit]


def build_prune(manifest: dict, paths: list[str], pdf_manifest: dict | None = None,
                ocr_manifest: dict | None = None):
    updated = {
        "version": manifest["version"],
        "files": manifest["files"],
        "orphans": {path: entry for path, entry in manifest.get("orphans", {}).items() if path not in paths},
    }
    operations = [CommitOperationDelete(path_in_repo=path) for path in paths]
    operations.extend([
        CommitOperationAdd(path_in_repo=MANIFEST_NAME, path_or_fileobj=canonical_json(updated, pretty=True)),
        CommitOperationAdd(path_in_repo=SIDECAR_NAME, path_or_fileobj=encode_index(updated, pdf_manifest, ocr_manifest)),
    ])
    return updated, operations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    parser.add_argument("--grace-days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grace_days < 0 or args.limit < 1:
        raise ValueError("grace-days must be non-negative and limit must be positive")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    revision = api.repo_info(repo_id=args.assets_repo, repo_type="dataset").sha
    manifest = remote_manifest(api, args.assets_repo, revision)
    pdf_manifest = remote_pdf_manifest(api, args.assets_repo, revision)
    ocr_manifest = remote_pdf_ocr_manifest(api, args.assets_repo, revision)
    paths = expired_orphans(manifest, date.today(), args.grace_days, args.limit)
    print(f"found {len(paths)} Reader Asset orphan(s) eligible for deletion")
    for path in paths:
        print(path)
    if not args.apply or not paths:
        return 0
    _, operations = build_prune(manifest, paths, pdf_manifest, ocr_manifest)
    api.create_commit(
        repo_id=args.assets_repo,
        repo_type="dataset",
        operations=operations,
        commit_message="Prune orphaned reader assets",
        parent_commit=revision,
    )
    print(f"deleted {len(paths)} orphaned Reader Asset object(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
