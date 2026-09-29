#!/usr/bin/env python3
"""Atomically publish converted Reader Assets and their manifest."""

import argparse
import json
import os
import random
import tempfile
import time
from datetime import date
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, sync_bucket
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

try:
    from .build_reader_assets_index import encode_index
    from .pdf_range_state import remote_state
    from .reader_bucket import INDEX_FILES, read_bytes as read_bucket_bytes, read_json as read_bucket_json, stage_index
    from .reader_lifecycle import merge as merge_lifecycle, staging_record
    from .reader_assets import (
        MANIFEST_NAME, READER_ASSETS_REPO, canonical_json, empty_manifest, load_json,
        reusable_object_key, validate_manifest, validate_storage_path,
        READER_ASSETS_BUCKET, READER_STAGING_BUCKET,
    )
except ImportError:
    from build_reader_assets_index import encode_index
    from pdf_range_state import remote_state
    from reader_bucket import INDEX_FILES, read_bytes as read_bucket_bytes, read_json as read_bucket_json, stage_index
    from reader_lifecycle import merge as merge_lifecycle, staging_record
    from reader_assets import (
        MANIFEST_NAME, READER_ASSETS_REPO, canonical_json, empty_manifest, load_json,
        reusable_object_key, validate_manifest, validate_storage_path,
        READER_ASSETS_BUCKET, READER_STAGING_BUCKET,
    )

try:
    from . import shared
except ImportError:
    import shared

SIDECAR_NAME = "reader_assets.json.gz"
BUCKET_READER_MODES = {"docx", "html", "text", "markdown", "image", "foliate", "epub"}
BUCKET_STAGING_MODES = {"pdf"}


def bundle_is_published(manifest: dict, data: dict) -> bool:
    """Recognize a commit that succeeded remotely before its response timed out."""
    entries = manifest.get("files", {})
    for result in data.get("results", []):
        key = result.get("key")
        current = entries.get(key)
        if not key or not current or current.get("status") != result.get("status"):
            return False
        if result.get("status") == "ready":
            for field in ("source_revision", "source_sha256", "source_extension", "profile",
                          "reader_mode", "path", "bytes", "sha256", "chapter_manifest",
                          "chapter_bundle_profile", "chapter_bundle_error", "fallback_path"):
                expected = result.get(field)
                if current.get(field) != expected:
                    return False
            if result.get("bucket") is not None and current.get("bucket") != result.get("bucket"):
                return False
        elif current.get("error") != result.get("error"):
            return False
    return bool(data.get("results"))


def bucket_paths(data: dict, bundle: Path | None = None) -> list[str]:
    paths = set()
    for result in data.get("results", []):
        if result.get("status") != "ready":
            continue
        if (result.get("reader_mode") in BUCKET_READER_MODES | BUCKET_STAGING_MODES
                and isinstance(result.get("path"), str)):
            paths.add(result["path"])
        if result.get("chapter_manifest"):
            if bundle is not None:
                root = bundle / Path(result["chapter_manifest"]).parent
                if root.is_dir():
                    paths.update((Path(result["chapter_manifest"]).parent / item.relative_to(root)).as_posix()
                                 for item in root.rglob("*") if item.is_file())
    return sorted(paths)


def staging_paths(data: dict) -> list[str]:
    return sorted({result["path"] for result in data.get("results", [])
                   if result.get("status") == "ready"
                   and result.get("reader_mode") in BUCKET_STAGING_MODES
                   and isinstance(result.get("path"), str)})


def _sync_bucket_with_retry(local_dir: str, token: str | None, paths: list[str],
                            bucket: str = READER_ASSETS_BUCKET, max_attempts: int = 8) -> None:
    if not paths:
        return
    for attempt in range(max_attempts):
        try:
            sync_bucket(local_dir, f"hf://buckets/{bucket}", include=paths,
                        token=token, quiet=False)
            return
        except HfHubHTTPError as exc:
            status = shared.hf_status_code(exc)
            if status not in {429, 500, 502, 503, 504} or attempt + 1 == max_attempts:
                raise
            delay = shared.hf_retry_delay(attempt) + random.uniform(0, 2)
            print(f"transient Reader bucket upload error ({status}); retrying in {delay:.1f}s")
            time.sleep(delay)


def orphan_entry(entry: dict) -> dict:
    orphan = {field: entry[field] for field in (
        "source_sha256", "profile", "reader_mode", "path", "bytes", "sha256"
    ) if field in entry}
    orphan["since"] = date.today().isoformat()
    return orphan


def file_sha256(path: Path) -> str:
    return shared.hash_file(path)[0]


def remote_manifest(api: HfApi, repo_id: str, revision: str | None = None) -> dict:
    if isinstance(api, HfApi):
        try:
            return validate_manifest(read_bucket_json(INDEX_FILES["manifest"], os.environ.get("HF_TOKEN")))
        except (FileNotFoundError, OSError, ValueError):
            pass
    try:
        if not api.file_exists(
                repo_id=repo_id, repo_type="dataset", filename=MANIFEST_NAME, revision=revision):
            return empty_manifest()
    except RepositoryNotFoundError:
        return empty_manifest()
    path = api.hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=MANIFEST_NAME, revision=revision,
    )
    return validate_manifest(load_json(Path(path)))


def remote_pdf_manifest(api: HfApi, repo_id: str, revision: str | None = None) -> dict:
    if isinstance(api, HfApi):
        try:
            data = read_bucket_json(INDEX_FILES["pdf"], os.environ.get("HF_TOKEN"))
            if data.get("version") != 1 or not isinstance(data.get("files"), dict):
                raise ValueError("invalid PDF asset manifest")
            return data
        except (FileNotFoundError, OSError, ValueError):
            pass
    try:
        path = api.hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="pdf_manifest.json", revision=revision,
        )
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) != 404:
            raise
        return {"version": 1, "files": {}}
    data = load_json(Path(path))
    if data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError("invalid PDF asset manifest")
    return data


def remote_pdf_ocr_manifest(api: HfApi, repo_id: str, revision: str | None = None) -> dict:
    if isinstance(api, HfApi):
        try:
            data = read_bucket_json(INDEX_FILES["ocr"], os.environ.get("HF_TOKEN"))
            if data.get("version") != 1 or not isinstance(data.get("files"), dict):
                raise ValueError("invalid PDF OCR manifest")
            return data
        except (FileNotFoundError, OSError, ValueError):
            pass
    try:
        path = api.hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="pdf_ocr_manifest.json", revision=revision,
        )
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) != 404:
            raise
        return {"version": 1, "files": {}}
    data = load_json(Path(path))
    if data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError("invalid PDF OCR manifest")
    return data


def build_publish(api: HfApi, repo_id: str, bundle: Path, revision: str | None = None, range_manifest: dict | None = None):
    data = load_json(bundle / "bundle.json")
    if data.get("version") != 1 or not isinstance(data.get("results"), list):
        raise ValueError("invalid reader asset bundle")
    manifest = remote_manifest(api, repo_id, revision)
    pdf_manifest = remote_pdf_manifest(api, repo_id, revision)
    ocr_manifest = remote_pdf_ocr_manifest(api, repo_id, revision)
    files = dict(manifest["files"])
    orphans = dict(manifest.get("orphans", {}))
    reusable = {}
    candidates = list(files.items()) + [("", entry) for entry in orphans.values()]
    for key, candidate in candidates:
        if (candidate.get("status", "ready") == "ready" and candidate.get("source_sha256")
                and candidate.get("profile") and candidate.get("path") and candidate.get("sha256")
                and isinstance(candidate.get("bytes"), int) and candidate["bytes"] > 0
                and candidate.get("reader_mode")):
            extension = candidate.get("source_extension", "")
            if not key and extension in {"htm", "html"}:
                continue
            identity = reusable_object_key(
                candidate["source_sha256"], candidate["profile"], extension=extension,
                source_revision=candidate.get("source_revision", ""), key=key,
            )
            reusable[identity] = candidate
    artifacts = {}
    for result in data["results"]:
        entry = {key: value for key, value in result.items() if key != "key"}
        if result.get("status") == "ready":
            validate_storage_path(result.get("path"))
            remote = None
            if not data.get("force_rebuild"):
                identity = reusable_object_key(
                    result.get("source_sha256", ""), result.get("profile", ""),
                    extension=result.get("source_extension", ""),
                    source_revision=result.get("source_revision", ""), key=result.get("key", ""),
                )
                remote = reusable.get(identity)
            if remote:
                for field in ("path", "bytes", "sha256", "reader_mode", "bucket"):
                    if field in remote:
                        entry[field] = remote[field]
                entry.pop("reused", None)
            else:
                artifact = bundle / result["path"]
                if not artifact.is_file() or artifact.stat().st_size != result["bytes"]:
                    raise ValueError(f"missing or invalid artifact for {result['key']}")
                if file_sha256(artifact) != result["sha256"]:
                    raise ValueError(f"artifact digest mismatch for {result['key']}")
                artifacts[result["path"]] = str(artifact)
                if result.get("chapter_manifest"):
                    prefix = Path(result["chapter_manifest"]).parent
                    root = bundle / prefix
                    if not root.is_dir():
                        raise ValueError(f"missing EPUB chapter bundle for {result['key']}")
                    for child in sorted(root.rglob("*")):
                        if child.is_file():
                            path = (prefix / child.relative_to(root)).as_posix()
                            validate_storage_path(path)
                            artifacts[path] = str(child)
            if result.get("reader_mode") in BUCKET_READER_MODES:
                entry["bucket"] = READER_ASSETS_BUCKET
            if result.get("reader_mode") in BUCKET_STAGING_MODES:
                entry["bucket"] = READER_STAGING_BUCKET
                entry["bucket_staging"] = True
            if result.get("chapter_manifest"):
                entry["chapter_bucket"] = READER_ASSETS_BUCKET
        elif result.get("status") != "failed":
            raise ValueError("unknown reader asset result status")
        elif files.get(result["key"], {}).get("status") == "ready":
            previous = dict(files[result["key"]])
            previous.update({
                "failed_source_revision": result.get("source_revision", ""),
                "failed_profile": result.get("profile", ""),
                "failed_error": result.get("error", "RuntimeError"),
            })
            files[result["key"]] = previous
            continue
        entry.pop("reused", None)
        previous = files.get(result["key"], {})
        if (previous.get("status") == "ready" and previous.get("path")
                and previous["path"] != entry.get("path")):
            orphans.setdefault(previous["path"], orphan_entry(previous))
        files[result["key"]] = entry
        orphans.pop(entry.get("path", ""), None)
        if data.get("force_rebuild") and entry.get("status") == "ready":
            for key, candidate in list(files.items()):
                if candidate.get("status") == "ready" and candidate.get("path") == entry["path"]:
                    files[key] = {
                        **candidate,
                        "bytes": entry["bytes"],
                        "sha256": entry["sha256"],
                        "reader_mode": entry["reader_mode"],
                    }
            for path, candidate in list(orphans.items()):
                if candidate.get("path", path) == entry["path"]:
                    orphans[path] = {
                        **candidate,
                        "bytes": entry["bytes"],
                        "sha256": entry["sha256"],
                        "reader_mode": entry["reader_mode"],
                    }
    active_keys = set(data.get("active_keys", []))
    if data.get("authoritative_snapshot") is True:
        for key in set(files) - active_keys:
            removed = files.pop(key)
            if removed.get("status") == "ready" and removed.get("path"):
                orphans.setdefault(removed["path"], orphan_entry(removed))
    referenced = {entry.get("path") for entry in files.values() if entry.get("status") == "ready"}
    orphans = {path: entry for path, entry in orphans.items() if path not in referenced}
    updated = {
        "version": 1,
        "files": dict(sorted(files.items())),
        "orphans": dict(sorted(orphans.items())),
    }
    validate_manifest(updated)
    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=source)
        for path, source in sorted(artifacts.items())
    ]
    operations.append(CommitOperationAdd(path_in_repo=MANIFEST_NAME, path_or_fileobj=canonical_json(updated, pretty=True)))
    operations.append(CommitOperationAdd(
        path_in_repo=SIDECAR_NAME, path_or_fileobj=encode_index(updated, pdf_manifest, range_manifest, ocr_manifest)))
    return updated, operations


def publish_bundle(api: HfApi, repo_id: str, bundle: Path, *, max_attempts: int = 20) -> tuple[dict, int]:
    data = load_json(bundle / "bundle.json")
    result_keys = {result.get("key") for result in data.get("results", []) if result.get("key")}
    if type(api) is HfApi:
        return publish_bucket_bundle(api, repo_id, bundle, data, result_keys, max_attempts)
    baseline = None
    objects_uploaded = False
    bucket_uploaded = False
    for attempt in range(max_attempts):
        try:
            revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
            current = remote_manifest(api, repo_id, revision)
            if attempt and bundle_is_published(current, data):
                return current, 0
            current_entries = {key: current["files"].get(key) for key in result_keys}
            if baseline is None:
                baseline = current_entries
            elif current_entries != baseline:
                raise RuntimeError("reader asset key changed during publication retry")
            dataset_objects = ([result for result in data.get("results", [])
                                if result.get("status") == "ready"] if type(api) is not HfApi else
                               [result for result in data.get("results", [])
                                if result.get("status") == "ready"
                                and result.get("reader_mode") not in BUCKET_READER_MODES | BUCKET_STAGING_MODES
                                and not (result.get("reader_mode") == "foliate"
                                         and result.get("chapter_manifest"))])
            if not objects_uploaded and dataset_objects:
                for result in dataset_objects:
                    artifact = bundle / result["path"]
                    if artifact.is_file():
                        api.upload_file(path_or_fileobj=str(artifact), path_in_repo=result["path"],
                                        repo_id=repo_id, repo_type="dataset",
                                        commit_message="Upload non-static Reader Asset")
                revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
                objects_uploaded = True
            if not bucket_uploaded:
                bucket_token = os.environ.get("HF_TOKEN")
                if bucket_token and type(api) is HfApi:
                    staging = set(staging_paths(data))
                    _sync_bucket_with_retry(str(bundle), bucket_token,
                                            [path for path in bucket_paths(data, bundle) if path not in staging])
                    _sync_bucket_with_retry(str(bundle), bucket_token, sorted(staging), READER_STAGING_BUCKET)
                bucket_uploaded = True
            range_manifest = remote_state(api, repo_id, revision)
            manifest, operations = build_publish(api, repo_id, bundle, revision, range_manifest)
            operations = [operation for operation in operations
                          if operation.path_in_repo in {MANIFEST_NAME, SIDECAR_NAME}]
            api.create_commit(
                repo_id=repo_id, repo_type="dataset", operations=operations,
                commit_message="Update reader assets", parent_commit=revision,
            )
            index_root = bundle / INDEX_FILES["manifest"].split("/", 1)[0]
            stage_index(index_root.parent, "manifest", canonical_json(manifest, pretty=True))
            sidecar_operation = next(operation for operation in operations
                                     if operation.path_in_repo == SIDECAR_NAME)
            stage_index(index_root.parent, "sidecar", sidecar_operation.path_or_fileobj)
            bucket_token = os.environ.get("HF_TOKEN")
            lifecycle_path = index_root.parent / INDEX_FILES["lifecycle"]
            try:
                lifecycle = read_bucket_json(INDEX_FILES["lifecycle"], bucket_token)
            except (FileNotFoundError, OSError, ValueError):
                lifecycle = {"version": 1, "files": {}}
            lifecycle_updates = []
            for result in data.get("results", []):
                if result.get("status") != "ready":
                    continue
                if not (result.get("reader_mode") in BUCKET_READER_MODES | BUCKET_STAGING_MODES
                        or result.get("chapter_manifest")):
                    continue
                result_with_paths = dict(result)
                result_with_paths["bucket_staging"] = result.get("reader_mode") in BUCKET_STAGING_MODES
                result_with_paths["bucket_paths"] = bucket_paths({"results": [result]}, bundle)
                lifecycle_updates.append(staging_record(result_with_paths))
            lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
            lifecycle_path.write_text(json.dumps(merge_lifecycle(lifecycle, lifecycle_updates),
                                                 ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                      encoding="utf-8")
            if bucket_token and type(api) is HfApi:
                _sync_bucket_with_retry(str(index_root.parent), bucket_token,
                                        ["reader-index/**"])
            return manifest, len(result_keys)
        except HfHubHTTPError as exc:
            status = shared.hf_status_code(exc)
            if status in {429, 500, 502, 503, 504} and attempt + 1 < max_attempts:
                delay = shared.hf_retry_delay(attempt) + random.uniform(0, 2)
                print(f"transient Hugging Face upload error ({status}); retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            if status not in {409, 412} or attempt + 1 == max_attempts:
                raise
            print(f"reader asset parent changed; rebuilding publication ({attempt + 2}/{max_attempts})")
            time.sleep(random.uniform(0.5, min(8.0, 0.5 * (attempt + 1))))
    raise RuntimeError("reader asset publication retry limit reached")


def publish_bucket_bundle(api: HfApi, repo_id: str, bundle: Path, data: dict,
                          result_keys: set[str], max_attempts: int) -> tuple[dict, int]:
    """Publish Reader objects and all indexes atomically in the bucket namespace."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    for attempt in range(max_attempts):
        try:
            current = remote_manifest(api, repo_id)
            if attempt and bundle_is_published(current, data):
                return current, 0
            range_manifest = remote_state(api, repo_id)
            manifest, operations = build_publish(api, repo_id, bundle, None, range_manifest)
            paths = bucket_paths(data, bundle)
            if paths:
                staging = set(staging_paths(data))
                _sync_bucket_with_retry(str(bundle), token, [path for path in paths if path not in staging])
                _sync_bucket_with_retry(str(bundle), token, sorted(staging), READER_STAGING_BUCKET)
            lifecycle = {"version": 1, "files": {}, "orphans": {}}
            try:
                lifecycle = read_bucket_json(INDEX_FILES["lifecycle"], token)
            except (FileNotFoundError, OSError, ValueError):
                pass
            updates = []
            for result in data.get("results", []):
                if result.get("status") != "ready":
                    continue
                if not (result.get("reader_mode") in BUCKET_READER_MODES | BUCKET_STAGING_MODES
                        or result.get("chapter_manifest")):
                    continue
                item = dict(result)
                item["bucket_staging"] = result.get("reader_mode") in BUCKET_STAGING_MODES
                item["bucket_paths"] = bucket_paths({"results": [result]}, bundle)
                updates.append(staging_record(item))
            lifecycle = merge_lifecycle(lifecycle, updates)
            with tempfile.TemporaryDirectory(prefix="reader-index-") as root:
                index_root = Path(root) / "reader-index"
                index_root.mkdir(parents=True, exist_ok=True)
                (index_root / "manifest.json").write_bytes(canonical_json(manifest, pretty=True))
                sidecar = next(operation.path_or_fileobj for operation in operations
                               if operation.path_in_repo == SIDECAR_NAME)
                if isinstance(sidecar, str):
                    sidecar = Path(sidecar).read_bytes()
                (index_root / SIDECAR_NAME).write_bytes(sidecar)
                (index_root / INDEX_FILES["lifecycle"].rsplit("/", 1)[-1]).write_text(
                    json.dumps(lifecycle, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
                sync_bucket(root, f"hf://buckets/{READER_ASSETS_BUCKET}",
                            include=["reader-index/**"], token=token, quiet=False)
            return manifest, len(result_keys)
        except HfHubHTTPError as exc:
            status = shared.hf_status_code(exc)
            if status not in {409, 412, 429, 500, 502, 503, 504} or attempt + 1 == max_attempts:
                raise
            delay = shared.hf_retry_delay(attempt) + random.uniform(0, 2)
            print(f"transient Reader bucket publication error ({status}); retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("Reader bucket publication retry limit reached")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("output/reader-assets/bundle"))
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.dry_run:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token or None)
    if args.dry_run:
        manifest, operations = build_publish(api, args.assets_repo, args.bundle)
        print(f"dry run: validated {len(operations) - 2} artifact(s), {len(manifest['files'])} manifest entries")
        return 0
    _, artifact_count = publish_bundle(api, args.assets_repo, args.bundle)
    print(f"published {artifact_count} artifact(s) to {args.assets_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
