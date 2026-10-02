#!/usr/bin/env python3
"""Atomically publish converted Reader Assets and their manifest."""

import argparse
import concurrent.futures
import json
import mimetypes
import os
import random
import shutil
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
EBOOK_CHAPTERS_PREFIX = "ebook-chapters/"


def bucket_chapter_path(path: str) -> str:
    """Store chapter bundles under the namespace accepted by Reader APIs."""
    return path if path.startswith(EBOOK_CHAPTERS_PREFIX) else EBOOK_CHAPTERS_PREFIX + path


def local_chapter_path(path: str) -> str:
    return path[len(EBOOK_CHAPTERS_PREFIX):] if path.startswith(EBOOK_CHAPTERS_PREFIX) else path


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
                if field == "chapter_manifest" and data.get("bucket_migration") and expected:
                    expected = bucket_chapter_path(expected)
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
        if result.get("reader_mode") == "pdf" and result.get("pdf_to_dataset") is True:
            continue
        if (result.get("reader_mode") in BUCKET_READER_MODES | BUCKET_STAGING_MODES
                and isinstance(result.get("path"), str)):
            paths.add(result["path"])
            if result.get("page_stream") and bundle is not None:
                root = (bundle / Path(result["path"]).parent / "pages")
                if root.is_dir():
                    count = int(result.get("page_count") or 0)
                    paths.update((Path(result["path"]).parent / "pages"
                                  / f"page-{number:06d}.webp").as_posix()
                                 for number in range(1, count + 1))
        if result.get("chapter_manifest"):
            if bundle is not None:
                local_manifest = local_chapter_path(result["chapter_manifest"])
                remote_prefix = Path(bucket_chapter_path(result["chapter_manifest"])).parent
                root = bundle / Path(local_manifest).parent
                if root.is_dir():
                    paths.update((remote_prefix / item.relative_to(root)).as_posix()
                                 for item in root.rglob("*") if item.is_file())
    return sorted(paths)


def staging_paths(data: dict) -> list[str]:
    return sorted({result["path"] for result in data.get("results", [])
                   if result.get("status") == "ready"
                   and result.get("reader_mode") in BUCKET_STAGING_MODES
                   and not result.get("page_stream")
                   and isinstance(result.get("path"), str)})


def load_bundle_data(bundle: Path) -> dict:
    data = load_json(bundle / "bundle.json")
    if data.get("version") != 1 or not isinstance(data.get("results"), list):
        raise ValueError(f"invalid reader asset bundle: {bundle}")
    return data


def combine_bundles(bundles: list[Path]) -> tuple[dict, dict[str, Path]]:
    """Combine shard bundles without copying their potentially large artifacts."""
    if not bundles:
        raise ValueError("at least one reader asset bundle is required")
    data = [load_bundle_data(bundle) for bundle in bundles]
    force_values = {bool(item.get("force_rebuild")) for item in data}
    authoritative_values = {bool(item.get("authoritative_snapshot")) for item in data}
    migration_values = {bool(item.get("bucket_migration")) for item in data}
    pdf_dataset_values = {bool(item.get("pdf_to_dataset")) for item in data}
    if (len(force_values) > 1 or len(authoritative_values) > 1
            or len(migration_values) > 1 or len(pdf_dataset_values) > 1):
        raise ValueError("reader asset bundles have incompatible publication modes")
    results = []
    roots: dict[str, Path] = {}
    keys = set()
    for bundle, item in zip(bundles, data):
        for result in item["results"]:
            key = result.get("key")
            if key in keys:
                raise ValueError(f"duplicate reader asset result: {key}")
            if not key:
                raise ValueError("reader asset result is missing key")
            keys.add(key)
            results.append(result)
            for field in ("path", "chapter_manifest"):
                path = result.get(field)
                if isinstance(path, str):
                    previous = roots.get(path)
                    if previous is not None and previous != bundle:
                        raise ValueError(f"duplicate reader asset artifact path: {path}")
                    roots[path] = bundle
    active_keys = sorted({key for item in data for key in item.get("active_keys", [])})
    combined = {
        "version": 1,
        "results": results,
        "force_rebuild": force_values.pop(),
        "authoritative_snapshot": authoritative_values.pop(),
        "bucket_migration": migration_values.pop(),
        "pdf_to_dataset": pdf_dataset_values.pop(),
        "active_keys": active_keys,
    }
    return combined, roots


def artifact_files(data: dict, roots: dict[str, Path], bundle: Path | None = None) -> dict[str, tuple[Path, str]]:
    """Return bucket paths and local files for results from multiple bundles."""
    artifacts: dict[str, tuple[Path, str]] = {}
    for result in data.get("results", []):
        if result.get("status") != "ready":
            continue
        if result.get("reader_mode") == "pdf" and result.get("pdf_to_dataset") is True:
            continue
        path = result.get("path")
        root = roots.get(path, bundle)
        if not isinstance(path, str) or root is None:
            continue
        artifact = root / path
        if artifact.is_file():
            artifacts[path] = (root, str(artifact))
        if result.get("page_stream"):
            page_root = root / Path(path).parent / "pages"
            if not page_root.is_dir():
                raise ValueError(f"missing spreadsheet page stream for {result['key']}")
            count = int(result.get("page_count") or 0)
            for number in range(1, count + 1):
                child = page_root / f"page-{number:06d}.webp"
                if not child.is_file():
                    raise ValueError(f"spreadsheet page {number} is missing for {result['key']}")
                remote = (Path(path).parent / "pages" / child.name).as_posix()
                artifacts[remote] = (root, str(child))
        if result.get("chapter_manifest"):
            chapter_path = result["chapter_manifest"]
            local_manifest = local_chapter_path(chapter_path)
            local_prefix = Path(local_manifest).parent
            remote_prefix = Path(bucket_chapter_path(chapter_path)).parent
            chapter_root = roots.get(chapter_path, root) / local_prefix
            if not chapter_root.is_dir():
                raise ValueError(f"missing EPUB chapter bundle for {result['key']}")
            for child in sorted(chapter_root.rglob("*")):
                if child.is_file():
                    relative = (remote_prefix / child.relative_to(chapter_root)).as_posix()
                    validate_storage_path(relative)
                    local_relative = (local_prefix / child.relative_to(chapter_root)).as_posix()
                    artifacts[relative] = (root, str(root / local_relative))
    return artifacts


def sync_artifacts(artifacts: dict[str, tuple[Path, str]], token: str, bucket: str,
                   max_attempts: int = 8) -> None:
    """Upload files grouped by their original bundle root."""
    if s3_upload_enabled(bucket):
        s3_upload_artifacts(artifacts, bucket, max_attempts)
        return
    grouped: dict[Path, list[str]] = {}
    for remote_path, (root, _local_path) in artifacts.items():
        grouped.setdefault(root, []).append(remote_path)
    for root, paths in grouped.items():
        _sync_bucket_with_retry(str(root), token, sorted(paths), bucket, max_attempts)


def s3_upload_enabled(bucket: str) -> bool:
    """Use object-level S3 uploads when credentials are available."""
    return bucket in {READER_ASSETS_BUCKET, READER_STAGING_BUCKET} and bool(
        os.environ.get("HF_S3_ACCESS_KEY_ID") and os.environ.get("HF_S3_SECRET_ACCESS_KEY")
    )


def _s3_location(bucket: str) -> tuple[str, str]:
    return bucket.rsplit("/", 1) if "/" in bucket else (
        os.environ.get("HF_S3_NAMESPACE", "vomebook"), bucket
    )


def _s3_client(bucket: str):
    import boto3
    from botocore.config import Config

    namespace, _bucket_name = _s3_location(bucket)
    return boto3.client(
        "s3",
        endpoint_url=f"https://s3.hf.co/{namespace}",
        aws_access_key_id=os.environ["HF_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["HF_S3_SECRET_ACCESS_KEY"],
        config=Config(
            region_name="us-east-1",
            s3={"addressing_style": "path"},
            retries={"mode": "adaptive", "max_attempts": 8},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def s3_upload_artifacts(artifacts: dict[str, tuple[Path, str]], bucket: str,
                        max_attempts: int = 8) -> None:
    """Upload exactly the known artifact paths without listing the bucket."""
    _namespace, bucket_name = _s3_location(bucket)
    client = _s3_client(bucket)
    try:
        workers = max(1, int(os.environ.get("HF_S3_UPLOAD_WORKERS", "8")))
    except ValueError as error:
        raise RuntimeError("HF_S3_UPLOAD_WORKERS must be a positive integer") from error

    def upload(item: tuple[str, tuple[Path, str]]) -> None:
        remote_path, (_root, local_path) = item
        content_type = mimetypes.guess_type(remote_path)[0] or "application/octet-stream"
        expected_bytes = os.path.getsize(local_path)
        for attempt in range(max_attempts):
            try:
                client.upload_file(
                    local_path, bucket_name, remote_path,
                    ExtraArgs={"ContentType": content_type},
                )
                uploaded = client.head_object(Bucket=bucket_name, Key=remote_path)
                if uploaded.get("ContentLength") != expected_bytes:
                    raise RuntimeError(f"uploaded Reader object size mismatch: {remote_path}")
                return
            except Exception as error:
                response = getattr(error, "response", {})
                status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                code = response.get("Error", {}).get("Code")
                retryable = status in {408, 429, 500, 502, 503, 504} or code in {"SlowDown", "RequestTimeout"}
                if not retryable or attempt + 1 == max_attempts:
                    raise
                delay = shared.hf_retry_delay(attempt, cap=120) + random.uniform(0, 2)
                print(f"transient S3 Reader upload error ({code or status}); retrying in {delay:.1f}s")
                time.sleep(delay)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, max(1, len(artifacts)))) as executor:
        list(executor.map(upload, sorted(artifacts.items())))


def s3_upload_tree(root: Path, bucket: str, prefix: str = "reader-index") -> None:
    """Upload a small generated tree by its local paths, without remote listing."""
    root = Path(root)
    artifacts = {}
    local_root = root / prefix
    for path in sorted(local_root.rglob("*")):
        if path.is_file():
            artifacts[path.relative_to(root).as_posix()] = (root, str(path))
    if artifacts:
        s3_upload_artifacts(artifacts, bucket)


def materialize_bundles(bundles: list[Path], data: dict, output: Path) -> Path:
    """Build one upload tree using hardlinks instead of copying large objects."""
    output.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        for source in sorted(bundle.rglob("*")):
            if not source.is_file() or source.name == "bundle.json":
                continue
            relative = source.relative_to(bundle)
            destination = output / relative
            if destination.exists():
                raise ValueError(f"duplicate reader asset artifact path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
    (output / "bundle.json").write_bytes(canonical_json(data, pretty=True))
    return output


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


def build_publish(api: HfApi, repo_id: str, bundle: Path, revision: str | None = None,
                  range_manifest: dict | None = None, *, data_override: dict | None = None,
                  artifact_roots: dict[str, Path] | None = None):
    data = data_override if data_override is not None else load_bundle_data(bundle)
    if data.get("version") != 1 or not isinstance(data.get("results"), list):
        raise ValueError("invalid reader asset bundle")
    artifact_roots = artifact_roots or {}
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
            if not data.get("force_rebuild") and not data.get("bucket_migration"):
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
                result_root = artifact_roots.get(result["path"], bundle)
                artifact = result_root / result["path"]
                if not artifact.is_file() or artifact.stat().st_size != result["bytes"]:
                    raise ValueError(f"missing or invalid artifact for {result['key']}")
                if file_sha256(artifact) != result["sha256"]:
                    raise ValueError(f"artifact digest mismatch for {result['key']}")
                artifacts[result["path"]] = str(artifact)
                if result.get("chapter_manifest"):
                    local_manifest = local_chapter_path(result["chapter_manifest"])
                    local_prefix = Path(local_manifest).parent
                    remote_prefix = Path(bucket_chapter_path(result["chapter_manifest"])).parent
                    root = artifact_roots.get(result["chapter_manifest"], bundle) / local_prefix
                    if not root.is_dir():
                        raise ValueError(f"missing EPUB chapter bundle for {result['key']}")
                    for child in sorted(root.rglob("*")):
                        if child.is_file():
                            path = (remote_prefix / child.relative_to(root)).as_posix()
                            validate_storage_path(path)
                            artifacts[path] = str(child)
            if result.get("reader_mode") in BUCKET_READER_MODES:
                entry["bucket"] = READER_ASSETS_BUCKET
            if (result.get("reader_mode") in BUCKET_STAGING_MODES
                    and result.get("pdf_to_dataset") is not True
                    and not result.get("page_stream")):
                entry["bucket"] = READER_STAGING_BUCKET
                entry["bucket_staging"] = True
            if result.get("page_stream"):
                entry["bucket"] = READER_ASSETS_BUCKET
            if result.get("chapter_manifest"):
                if data.get("bucket_migration"):
                    entry["chapter_manifest"] = bucket_chapter_path(result["chapter_manifest"])
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
    data = load_bundle_data(bundle)
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
            if bucket_token and type(api) is HfApi:
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
                    result_with_paths["bucket_staging"] = (
                        result.get("reader_mode") in BUCKET_STAGING_MODES
                        and not result.get("page_stream")
                    )
                    result_with_paths["bucket_paths"] = bucket_paths({"results": [result]}, bundle)
                    lifecycle_updates.append(staging_record(result_with_paths))
                lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
                lifecycle_path.write_text(json.dumps(merge_lifecycle(lifecycle, lifecycle_updates),
                                                     ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                          encoding="utf-8")
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


def publish_dataset_pdf_artifacts(api: HfApi, repo_id: str, data: dict,
                                  bundle: Path, artifact_roots: dict[str, Path] | None = None,
                                  max_attempts: int = 20) -> None:
    """Publish generated PDFs to the Dataset while indexing them in the bucket."""
    if not any(result.get("pdf_to_dataset") for result in data.get("results", [])):
        return
    artifact_roots = artifact_roots or {}
    artifacts = {}
    for result in data.get("results", []):
        if (result.get("status") != "ready" or result.get("reader_mode") != "pdf"
                or result.get("pdf_to_dataset") is not True):
            continue
        path = result.get("path")
        root = artifact_roots.get(path, bundle)
        if not isinstance(path, str) or not (root / path).is_file():
            raise ValueError(f"missing Dataset PDF artifact for {result.get('key')}")
        artifacts[path] = root / path
    if not artifacts:
        return
    for attempt in range(max_attempts):
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local))
                      for path, local in sorted(artifacts.items())]
        try:
            api.create_commit(
                repo_id=repo_id, repo_type="dataset", operations=operations,
                commit_message="Publish Reader PDF assets", parent_commit=info.sha,
            )
            return
        except HfHubHTTPError as exc:
            status = shared.hf_status_code(exc)
            if status not in {409, 412, 429, 500, 502, 503, 504} or attempt + 1 == max_attempts:
                raise
            time.sleep(shared.hf_retry_delay(attempt) + random.uniform(0, 2))
    raise RuntimeError("Dataset PDF publication retry limit reached")


def publish_bucket_bundle(api: HfApi, repo_id: str, bundle: Path, data: dict,
                          result_keys: set[str], max_attempts: int,
                          artifact_roots: dict[str, Path] | None = None) -> tuple[dict, int]:
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
            manifest, operations = build_publish(
                api, repo_id, bundle, None, range_manifest,
                data_override=data, artifact_roots=artifact_roots,
            )
            publish_dataset_pdf_artifacts(api, repo_id, data, bundle, artifact_roots, max_attempts)
            artifacts = artifact_files(data, artifact_roots or {}, bundle)
            if artifacts:
                staging = set(staging_paths(data))
                regular = {path: value for path, value in artifacts.items() if path not in staging}
                staged = {path: value for path, value in artifacts.items() if path in staging}
                if regular:
                    sync_artifacts(regular, token, READER_ASSETS_BUCKET)
                if staged:
                    sync_artifacts(staged, token, READER_STAGING_BUCKET)
            lifecycle = {"version": 1, "files": {}, "orphans": {}}
            try:
                lifecycle = read_bucket_json(INDEX_FILES["lifecycle"], token)
            except (FileNotFoundError, OSError, ValueError):
                pass
            updates = []
            for result in data.get("results", []):
                if result.get("status") != "ready":
                    continue
                if result.get("reader_mode") == "pdf" and result.get("pdf_to_dataset") is True:
                    continue
                if not (result.get("reader_mode") in BUCKET_READER_MODES | BUCKET_STAGING_MODES
                        or result.get("chapter_manifest")):
                    continue
                item = dict(result)
                item["bucket_staging"] = (
                    result.get("reader_mode") in BUCKET_STAGING_MODES
                    and not result.get("page_stream")
                )
                item["bucket_paths"] = bucket_paths(
                    {"results": [result]}, (artifact_roots or {}).get(result.get("path"), bundle))
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
                if s3_upload_enabled(READER_ASSETS_BUCKET):
                    s3_upload_tree(root, READER_ASSETS_BUCKET)
                else:
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


def publish_bundles(api: HfApi, repo_id: str, bundles: list[Path], *, max_attempts: int = 20) -> tuple[dict, int]:
    """Publish all bundles in one shard with one manifest/index update."""
    data, _artifact_roots = combine_bundles(bundles)
    result_keys = {result["key"] for result in data["results"]}
    if type(api) is HfApi:
        with tempfile.TemporaryDirectory(prefix="reader-shard-") as temporary:
            merged = materialize_bundles(bundles, data, Path(temporary) / "bundle")
            return publish_bucket_bundle(api, repo_id, merged, data, result_keys, max_attempts)
    raise RuntimeError("batch Reader publication requires the bucket API")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("output/reader-assets/bundle"))
    parser.add_argument("--bundles", type=Path, nargs="+",
                        help="Publish multiple conversion bundles as one shard")
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.dry_run:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token or None)
    bundles = args.bundles or [args.bundle]
    if args.dry_run:
        if len(bundles) != 1:
            raise RuntimeError("--dry-run accepts only one bundle")
        manifest, operations = build_publish(api, args.assets_repo, args.bundle)
        print(f"dry run: validated {len(operations) - 2} artifact(s), {len(manifest['files'])} manifest entries")
        return 0
    if args.bundles:
        _, artifact_count = publish_bundles(api, args.assets_repo, bundles)
    else:
        _, artifact_count = publish_bundle(api, args.assets_repo, args.bundle)
    print(f"published {artifact_count} artifact(s) to {args.assets_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
