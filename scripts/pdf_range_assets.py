#!/usr/bin/env python3
"""Incrementally assess original and generated PDFs; atomically publish winners."""
import argparse
import concurrent.futures
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem, sync_bucket
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import pdf_range, pdf_range_state, reader_assets, shared
    from .build_reader_assets_index import build_index, encode_index
    from .publish_reader_assets import remote_manifest, remote_pdf_manifest, remote_pdf_ocr_manifest
    from .reader_bucket import update_lifecycle_consumer
except ImportError:
    import pdf_range, pdf_range_state, reader_assets, shared
    from build_reader_assets_index import build_index, encode_index
    from publish_reader_assets import remote_manifest, remote_pdf_manifest, remote_pdf_ocr_manifest
    from reader_bucket import update_lifecycle_consumer


def discover(records, revisions, manifest, pdf_manifest, api, state, assets_repo, assets_revision, exact=False):
    """Fingerprint files at fixed revisions; repo commits only invalidate metadata."""
    bases = manifest.get("files", {})
    image_keys = {k for k, v in build_index(manifest, pdf_manifest)["f"].items() if v.get("b")}
    grouped, items = {}, {}
    for record in records:
        repo = str(record.get("Repo") or "")
        path = reader_assets.relative_path(record)
        key = reader_assets.asset_key(repo, path)
        if key in image_keys:
            continue
        base = bases.get(key, {})
        if base.get("status") == "ready":
            if base.get("reader_mode") == "pdf" and base.get("sha256") and base.get("path"):
                items[key] = {"key": key, "repo": repo, "source_path": path, "source_kind": "generated",
                              "input_repo": assets_repo, "input_path": base["path"],
                              "input_revision": assets_revision, "input_token": "sha256:" + base["sha256"],
                              "input_sha256": base["sha256"], "input_profile": base.get("profile", ""),
                              "source_bytes": base["bytes"], "source_revision": base.get("source_revision", "")}
            continue
        if str(record.get("Extension", "")).lower().lstrip(".") != "pdf" or not revisions.get(repo):
            continue
        # Inputs with required repairs/decryption must first finish that conversion.
        if reader_assets.source_conversion_contract(repo, path, "pdf") is not None:
            continue
        grouped.setdefault(repo, {})[path] = key
    inventories = dict(state.get("inventories", {}))
    for repo, paths in sorted(grouped.items()):
        revision = revisions[repo]
        cached = inventories.get(repo, {})
        if cached.get("revision") == revision and all(p in cached.get("files", {}) for p in paths):
            metadata = cached["files"]
        else:
            metadata = {}
            # One paginated tree walk per changed repo also obtains LFS fingerprints.
            listing = (api.get_paths_info(repo_id=repo, paths=list(paths), repo_type="dataset", revision=revision)
                       if exact else api.list_repo_tree(repo_id=repo, repo_type="dataset", revision=revision, recursive=True))
            for file in listing:
                path = getattr(file, "path", "")
                if path not in paths or not hasattr(file, "blob_id"):
                    continue
                lfs = getattr(file, "lfs", None)
                sha = getattr(lfs, "sha256", None) if lfs else None
                metadata[path] = {"token": "sha256:" + sha if sha else "git:" + file.blob_id,
                                  "sha256": sha or "", "bytes": file.size}
            missing = set(paths) - set(metadata)
            if missing:
                raise RuntimeError(f"source snapshot missing {len(missing)} file(s) in {repo}")
            inventories[repo] = {"revision": revision, "files": metadata}
        for path, key in paths.items():
            meta = metadata[path]
            items[key] = {"key": key, "repo": repo, "source_path": path, "source_kind": "upstream",
                          "input_repo": repo, "input_path": path, "input_revision": revision,
                          "source_revision": revision, "input_token": meta["token"],
                          "input_sha256": meta["sha256"], "input_profile": "upstream",
                          "source_bytes": meta["bytes"]}
    # Generated outputs whose source temporarily disappeared must not stay active.
    return items, inventories


def identity(item, tool_version, assessment=None):
    value = [item["input_token"], item["input_profile"], pdf_range.PROFILE,
             assessment or pdf_range.ASSESSMENT, tool_version]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def plan(items, state, tool_version, limit, exact_key="", retry_failed=False, retry_blocked=False,
         retry_stale_blocked=False, retry_reason=""):
    files, pending = {}, []
    reusable = {entry.get("identity"): entry for entry in state.get("files", {}).values()
                if entry.get("status") in {"optimized", "unchanged", "no-gain", "unsupported"}}
    for key, item in sorted(items.items()):
        fingerprint = identity(item, tool_version)
        previous = state.get("files", {}).get(key, {})
        # v4 admits additional passive structures without changing the criteria
        # for already completed v3 measurements. Only blocked inputs need retry.
        if (pdf_range.ASSESSMENT == "pdfjs-6.3.289-range-1m-scene-v2-policy-v4"
                and previous.get("status") in {"optimized", "unchanged", "no-gain"}
                and previous.get("identity") == identity(item, tool_version,
                    "pdfjs-6.3.289-range-1m-scene-v2-policy-v3")):
            files[key] = previous
            continue
        reason_matches = not retry_reason or retry_reason in previous.get("reason", "")
        if retry_reason and previous.get("status") in {"failed", "unsupported"} and not reason_matches:
            files[key] = {**previous, **item}
            continue
        if ((retry_blocked or retry_stale_blocked) and reason_matches
                and previous.get("status") not in {"failed", "unsupported"}):
            files[key] = previous or {**item, "identity": fingerprint, "status": "pending"}
            continue
        if retry_stale_blocked and previous.get("identity") == fingerprint:
            files[key] = {**previous, **item}
            continue
        retry_requested = ((retry_blocked or retry_stale_blocked) and reason_matches) or (
            retry_failed and previous.get("status") == "failed")
        if (previous.get("input_token") == item["input_token"]
                and previous.get("input_path") == item.get("input_path")
                and previous.get("input_repo") == item.get("input_repo")):
            item = {**item, **{field: previous[field] for field in ("input_revision", "source_revision")
                              if field in previous}}
        size = item.get("source_bytes", pdf_range.MIN_BYTES)
        if size < pdf_range.MIN_BYTES or size > 2 * 1024 * pdf_range.MI:
            # Metadata-only decisions must not consume expensive assessment slots.
            files[key] = {**item, "identity": fingerprint,
                          "status": "unchanged" if size < pdf_range.MIN_BYTES else "unsupported",
                          "reason": "below-4-mib" if size < pdf_range.MIN_BYTES else "source-exceeds-2-gib"}
            continue
        if previous.get("identity") == fingerprint:
            current = {**previous, **item}
            if previous.get("status") != "pending" and not retry_requested:
                files[key] = current
                continue
        elif fingerprint in reusable and not retry_requested:
            cached = reusable[fingerprint]
            files[key] = {**cached, **item, "input_sha256": cached.get("input_sha256", item.get("input_sha256", ""))}
            continue
        else:
            current = {**item, "identity": fingerprint, "status": "pending"}
        if (previous.get("status") == "optimized" and previous.get("input_token") == item["input_token"]
                and previous.get("input_profile") == item["input_profile"]):
            # Tool upgrades queue a new assessment while keeping a valid readable
            # output of the same content. Input changes always remove the old route.
            files[key] = {**previous, **item}
        else:
            files[key] = current
        if not exact_key or key == exact_key:
            pending.append({**current, "_previous_identity": previous.get("identity")})
    # Interleave repositories and generated/original inputs. Lexicographic source
    # order otherwise postpones generated assets behind tens of thousands of PDFs.
    group = lambda row: (row.get("source_kind", "upstream"), row.get("repo", ""))
    completed = Counter(group(row) for row in files.values()
                        if row.get("status") not in {"pending"}
                        and row.get("reason") not in {"below-4-mib", "source-exceeds-2-gib"})
    selected = []
    for retry in (False, True):
        groups = defaultdict(deque)
        lane_load = Counter()
        for row in sorted(pending, key=lambda row: (
                state.get("files", {}).get(row["key"], {}).get("status") not in {"failed", "unsupported"}, row["key"])):
            if (row.get("status") == "failed") == retry:
                # A policy upgrade must not make the entire completed corpus
                # jump ahead of files that have never been assessed.
                previous_status = state.get("files", {}).get(row["key"], {}).get("status", "pending")
                lane = 0 if previous_status == "pending" else 1
                groups[(lane, *group(row))].append(row)
        while groups and len(selected) < limit:
            bucket = min(groups, key=lambda key: (lane_load[key[0]], key[0], completed[key[1:]], key))
            selected.append(groups[bucket].popleft())
            completed[bucket[1:]] += 1
            lane_load[bucket[0]] += 1
            if not groups[bucket]:
                del groups[bucket]
    return files, selected


def compact_report(report):
    def measurement(value):
        return {k: v for k, v in value.items() if k != "renders"}
    result = {k: v for k, v in report.items() if k not in {"before", "candidates"}}
    if "before" in report:
        result["before"] = measurement(report["before"])
    if "candidates" in report:
        result["candidates"] = {key: {**value, **({"measurement": measurement(value["measurement"])}
                                                     if "measurement" in value else {})}
                                for key, value in report["candidates"].items()}
    return result


def verify_bucket_object(fs: HfFileSystem, path: str, expected_sha: str, expected_bytes: int) -> None:
    digest = hashlib.sha256()
    size = 0
    with fs.open(f"hf://buckets/{shared.PDF_RANGE_BUCKET}/{path}", "rb") as stream:
        while chunk := stream.read(shared.CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != expected_sha or size != expected_bytes:
        raise ValueError("reused optimized artifact digest mismatch")


def remote_snapshot(api, repo):
    """Read the shared baseline, retrying transient HF throttling per Runner."""
    for attempt in range(6):
        try:
            revision = api.repo_info(repo_id=repo, repo_type="dataset").sha
            state = pdf_range_state.remote_state(api, repo, revision)
            manifest = remote_manifest(api, repo, revision)
            pdf_manifest = remote_pdf_manifest(api, repo, revision)
            return revision, state, manifest, pdf_manifest
        except HfHubHTTPError as error:
            if not shared.is_retryable_hf_status(shared.hf_status_code(error)) or attempt == 5:
                raise
            delay = shared.hf_retry_delay(attempt) + random.uniform(0, 2)
            print(f"HF metadata retry after {shared.hf_status_code(error)}; sleeping {delay:.1f}s", flush=True)
            time.sleep(delay)


def planned_results(files, baseline, assessment_keys, shard_count=1, shard_index=0):
    """Publish terminal metadata decisions and reused assessments exactly once."""
    results = []
    for key, row in sorted(files.items()):
        previous = baseline.get("files", {}).get(key, {})
        if key in assessment_keys or row.get("status") == "pending":
            continue
        if (previous.get("status"), previous.get("identity")) == (row.get("status"), row.get("identity")):
            continue
        results.append({**row, "_previous_identity": previous.get("identity")})
    return results[shard_index::shard_count]


def process(item, bundle, vendor, api):
    # A retry starts with input identity only. Otherwise an unchanged/unsupported
    # result can accidentally retain candidates or output paths from its failure.
    input_fields = ("key", "repo", "source_path", "source_kind", "input_repo", "input_path",
                    "input_revision", "input_token", "input_sha256", "input_profile",
                    "source_bytes", "source_revision", "identity", "_previous_identity")
    result = {field: item[field] for field in input_fields if field in item}
    if item["source_bytes"] < pdf_range.MIN_BYTES:
        return {**result, "status": "unchanged", "reason": "below-4-mib"}
    if item["source_bytes"] > 2 * 1024 * pdf_range.MI:
        return {**result, "status": "unsupported", "reason": "source-exceeds-2-gib"}
    download_cache = tempfile.TemporaryDirectory(dir=bundle)
    try:
        source = Path(api.hf_hub_download(repo_id=item["input_repo"], repo_type="dataset",
                                         filename=item["input_path"], revision=item["input_revision"],
                                         cache_dir=download_cache.name))
        digest, size = shared.hash_file(source)
        if item.get("input_sha256") and digest != item["input_sha256"]:
            raise ValueError("input SHA-256 mismatch")
        if item["input_token"].startswith("git:"):
            hasher = hashlib.sha1(f"blob {size}\0".encode())
            with source.open("rb") as stream:
                while chunk := stream.read(shared.CHUNK_BYTES):
                    hasher.update(chunk)
            if hasher.hexdigest() != item["input_token"][4:]:
                raise ValueError("input Git blob mismatch")
        result.update(input_sha256=digest, source_bytes=size)
        with tempfile.TemporaryDirectory(dir=bundle) as temporary:
            work = Path(temporary)
            report, chosen = assess_isolated(source, work, vendor)
            result.update(compact_report(report))
            if chosen:
                output_sha, output_bytes = shared.hash_file(chosen)
                profile = f"{pdf_range.PROFILE}-{item['identity'][:16]}-{report['method']}"
                output_path = f"objects/{digest[:2]}/{digest}/{profile}/document.pdf"
                destination = bundle / output_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(chosen, destination)
                result.update(path=output_path, sha256=output_sha, bytes=output_bytes)
    except pdf_range.UnsupportedPDF as error:
        result.update(status="unsupported", reason=str(error)[:400])
    except Exception as error:
        details = pdf_range.failure_details(error)
        result.update(status="failed", reason=details["error"], error_category=details["error_category"])
    finally:
        download_cache.cleanup()
    return result


def assess_isolated(source, work, vendor, timeout=900):
    """Bound the entire assessment, including parser loops and browser teardown."""
    command = [sys.executable, str(Path(__file__).with_name("pdf_range_worker.py")),
               "--source", str(source.resolve()), "--work", str(work.resolve()), "--vendor", str(vendor.resolve())]
    with (work / "assessment.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"status": "failed", "reason": "assessment-total-timeout",
                    "error_category": "timeout", "timeout_seconds": timeout}, None
    report_path = work / "assessment.json"
    if process.returncode or not report_path.is_file():
        with (work / "assessment.log").open("rb") as log:
            log.seek(max(0, log.seek(0, 2) - 1600))
            detail = log.read().decode("utf-8", "replace")
        raise RuntimeError(f"PDF assessment worker exit {process.returncode}: {detail}")
    report = json.loads(report_path.read_text())
    chosen = work / (report["method"] + ".pdf") if report["status"] == "optimized" else None
    if chosen is not None and (chosen.parent != work or not chosen.is_file()):
        raise ValueError("assessment worker returned an invalid artifact")
    return report, chosen


def publish(api, repo, baseline, state, bundle, results):
    """Rebuild all routes against the commit parent; keep unrelated publisher work."""
    artifacts, reused = {}, {}
    artifact_fields = ("identity", "path", "sha256", "bytes")
    accepted_artifacts = {tuple(row.get(key) for key in artifact_fields)
                          for row in baseline.get("files", {}).values() if row.get("status") == "optimized"}
    for result in results:
        if result.get("status") == "optimized":
            path = bundle / result["path"]
            if not path.is_file():
                # Reuse only an artifact already accepted in this exact baseline.
                signature = tuple(result.get(key) for key in artifact_fields)
                if signature not in accepted_artifacts:
                    raise ValueError("optimized artifact missing from bundle and baseline")
                reused[result["path"]] = result
                continue
            if shared.hash_file(path) != (result["sha256"], result["bytes"]):
                raise ValueError("optimized artifact digest mismatch")
            # The shard publisher uploads structure-optimized PDFs to the
            # dedicated Bucket before this metadata commit. Keep Reader-Assets
            # free of these large immutable delivery objects.
    for attempt in range(6):
        revision = api.repo_info(repo_id=repo, repo_type="dataset").sha
        current = pdf_range_state.remote_state(api, repo, revision)
        if current == state:
            return revision
        if current != baseline:
            raise RuntimeError("PDF range state changed concurrently; rerun from current state")
        if reused:
            if state.get("artifact_bucket") == shared.PDF_RANGE_BUCKET:
                fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
                for path, result in reused.items():
                    verify_bucket_object(fs, path, result["sha256"], result["bytes"])
            else:
                remote = {file.path: file for file in api.get_paths_info(
                    repo_id=repo, paths=list(reused), repo_type="dataset", revision=revision)}
                for path, result in reused.items():
                    file = remote.get(path)
                    if (not file or file.size != result["bytes"] or
                            getattr(getattr(file, "lfs", None), "sha256", None) != result["sha256"]):
                        raise ValueError("reused optimized artifact digest mismatch")
        base = remote_manifest(api, repo, revision)
        images = remote_pdf_manifest(api, repo, revision)
        ocr = remote_pdf_ocr_manifest(api, repo, revision)
        operations = [*artifacts.values(),
                      CommitOperationAdd(path_in_repo=pdf_range_state.MANIFEST_NAME,
                                         path_or_fileobj=reader_assets.canonical_json(state, pretty=True)),
                      CommitOperationAdd(path_in_repo="reader_assets.json.gz",
                                         path_or_fileobj=encode_index(base, images, state, ocr))]
        try:
            commit = api.create_commit(repo_id=repo, repo_type="dataset", operations=operations,
                                       commit_message="Optimize PDF layouts after range verification", parent_commit=revision)
            if type(api) is HfApi and os.environ.get("HF_TOKEN"):
                with tempfile.TemporaryDirectory(prefix="pdf-range-index-") as root:
                    index_root = Path(root) / "reader-index"
                    index_root.mkdir(parents=True, exist_ok=True)
                    (index_root / "pdf_range_manifest.json").write_text(
                        reader_assets.canonical_json(state, pretty=True).decode("utf-8"), encoding="utf-8")
                    (index_root / "reader_assets.json.gz").write_bytes(encode_index(base, images, state, ocr))
                    sync_bucket(root, f"hf://buckets/{shared.PDF_PAGES_BUCKET}",
                                include=["reader-index/**"], token=os.environ["HF_TOKEN"], quiet=False)
                for result in results:
                    if result.get("status") in {"optimized", "unchanged", "unsupported"}:
                        update_lifecycle_consumer(result["key"], "range", "done", os.environ["HF_TOKEN"])
                    elif result.get("status") == "failed":
                        update_lifecycle_consumer(result["key"], "range", "failed", os.environ["HF_TOKEN"])
            return commit.oid
        except HfHubHTTPError as error:
            if not shared.is_retryable_hf_status(shared.hf_status_code(error)) or attempt == 5:
                raise
            time.sleep(shared.hf_retry_delay(attempt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--assets-repo", default=reader_assets.READER_ASSETS_REPO)
    parser.add_argument("--bundle", type=Path, default=Path("output/pdf-range"))
    parser.add_argument("--vendor", type=Path, default=Path(os.environ.get("PDF_RANGE_VENDOR", "node_modules/pdfjs-dist")))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--repo", default="")
    parser.add_argument("--path", default="")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-blocked", action="store_true",
                        help="Assess only previously failed/unsupported PDFs, including unchanged inputs")
    parser.add_argument("--retry-stale-blocked", action="store_true",
                        help="Assess only failed/unsupported PDFs whose input or validation identity changed")
    parser.add_argument("--retry-reason", default="",
                        help="Assess only records whose previous reason contains this text")
    parser.add_argument("--tool-version", default="",
                        help="Assessment tool version for planning; defaults to qpdf --version")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean-published", action="store_true", help="Remove this bundle's uploaded objects after successful publication")
    args = parser.parse_args()
    if not 1 <= args.limit <= 2000 or not 1 <= args.workers <= 4:
        parser.error("limit must be 1..2000 and workers 1..4")
    if not 1 <= args.shard_count <= 16 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-count must be 1..16 and shard-index must be in range")
    if bool(args.repo) != bool(args.path):
        parser.error("repo and path must be provided together")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revision, baseline, base, images = remote_snapshot(api, args.assets_repo)
    records = reader_assets.decode_search_payload(json.loads(args.search_data.read_text()))
    revisions = json.loads(args.revisions.read_text())
    if args.repo:
        records = [row for row in records if row.get("Repo") == args.repo
                   and reader_assets.relative_path(row) == args.path]
        if not records:
            parser.error("requested source is absent from this inventory")
    items, inventories = discover(records, revisions, base, images, api, baseline, args.assets_repo, revision, bool(args.repo))
    version = args.tool_version or subprocess.check_output(["qpdf", "--version"], text=True).splitlines()[0]
    key = reader_assets.asset_key(args.repo, args.path) if args.repo else ""
    files, pending = plan(items, baseline, version, args.limit * args.shard_count, key,
                          args.retry_failed, args.retry_blocked, args.retry_stale_blocked,
                          args.retry_reason)
    results = planned_results(files, baseline, {row["key"] for row in pending}, args.shard_count, args.shard_index)
    pending = pending[args.shard_index::args.shard_count]
    if args.repo:
        files = {**baseline.get("files", {}), **files}
    args.bundle.mkdir(parents=True, exist_ok=True)
    print(f"PDF layout inventory {len(items)}; batch {len(pending)}; planned results {len(results)}", flush=True)
    if args.dry_run:
        (args.bundle / "plan.json").write_bytes(reader_assets.canonical_json(pending, pretty=True))
        return
    (args.bundle / "results.json").write_bytes(reader_assets.canonical_json(results, pretty=True))
    workers = 1 if any(item["source_bytes"] > 256 * pdf_range.MI for item in pending) else args.workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        grouped = {}
        for item in pending:
            grouped.setdefault(item["identity"], []).append(item)
        futures = {pool.submit(process, aliases[0], args.bundle, args.vendor, api): aliases
                   for aliases in grouped.values()}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for alias in futures[future]:
                inputs = {field: alias[field] for field in (
                    "key", "repo", "source_path", "source_kind", "input_repo", "input_path",
                    "input_revision", "input_token", "input_profile", "source_revision", "identity",
                    "_previous_identity") if field in alias}
                resolved = {**result, **inputs}
                results.append(resolved)
                files[alias["key"]] = {k: v for k, v in resolved.items() if k != "_previous_identity"}
            print(json.dumps({"file": result["source_path"], "status": result["status"],
                               "reason": result.get("reason"), "method": result.get("method")}, ensure_ascii=False), flush=True)
            checkpoint = args.bundle / "results.json.tmp"
            checkpoint.write_bytes(reader_assets.canonical_json(results, pretty=True))
            checkpoint.replace(args.bundle / "results.json")
    state = {"version": 1, "artifact_bucket": shared.PDF_RANGE_BUCKET,
             "files": files, "inventories": inventories}
    (args.bundle / "results.json").write_bytes(reader_assets.canonical_json(results, pretty=True))
    (args.bundle / pdf_range_state.MANIFEST_NAME).write_bytes(reader_assets.canonical_json(state, pretty=True))
    if not args.build_only and state != baseline:
        print("Published revision " + publish(api, args.assets_repo, baseline, state, args.bundle, results), flush=True)
        if args.clean_published and (args.bundle / "objects").is_dir():
            shutil.rmtree(args.bundle / "objects")


if __name__ == "__main__":
    main()
