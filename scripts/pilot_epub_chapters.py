#!/usr/bin/env python3
"""Build and publish incremental EPUB chapter streams to Reader Assets v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem

try:
    from .convert_reader_assets import convert_item
    from .epub_chapters import build_bundle, bundle_version
    from .recover_chm import recover as recover_chm
    from .reader_assets import bucket_conversion_contract, decode_search_payload, relative_path, source_url
    from .shared import batch_bucket_files_with_retry
except ImportError:
    from convert_reader_assets import convert_item
    from epub_chapters import build_bundle, bundle_version
    from recover_chm import recover as recover_chm
    from reader_assets import bucket_conversion_contract, decode_search_payload, relative_path, source_url
    from shared import batch_bucket_files_with_retry


TARGET_BUCKET = "vomebook/reader-assets-v2"
ROOT = "chapters/ebook/epub"
SUPPORTED_EXTENSIONS = {"epub", "mobi", "azw3", "fb2", "chm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-source-bytes", type=int, default=0)
    parser.add_argument("--extension", default="epub")
    parser.add_argument("--bucket", default=TARGET_BUCKET)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def download(url: str, target: Path, token: str | None) -> None:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def existing_index(bucket: str, token: str | None, root: str, name: str = "index.json") -> dict:
    try:
        fs = HfFileSystem(token=token)
        with fs.open(f"hf://buckets/{bucket}/{root}/{name}", "rb") as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("files"), list):
            return payload
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        pass
    return {"version": 1, "kind": "ebook-chapter-stream-index", "files": []}


def shard_for(key: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % count


def select_records(path: Path, revisions: dict, max_bytes: int, extension_filter: str) -> list[dict]:
    records = decode_search_payload(json.loads(path.read_text(encoding="utf-8")))
    selected = []
    for record in records:
        extension = str(record.get("Extension") or "").lower().lstrip(".")
        size = int(record.get("Size") or 0)
        if extension != extension_filter or (max_bytes and size > max_bytes):
            continue
        repo = str(record.get("Repo") or "")
        revision = str(revisions.get(repo) or "")
        if repo and revision:
            selected.append({"repo": repo, "path": relative_path(record), "revision": revision, "source_bytes": size})
    selected.sort(key=lambda item: (item["repo"], item["path"]))
    return selected


def build_one(item: dict, work: Path, token: str | None, bucket: str) -> tuple[dict, dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    extension = item["extension"]
    source = work / f"source.{extension}"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    source_bytes = source.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    native_fallback = None
    if extension == "chm":
        contract = bucket_conversion_contract(item["repo"], item["path"], extension, item["source_bytes"])
        if contract is None:
            raise ValueError("no CHM conversion contract")
        profile, mode, output_name = contract
        converted_root = work / "converted"
        try:
            converted = convert_item({
                "key": f"{item['repo']}\0{item['path']}", "repo": item["repo"], "path": item["path"],
                "extension": extension, "source_revision": item["revision"], "source_url": source_url(
                    item["repo"], item["revision"], item["path"]), "source_bytes": item["source_bytes"],
                "profile": profile, "reader_mode": mode, "output_name": output_name,
            }, converted_root)
            if not converted.get("chapter_manifest") or converted.get("chapter_bundle_error"):
                raise RuntimeError(converted.get("chapter_bundle_error") or "CHM chapter stream was not produced")
            source_chapter_dir = converted_root / Path(converted["chapter_manifest"]).parent
            native_source = converted_root / converted["path"]
            bundle = work / "bundle"
            shutil.copytree(source_chapter_dir, bundle)
            manifest = json.loads((bundle / "chapter-manifest.json").read_text(encoding="utf-8"))
        except Exception as primary_error:
            # Calibre and its CHM navigation repair are not reliable for every
            # legacy container. Recover static pages directly, then feed the
            # repaired EPUB through the same chapter-stream builder.
            recovered = work / "recovered.epub"
            try:
                recover_chm(source, recovered, Path(item["path"]).stem)
                bundle = work / "bundle"
                manifest = build_bundle(recovered, bundle)
                native_source = recovered
                print(f"CHM recovery fallback succeeded: {item['repo']}/{item['path']}")
            except Exception as recovery_error:
                raise RuntimeError(
                    f"CHM conversion failed ({type(primary_error).__name__}: {primary_error}); "
                    f"recovery failed ({type(recovery_error).__name__}: {recovery_error})"
                ) from recovery_error
        native_fallback = f"native/ebook/chm/{source_digest}/document.epub"
    else:
        chapter_source = source
        if extension != "epub":
            chapter_source = work / "chapter-source.epub"
            subprocess.run(["ebook-convert", str(source), str(chapter_source), "--flow-size", "0"],
                           check=True, timeout=600)
        bundle = work / "bundle"
        manifest = build_bundle(chapter_source, bundle)
    if extension == "chm":
        version = bundle_version(bundle)
        root = f"chapters/ebook/{extension}/{source_digest}/{version}"
    else:
        version = bundle_version(bundle)
        root = f"chapters/ebook/{extension}/{source_digest}/{version}"
    uploads = {}
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        uploads[f"{root}/{path.relative_to(bundle).as_posix()}"] = str(path)
    if native_fallback:
        uploads[native_fallback] = str(native_source)
    return {
        "key": f"{item['repo']}\0{item['path']}", "repo": item["repo"], "path": item["path"],
        "source_extension": extension, "source_revision": item["revision"], "source_sha256": source_digest,
        "source_bytes": len(source_bytes), "mode": "chapter-stream", "bucket": bucket,
        "root": root, "manifest": f"{root}/chapter-manifest.json",
        "chapter_count": len(manifest["chapters"]),
        "search_index": f"{root}/epub-search-index.json.gz",
        **({"fallback": native_fallback, "object": native_fallback} if native_fallback else {}),
    }, uploads


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.max_source_bytes < 0:
        raise ValueError("limit and max-source-bytes must be non-negative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    token = os.environ.get("HF_TOKEN")
    extension = args.extension.lower().lstrip(".")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported ebook extension: {extension}")
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    root_prefix = f"chapters/ebook/{extension}"
    index_name = "index.json" if args.shard_count == 1 else f"index-{args.shard_index:02d}.json"
    previous = existing_index(args.bucket, token, root_prefix, index_name)
    completed = {entry.get("key"): entry.get("source_revision") for entry in previous.get("files", []) if isinstance(entry, dict)}
    selected = [item for item in select_records(args.search_data, revisions, args.max_source_bytes, extension)
                if shard_for(f"{item['repo']}\0{item['path']}", args.shard_count) == args.shard_index
                if completed.get(f"{item['repo']}\0{item['path']}") != item["revision"]]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        print("no pending EPUB sources")
        return 0
    uploads, entries, failures = {}, [], []
    with tempfile.TemporaryDirectory(prefix="reader-epub-") as directory:
        root = Path(directory)
        for index, item in enumerate(selected):
            item["extension"] = extension
            try:
                entry, files = build_one(item, root / str(index), token, args.bucket)
            except Exception as error:
                failures.append({
                    "key": f"{item['repo']}\0{item['path']}",
                    "repo": item["repo"], "path": item["path"],
                    "source_revision": item["revision"],
                    "error": f"{type(error).__name__}: {error}",
                })
                print(f"failed: {item['repo']}/{item['path']}: {type(error).__name__}: {error}")
                continue
            entries.append(entry)
            uploads.update(files)
        if not entries and failures:
            print(f"no EPUB source converted; failures={len(failures)}")
            return 0
        merged = {entry.get("key"): entry for entry in previous.get("files", []) if isinstance(entry, dict)}
        merged.update({entry["key"]: entry for entry in entries})
        index_file = root / "index.json"
        index_file.write_text(json.dumps({
            "version": 1, "kind": "ebook-chapter-stream-index",
            "files": [merged[key] for key in sorted(merged)],
            "failures": failures,
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        uploads[f"{root_prefix}/{index_name}"] = str(index_file)
        print(f"planned {len(entries)} EPUB stream(s), {len(uploads)} object(s), failures={len(failures)}")
        if args.apply:
            batch_bucket_files_with_retry(args.bucket, [(local, remote) for remote, local in sorted(uploads.items())], token)
            print(f"published {len(uploads)} object(s) to {args.bucket}")
        else:
            print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
