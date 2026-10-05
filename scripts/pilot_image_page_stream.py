#!/usr/bin/env python3
"""Build and publish a small non-PDF image page-stream pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import HfFileSystem, batch_bucket_files
from PIL import Image, ImageFile, ImageSequence

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from .reader_assets import decode_search_payload, relative_path, source_url
except ImportError:
    from reader_assets import decode_search_payload, relative_path, source_url


TARGET_BUCKET = "vomebook/reader-assets-v2"
IMAGE_ROOT = "pages/image"
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "bmp", "tif", "tiff", "webp", "png"}
MAX_WEBP_EDGE = 16_383


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-data", type=Path, default=Path("output/search_data.json"))
    parser.add_argument("--revisions", type=Path, default=Path("state/commits.json"))
    parser.add_argument("--limit", type=int, default=0,
                        help="maximum sources; 0 means all")
    parser.add_argument("--max-source-bytes", type=int, default=0,
                        help="maximum source bytes; 0 means unlimited")
    parser.add_argument("--extension", default="all")
    parser.add_argument("--bucket", default=TARGET_BUCKET)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def download(url: str, target: Path, token: str | None) -> None:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def source_records(path: Path, revisions: dict, max_bytes: int, extension_filter: str) -> list[dict]:
    records = decode_search_payload(json.loads(path.read_text(encoding="utf-8")))
    selected = []
    for record in records:
        extension = str(record.get("Extension") or "").lower().lstrip(".")
        size = int(record.get("Size") or 0)
        if (extension not in SUPPORTED_EXTENSIONS
                or (extension_filter != "all" and extension != extension_filter)
                or size <= 0 or (max_bytes and size > max_bytes)):
            continue
        repo = str(record.get("Repo") or "")
        revision = str(revisions.get(repo) or "")
        if not repo or not revision:
            continue
        selected.append({
            "repo": repo,
            "path": relative_path(record),
            "extension": extension,
            "bytes": size,
            "revision": revision,
        })
    selected.sort(key=lambda item: (item["bytes"], item["repo"], item["path"]))
    return selected


def build_one(item: dict, work: Path, token: str | None, bucket: str) -> tuple[dict, dict[str, str]]:
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source"
    download(source_url(item["repo"], item["revision"], item["path"]), source, token)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    object_id = source_digest
    extension = item["extension"]
    root = f"{IMAGE_ROOT}/{extension}/{object_id}"
    manifest_path = f"{root}/page-manifest.json"
    pages = work / "pages"
    pages.mkdir()
    page_entries = []
    uploads: dict[str, str] = {}
    with Image.open(source) as image:
        if image.width < 1 or image.height < 1:
            raise ValueError("image has invalid dimensions")
        for number, frame in enumerate(ImageSequence.Iterator(image), start=1):
            output = pages / f"page-{number:06d}.webp"
            converted = frame.convert("RGB")
            if max(converted.size) > MAX_WEBP_EDGE:
                scale = MAX_WEBP_EDGE / max(converted.size)
                resized = converted.resize(
                    (max(1, round(converted.width * scale)),
                     max(1, round(converted.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                converted.close()
                converted = resized
            converted.save(output, "WEBP", quality=85, method=6)
            converted.close()
            page_bytes = output.read_bytes()
            page_digest = hashlib.sha256(page_bytes).hexdigest()
            relative = f"pages/page-{number:06d}.webp"
            remote = f"{root}/{relative}"
            page_entries.append({"path": relative, "bytes": len(page_bytes), "sha256": page_digest})
            uploads[remote] = str(output)
    manifest = {
        "kind": "image-page-stream",
        "version": 1,
        "source_extension": extension,
        "source_sha256": source_digest,
        "page_count": len(page_entries),
        "pages": page_entries,
    }
    manifest_file = work / "page-manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    uploads[manifest_path] = str(manifest_file)
    return {
        "key": f"{item['repo']}\0{item['path']}",
        "repo": item["repo"],
        "path": item["path"],
        "extension": extension,
        "source_revision": item["revision"],
        "source_bytes": len(source.read_bytes()),
        "source_sha256": source_digest,
        "mode": "page-stream",
        "bucket": bucket,
        "manifest": manifest_path,
        "page_count": len(page_entries),
    }, uploads


def existing_index(bucket: str, token: str | None, name: str = "index.json") -> dict:
    path = f"hf://buckets/{bucket}/{IMAGE_ROOT}/{name}"
    try:
        fs = HfFileSystem(token=token)
        with fs.open(path, "rb") as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("files"), list):
            return payload
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        pass
    return {"version": 1, "kind": "image-page-stream-index", "files": []}


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.max_source_bytes < 0:
        raise ValueError("limit and max-source-bytes must be non-negative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    token = os.environ.get("HF_TOKEN")
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    extension = args.extension.lower().lstrip(".")
    if extension != "all" and extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported image extension: {extension}")
    selected = source_records(args.search_data, revisions, args.max_source_bytes, extension)
    index_name = "index.json" if args.shard_count == 1 else f"index-{args.shard_index:02d}.json"
    previous = existing_index(args.bucket, token, index_name)
    canonical = existing_index(args.bucket, token)
    completed = {
        entry.get("key"): entry.get("source_revision")
        for entry in previous.get("files", []) if isinstance(entry, dict)
    }
    selected = [item for item in selected
                if int.from_bytes(hashlib.sha256(
                    f"{item['repo']}\0{item['path']}".encode()).digest()[:8], "big") % args.shard_count == args.shard_index
                if completed.get(f"{item['repo']}\0{item['path']}") != item["revision"]]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        print("no matching image sources")
        return 0
    uploads: dict[str, str] = {}
    entries = []
    with tempfile.TemporaryDirectory(prefix="reader-image-pilot-") as directory:
        work_root = Path(directory)
        for index, item in enumerate(selected):
            try:
                result, files = build_one(item, work_root / str(index), token, args.bucket)
            except Exception as error:
                print(f"failed: {item['repo']}/{item['path']}: {type(error).__name__}: {error}")
                continue
            entries.append(result)
            uploads.update(files)
        if not entries:
            print("no image source converted successfully")
            return 1
        previous_files = {entry.get("key"): entry for entry in previous.get("files", []) if isinstance(entry, dict)}
        previous_files.update({entry.get("key"): entry for entry in canonical.get("files", []) if isinstance(entry, dict)})
        previous_files.update({entry["key"]: entry for entry in entries})
        index_payload = {"version": 1, "kind": "image-page-stream-index", "files": [
            previous_files[key] for key in sorted(previous_files)
        ]}
        index_file = work_root / "index.json"
        index_file.write_text(json.dumps(index_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        uploads[f"{IMAGE_ROOT}/{index_name}"] = str(index_file)
        print(f"planned {len(entries)} image stream(s), {len(uploads)} object(s)")
        if args.apply:
            batch_bucket_files(args.bucket, add=[(local, remote) for remote, local in sorted(uploads.items())], token=token)
            print(f"published {len(uploads)} object(s) to {args.bucket}")
        else:
            print("report-only; pass --apply to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
