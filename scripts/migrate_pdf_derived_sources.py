#!/usr/bin/env python3
"""Migrate existing generated PDF inputs into a new PDF page bucket.

This copies already-produced PDF derivatives server-side. It never invokes a
CAJ/KDH converter, PDF renderer, or OCR engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils._hf_uris import parse_hf_uri


PAGE_SOURCE_EXTENSIONS = {"pdf", "djvu", "caj", "kdh"}
DEFAULT_ASSETS_REPO = "vomebook/Reader-Assets"
DEFAULT_BUCKET = "vomebook/pdf-pages-v2"
DEFAULT_GENERATION = "migration-20261007"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default=DEFAULT_ASSETS_REPO)
    parser.add_argument("--assets-revision", default="main")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--generation", default=DEFAULT_GENERATION)
    parser.add_argument("--report", type=Path, default=Path("/tmp/pdf-derived-migration.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def load_entries(api: HfApi, repo: str, revision: str, generation: str, bucket: str) -> list[dict]:
    local = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename="manifest.json", revision=revision)
    manifest = json.loads(Path(local).read_text(encoding="utf-8"))
    entries = []
    for key, entry in manifest.get("files", {}).items():
        if not isinstance(entry, dict) or entry.get("status", "ready") != "ready":
            continue
        extension = str(entry.get("source_extension") or "").lower().lstrip(".")
        source_path = entry.get("path")
        if (entry.get("reader_mode") != "pdf" or extension not in PAGE_SOURCE_EXTENSIONS
                or not isinstance(source_path, str) or not source_path.endswith("/document.pdf")
                or not isinstance(entry.get("bytes"), int) or not entry.get("sha256")):
            continue
        identity = "\0".join((key, str(entry.get("source_revision") or ""),
                              str(entry.get("source_sha256") or ""), str(entry.get("profile") or "")))
        object_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        root = f"derived/{generation}/{object_id}"
        entries.append({
            "key": key,
            "repo": key.split("\0", 1)[0],
            "source_path": key.split("\0", 1)[1],
            "source_extension": extension,
            "profile": entry.get("profile", ""),
            "source_revision": entry.get("source_revision", ""),
            "source_sha256": entry.get("source_sha256", ""),
            "derived_sha256": entry["sha256"],
            "derived_bytes": entry["bytes"],
            "old_path": source_path,
            "new_path": f"{root}/document.pdf",
            "manifest_path": f"{root}/derived-manifest.json",
            "source_uri": f"hf://datasets/{repo}/{source_path}",
            "bucket": bucket,
            "generation": generation,
        })
    return sorted(entries, key=lambda item: item["key"])


def source_hashes(api: HfApi, repo: str, revision: str, paths: list[str]) -> dict[str, str]:
    """Fetch selected source Xet hashes in bounded paths-info batches."""
    result = {}
    for start in range(0, len(paths), 100):
        for item in api.get_paths_info(repo, paths[start:start + 100], revision=revision,
                                       repo_type="dataset"):
            if getattr(item, "path", None) and getattr(item, "xet_hash", None):
                result[item.path] = item.xet_hash
    return result


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be between 1 and 16")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    entries = load_entries(api, args.assets_repo, args.assets_revision, args.generation, args.bucket)
    report = {
        "version": 1,
        "kind": "pdf-derived-migration",
        "assets_repo": args.assets_repo,
        "assets_revision": args.assets_revision,
        "bucket": args.bucket,
        "generation": args.generation,
        "source_extensions": sorted(PAGE_SOURCE_EXTENSIONS),
        "entries": entries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for item in entries:
        counts[item["source_extension"]] = counts.get(item["source_extension"], 0) + 1
    print(f"planned={len(entries)} by_extension={counts} report={args.report}")
    if not args.apply:
        print("dry-run; pass --apply to copy existing derived PDFs")
        return 0
    hashes = source_hashes(api, args.assets_repo, args.assets_revision,
                           [item["old_path"] for item in entries])
    server_copies = [
        ("dataset", args.assets_repo, hashes[item["old_path"]], item["new_path"])
        for item in entries if item["old_path"] in hashes
    ]
    download_entries = [item for item in entries if item["old_path"] not in hashes]
    print(f"server_side={len(server_copies)} download_fallback={len(download_entries)}")
    for start in range(0, len(server_copies), 100):
        api.batch_bucket_files(args.bucket, copy=server_copies[start:start + 100], token=token)
        print(f"copied server batch {min(start + 100, len(server_copies))}/{len(server_copies)}", flush=True)
    fallback_adds = []
    with tempfile.TemporaryDirectory(prefix="pdf-derived-migration-") as temporary:
        for index, item in enumerate(download_entries):
            local = hf_hub_download(repo_id=args.assets_repo, repo_type="dataset",
                                     filename=item["old_path"], revision=args.assets_revision,
                                     token=token, cache_dir=temporary)
            fallback_adds.append((local, item["new_path"]))
            if len(fallback_adds) >= 50:
                api.batch_bucket_files(args.bucket, add=fallback_adds, token=token)
                print(f"uploaded fallback {min(index + 1, len(download_entries))}/{len(download_entries)}", flush=True)
                fallback_adds = []
        if fallback_adds:
            api.batch_bucket_files(args.bucket, add=fallback_adds, token=token)
            print(f"uploaded fallback {len(download_entries)}/{len(download_entries)}", flush=True)
    report["copied"] = len(entries)
    report["server_side"] = len(server_copies)
    report["download_fallback"] = len(download_entries)
    report["already_present"] = 0
    report["copied_paths"] = [item["new_path"] for item in entries]
    args.report.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_adds = []
    for item in entries:
        body = (json.dumps({**item, "kind": "pdf-derived-source", "version": 1},
                           ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        manifest_adds.append((body, item["manifest_path"]))
    for start in range(0, len(manifest_adds), 100):
        api.batch_bucket_files(args.bucket, add=manifest_adds[start:start + 100], token=token)
    print(f"published_derived_manifests={len(manifest_adds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
