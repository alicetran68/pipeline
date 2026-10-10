#!/usr/bin/env python3
"""Read-only, bucket-qualified inventory and dependency graph for Reader GC."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import posixpath
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from .reader_bucket_store import HubBucketStore, S3BucketStore
    from .shared import READER_ASSETS_BUCKET, PDF_PAGES_BUCKET, PDF_OCR_INPUT_BUCKET
    from . import reader_lifecycle
except ImportError:
    from reader_bucket_store import HubBucketStore, S3BucketStore
    from shared import READER_ASSETS_BUCKET, PDF_PAGES_BUCKET, PDF_OCR_INPUT_BUCKET
    import reader_lifecycle


BUCKETS = (READER_ASSETS_BUCKET, PDF_PAGES_BUCKET, PDF_OCR_INPUT_BUCKET)
MANAGED_PREFIXES = (
    "objects/", "ebook-chapters/", "staging/", "pages/", "documents/",
    "chapters/", "media/", "native/", "derived/",
)
PATH_FIELDS = {
    "path", "paths", "object", "fallback", "manifest", "manifest_path", "root",
    "new_path", "reader_assets_path", "ocr_manifest", "p", "o", "i", "w", "j",
    "c", "f", "pd", "document", "stream", "text", "resources", "inputs", "outputs",
}
BUCKET_FIELDS = {
    "p": "b", "o": "ob", "c": "cb", "f": "fb", "pd": "pdb",
    "i": "ibucket", "w": "wbucket", "j": "jbucket",
    "reader_assets_path": "reader_assets_bucket",
}
MANIFEST_KINDS = {
    "pdf-pages", "image-page-stream", "image-pages", "pdf-render", "pdf-render-range",
    "pdf-ocr", "ebook-chapters", "pdf-derived-source", "office-document-stream",
    "spreadsheet-html-stream", "text-document-stream", "web-document-stream",
    "static-pdf-stream", "pdf-document-stream", "static-pdf-document-stream", "epub-chapters",
    "pdf-text-layer-index", "pdf-text-partition", "pdf-text-review",
    "pdf-reading", "pdf-preview-partition",
}


def is_root(path: str) -> bool:
    name = posixpath.basename(path)
    if path.startswith("reader-index/generations/"):
        return False
    return (path.startswith("reader-index/") and path.endswith((".json", ".json.gz"))) or (
        name == "index.json" or name.startswith("index-") and name.endswith(".json"))


def is_manifest(path: str) -> bool:
    name = posixpath.basename(path)
    return name.endswith("-manifest.json") or name.startswith("render-range-") and name.endswith(".json")


def decode(raw: bytes) -> dict:
    payload = json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


class ReferenceGraph:
    def __init__(self, store, include_input: bool = True, generation_grace_days: int = 30):
        self.store = store
        self.include_input = include_input
        self.files = {}
        self.ordered = {}
        self.references = set()
        self.pending = set()
        self.seen = set()
        self.root_hashes = {}
        self.blockers = set()
        self.missing = set()
        self.ambiguous = set()
        self.trees = set()
        self.generation_grace_days = generation_grace_days
        self.catalog = {}

    def load(self, ref: tuple[str, str]) -> dict | None:
        try:
            raw = self.store.read_bytes(*ref)
            payload = decode(raw)
            if is_root(ref[1]):
                self.root_hashes[ref] = hashlib.sha256(raw).hexdigest()
            return payload
        except Exception as error:
            # A permissions/transport failure must never resemble an empty root.
            self.blockers.add(f"unreadable JSON: {ref[0]}:{ref[1]} ({type(error).__name__})")
            return None

    def mark(self, bucket: str, path: str, *, tree: bool = False):
        if bucket not in BUCKETS:
            self.blockers.add(f"unsupported reference bucket: {bucket}")
            return
        if (not path or path.startswith("/") or "\\" in path or "\0" in path
                or any(part in {".", "..", ""} for part in path.split("/"))):
            self.blockers.add(f"invalid object path: {bucket}:{path}")
            return
        ref = (bucket, path)
        self.references.add(ref)
        if bucket not in self.files:
            self.blockers.add(f"referenced bucket was not inventoried: {bucket}")
            return
        if path not in self.files[bucket] and not tree:
            self.missing.add(ref)
            return
        if is_manifest(path) and ref not in self.seen:
            self.pending.add(ref)
        # HTML/CSS/XHTML, fonts, media companions and manifest siblings are an
        # immutable bundle. Keep the whole bundle instead of parsing markup.
        root = path if tree else posixpath.dirname(path)
        if root and not is_root(path):
            if (bucket, root) in self.trees:
                return
            self.trees.add((bucket, root))
            prefix = root + "/"
            ordered = self.ordered[bucket]
            start = bisect.bisect_left(ordered, prefix)
            end = bisect.bisect_left(ordered, prefix + "\U0010ffff")
            for child in ordered[start:end]:
                child_ref = (bucket, child)
                self.references.add(child_ref)
                if is_manifest(child) and child_ref not in self.seen:
                    self.pending.add(child_ref)

    def path(self, value: str, bucket: str, origin: str, explicit: bool, field: str):
        if is_manifest(origin) and field in PATH_FIELDS:
            relative = posixpath.join(posixpath.dirname(origin), value)
            if value.startswith(("pages/page-", "chapters/chapter-", "resources/")) or (
                    relative in self.files.get(bucket, set())):
                self.mark(bucket, relative, tree=field == "root")
                return
        if value.startswith(MANAGED_PREFIXES) or value.startswith("reader-index/"):
            if explicit:
                self.mark(bucket, value, tree=field == "root")
                return
            matches = [candidate for candidate, files in self.files.items()
                       if value in files or field == "root" and any(p.startswith(value + "/") for p in files)]
            if len(matches) > 1:
                self.ambiguous.add(value)
            if not matches:
                if "/ocr-input/" in value or value.lower().endswith(".jxl"):
                    bucket = PDF_OCR_INPUT_BUCKET
                self.mark(bucket, value, tree=field == "root")
            for candidate in matches:
                self.mark(candidate, value, tree=field == "root")
        elif field in PATH_FIELDS and is_manifest(origin) and not value.startswith(("https:", "http:")):
            # Only manifests use bundle-relative paths. Index `path` fields
            # commonly describe upstream source files and are not object keys.
            self.mark(bucket, posixpath.join(posixpath.dirname(origin), value), tree=field == "root")

    def walk(self, value, bucket: str, origin: str, explicit: bool = False, field: str = ""):
        if isinstance(value, dict):
            if origin.endswith("reader_catalog.json") and value.get("kind") == "reader-catalog":
                current = (value.get("current") or {}).get("generation")
                for generation in (value.get("generations") or {}).values():
                    if isinstance(generation, dict) and reader_lifecycle.generation_live(
                            generation, current, grace_days=self.generation_grace_days):
                        self.walk(generation, bucket, origin, explicit, "generation")
                return
            if origin.startswith("reader-index/processing/") and value.get("status") == "released":
                generation = value.get("generation")
                if isinstance(generation, str) and generation in self.catalog.get("generations", {}):
                    return
                self.blockers.add(f"released root lacks catalog handoff: {bucket}:{origin}")
            context = value.get("bucket")
            if context is not None:
                if not isinstance(context, str) or context not in BUCKETS:
                    self.blockers.add(f"invalid bucket in {bucket}:{origin}")
                    return
                bucket, explicit = context, True
            for key, item in value.items():
                if key in {"orphans", "source_url", "source_path", "source", "text", "error",
                           "processing_roots", "observations", "components"}:
                    continue
                if key == "path" and (value.get("kind") in {"pdf-render", "pdf-render-range"} or (
                        value.get("repo") and "new_path" not in value and any(
                            name in value for name in ("object", "manifest", "reader_assets_path")))):
                    continue
                target = value.get(BUCKET_FIELDS.get(key, ""))
                # `ob` also means OCR object bytes in compact page entries.
                if not isinstance(target, str) and key == "o":
                    target = value.get("obucket")
                if key == "o" and not target and origin.endswith("reader_assets.json.gz"):
                    target = value.get("b")
                if key == "p" and isinstance(value.get("b"), str):
                    target = value["b"]
                if key in {"i", "j"} and isinstance(item, str) and not target:
                    target = PDF_OCR_INPUT_BUCKET
                if isinstance(target, str):
                    if target not in BUCKETS:
                        self.blockers.add(f"unsupported reference bucket: {target}")
                        continue
                    self.walk(item, target, origin, True, key)
                else:
                    self.walk(item, bucket, origin, explicit, key)
        elif isinstance(value, list):
            for item in value:
                self.walk(item, bucket, origin, explicit, field)
        elif isinstance(value, str) and (field in PATH_FIELDS or field == "references"):
            self.path(value, bucket, origin, explicit, field)

    def validate(self, payload: dict, bucket: str, path: str) -> bool:
        if is_root(path):
            if payload.get("kind") in {"reader-v3-automation", "reader-v3-correction-state"}:
                if (payload.get("version") != 1 or not isinstance(payload.get("tasks"), dict)
                        or not isinstance(payload.get("days"), dict)
                        or any(not isinstance(task, dict) for task in payload["tasks"].values())):
                    self.blockers.add(f"invalid v3 task state: {bucket}:{path}")
                    return False
            elif payload.get("kind") == "pdf-text-correction":
                try:
                    try:
                        from .pdf_text_layer import sha
                    except ImportError:
                        from pdf_text_layer import sha
                    for field in ("base_generation", "raw_sha256", "page_identity"):
                        sha(payload.get(field))
                    if (payload.get("version") != 1 or type(payload.get("page")) is not int
                            or payload["page"] < 1 or not isinstance(payload.get("replacements"), list)
                            or not isinstance(payload.get("evidence"), dict)):
                        raise ValueError("invalid proposal")
                except (ValueError, TypeError):
                    self.blockers.add(f"invalid v3 correction proposal: {bucket}:{path}")
                    return False
            elif "lifecycle" in posixpath.basename(path):
                if not isinstance(payload.get("orphans", {}), dict):
                    self.blockers.add(f"invalid lifecycle: {bucket}:{path}")
                    return False
            elif path.endswith("reader_assets.json.gz"):
                if payload.get("v") != 1 or not isinstance(payload.get("f"), dict):
                    self.blockers.add(f"invalid sidecar: {bucket}:{path}")
                    return False
            elif path.endswith("reader_catalog.json"):
                if payload.get("version") != 1 or payload.get("kind") != "reader-catalog" \
                        or not isinstance(payload.get("generations"), dict):
                    self.blockers.add(f"invalid catalog: {bucket}:{path}")
                    return False
                current = payload.get("current")
                if not isinstance(current, dict) or current.get("generation") not in payload["generations"]:
                    self.blockers.add(f"missing current catalog generation: {bucket}:{path}")
                    return False
            elif path.endswith("reader_gc_state.json"):
                if payload.get("version") != 1 or not isinstance(payload.get("orphans"), dict):
                    self.blockers.add(f"invalid GC state: {bucket}:{path}")
                    return False
            elif not any(isinstance(payload.get(key), (dict, list)) for key in (
                    "files", "resources", "leases", "generations")):
                self.blockers.add(f"unknown root schema: {bucket}:{path}")
                return False
            for key in ("files", "f", "resources", "leases", "generations"):
                entries = payload.get(key)
                if entries is None:
                    continue
                if not isinstance(entries, (dict, list)):
                    self.blockers.add(f"invalid root entries: {bucket}:{path}")
                    return False
                entries = entries.values() if isinstance(entries, dict) else entries
                if any(not isinstance(entry, dict) for entry in entries):
                    self.blockers.add(f"invalid root entry: {bucket}:{path}")
                    return False
        elif is_manifest(path):
            if payload.get("kind") not in MANIFEST_KINDS:
                self.blockers.add(f"unknown manifest kind: {bucket}:{path}")
                return False
            if payload.get("kind") in {"pdf-text-layer-index", "pdf-reading"}:
                try:
                    try:
                        from .pdf_reading_v3 import validate_text_manifest, validate_reading_manifest
                    except ImportError:
                        from pdf_reading_v3 import validate_text_manifest, validate_reading_manifest
                    validator = validate_text_manifest if payload["kind"] == "pdf-text-layer-index" else validate_reading_manifest
                    validator(payload)
                except (ValueError, KeyError, TypeError):
                    self.blockers.add(f"invalid v3 manifest: {bucket}:{path}")
                    return False
            if payload.get("kind") in {"pdf-pages", "image-page-stream", "image-pages"}:
                count = payload.get("page_count")
                if type(count) is not int or count < 1:
                    self.blockers.add(f"invalid page count: {bucket}:{path}")
                    return False
                if payload["kind"] == "pdf-pages":
                    root = posixpath.dirname(path)
                    for number in range(1, count + 1):
                        self.mark(bucket, f"{root}/pages/page-{number:06d}.webp")
        return True

    def build(self) -> dict:
        if not self.include_input:
            self.blockers.add("input bucket inventory was explicitly skipped")
        for bucket in BUCKETS:
            if bucket == PDF_OCR_INPUT_BUCKET and not self.include_input:
                continue
            try:
                self.files[bucket] = self.store.list_files(bucket, ("",))
                self.ordered[bucket] = sorted(self.files[bucket])
            except Exception as error:
                self.blockers.add(f"inventory unavailable: {bucket} ({type(error).__name__})")
        required = (READER_ASSETS_BUCKET, "reader-index/reader_assets.json.gz")
        if required[1] not in self.files.get(required[0], set()):
            self.blockers.add("canonical Reader sidecar is missing")
        catalog_ref = (READER_ASSETS_BUCKET, "reader-index/reader_catalog.json")
        if catalog_ref[1] in self.files.get(catalog_ref[0], set()):
            self.catalog = self.load(catalog_ref) or {}
        for bucket, files in self.files.items():
            for path in sorted(files):
                if is_root(path):
                    ref = (bucket, path)
                    self.references.add(ref)
                    self.pending.add(ref)
        while self.pending:
            ref = min(self.pending)
            self.pending.remove(ref)
            if ref in self.seen:
                continue
            self.seen.add(ref)
            payload = self.load(ref)
            if payload is not None and self.validate(payload, *ref):
                self.walk(payload, *ref)
        # Publication currently has several independent locks. Re-read every
        # root and its key inventory to detect a changing snapshot.
        for bucket, files in self.files.items():
            try:
                latest = self.store.list_files(bucket, ("",))
                if {p for p in latest if is_root(p)} != {p for p in files if is_root(p)}:
                    self.blockers.add(f"root inventory changed: {bucket}")
            except Exception as error:
                self.blockers.add(f"root inventory recheck failed: {bucket} ({type(error).__name__})")
        for ref, digest in self.root_hashes.items():
            try:
                if hashlib.sha256(self.store.read_bytes(*ref)).hexdigest() != digest:
                    self.blockers.add(f"root changed: {ref[0]}:{ref[1]}")
            except Exception as error:
                self.blockers.add(f"root recheck failed: {ref[0]}:{ref[1]} ({type(error).__name__})")
        if self.missing:
            self.blockers.add("referenced objects are missing")
        buckets = {}
        complete = not self.blockers
        for bucket, files in self.files.items():
            unreferenced = sorted(path for path in files if (bucket, path) not in self.references
                                  and path.startswith(MANAGED_PREFIXES))
            unmanaged = sorted(path for path in files if (bucket, path) not in self.references
                               and not path.startswith(MANAGED_PREFIXES))
            buckets[bucket] = {
                "objects": len(files),
                "referenced": sum((bucket, path) in self.references for path in files),
                "formats": dict(sorted(Counter(
                    "".join(Path(path).suffixes[-2:]).lower() if path.endswith(".gz")
                    else Path(path).suffix.lower() or "(none)" for path in files).items())),
                "unreferenced": unreferenced,
                "candidates": unreferenced if complete else [],
                "unmanaged": unmanaged,
            }
        return {
            "version": 1, "mode": "report-only", "graph_complete": complete,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "deletion_enabled": False,
            "deletion_blockers": ["remaining producers lack shared mutation coordination"],
            "blockers": sorted(self.blockers), "buckets": buckets,
            "missing": [f"{b}:{p}" for b, p in sorted(self.missing)],
            "ambiguous_paths": sorted(self.ambiguous),
            "roots": [f"{b}:{p}" for b, p in sorted(self.root_hashes)],
        }


def read_gc_state(store) -> dict:
    try:
        payload = decode(store.read_bytes(READER_ASSETS_BUCKET, "reader-index/reader_gc_state.json"))
    except FileNotFoundError:
        return {"version": 1, "kind": "reader-gc-state", "orphans": {}}
    if payload.get("version") != 1 or payload.get("kind") != "reader-gc-state" \
            or not isinstance(payload.get("orphans"), dict):
        raise RuntimeError("invalid Reader GC state")
    return payload


def plan_retention(state: dict, report: dict, grace_days: int, limit: int, today=None) -> tuple[dict, dict]:
    if not report.get("graph_complete"):
        raise RuntimeError("reference graph is incomplete; refusing orphan observations")
    if grace_days < 1 or limit < 0:
        raise ValueError("orphan grace must be positive and limit non-negative")
    today = today or date.today()
    current = {f"{bucket}:{path}" for bucket, data in report["buckets"].items()
               for path in data.get("candidates", [])}
    orphans = {key: value for key, value in state.get("orphans", {}).items()
               if key in current and isinstance(value, dict)}
    for key in sorted(current):
        orphans.setdefault(key, {"first_seen": today.isoformat()})
    updated = {"version": 1, "kind": "reader-gc-state", "orphans": dict(sorted(orphans.items()))}
    cutoff = today - timedelta(days=grace_days)
    eligible = []
    for key, value in updated["orphans"].items():
        try:
            if date.fromisoformat(value["first_seen"]) <= cutoff:
                eligible.append(key)
        except (KeyError, TypeError, ValueError):
            continue
    eligible = sorted(eligible)[:limit] if limit else sorted(eligible)
    return updated, {"marked": len(current), "past_grace": len(eligible),
                     "past_grace_paths": eligible, "deletion_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-input-bucket", action="store_true")
    parser.add_argument("--record-observations", action="store_true")
    parser.add_argument("--storage", choices=("hub", "s3"), default="hub")
    parser.add_argument("--grace-days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generation-grace-days", type=int, default=30)
    args = parser.parse_args()
    if args.grace_days < 1 or args.limit < 0 or args.generation_grace_days < 30:
        parser.error("orphan grace must be positive, generation grace at least 30 days and limit non-negative")
    store = HubBucketStore() if args.storage == "hub" else S3BucketStore()
    report = ReferenceGraph(store, not args.skip_input_bucket,
                            args.generation_grace_days).build()
    if report["graph_complete"]:
        state, report["retention"] = plan_retention(read_gc_state(store), report, args.grace_days, args.limit)
        if args.record_observations:
            store.put_json(READER_ASSETS_BUCKET, "reader-index/reader_gc_state.json", state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    for bucket, counts in report["buckets"].items():
        print(f"{bucket}: objects={counts['objects']} referenced={counts['referenced']} "
              f"unreferenced={len(counts['unreferenced'])} unmanaged={len(counts['unmanaged'])}")
    print(f"graph_complete={report['graph_complete']} deletion_enabled=False report={args.output}")
    return 0 if report["graph_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
