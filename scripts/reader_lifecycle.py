#!/usr/bin/env python3
"""Lifecycle records and conservative collection rules for Reader objects."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
try:
    from .shared import READER_ASSETS_BUCKET, PDF_PAGES_BUCKET, PDF_OCR_INPUT_BUCKET
except ImportError:
    from shared import READER_ASSETS_BUCKET, PDF_PAGES_BUCKET, PDF_OCR_INPUT_BUCKET

LIFECYCLE_VERSION = 1
LIFECYCLE_NAME = "reader_lifecycle.json"
CATALOG_VERSION = 1
CATALOG_NAME = "reader_catalog.json"
PROCESSING_PREFIX = "processing/"
TERMINAL_SUCCESS = {"done", "skipped"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty_manifest() -> dict:
    return {"version": LIFECYCLE_VERSION, "files": {}, "orphans": {}}


def processing_record(resources: list[dict], operation: str, run: dict) -> dict:
    """Durable roots until the publisher explicitly hands off their ownership."""
    if not resources or any(not isinstance(item, dict)
                            or item.get("bucket") not in {READER_ASSETS_BUCKET, PDF_PAGES_BUCKET, PDF_OCR_INPUT_BUCKET}
                            or not isinstance(item.get("root"), str)
                            or not item["root"].startswith("objects/")
                            or any(part in {"", ".", ".."} for part in item["root"].split("/"))
                            for item in resources):
        raise ValueError("processing roots require bucket-qualified resources")
    return {"version": 1, "kind": "reader-processing-roots", "status": "processing",
            "operation": operation, "run": dict(run), "resources": resources,
            "created_at": now_iso(), "updated_at": now_iso(),
            "release_policy": "explicit-publisher-handoff"}


def release_processing(record: dict, generation: str, now: str | None = None) -> dict:
    if record.get("status") == "released" and isinstance(record.get("generation"), str) and record["generation"]:
        return dict(record)
    if record.get("kind") != "reader-processing-roots" or record.get("status") not in {
            "processing", "uploaded", "published"}:
        raise ValueError("processing record is not releasable")
    timestamp = now or now_iso()
    return {**record, "status": "released", "generation": generation,
            "released_at": timestamp, "updated_at": timestamp}


def validate_processing_handoff(record: dict, result: dict) -> None:
    paths = set()
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "processing_roots":
                    continue
                if key in {"path", "o", "i", "w", "j", "ocr_manifest"} and isinstance(child, str) \
                        and child.startswith("objects/"):
                    paths.add(child)
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result)
    resources = record.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("processing handoff has no resources")
    for resource in resources:
        root = resource.get("root", "") if isinstance(resource, dict) else ""
        if not root.startswith("objects/") or not any(path.startswith(root + "/") for path in paths):
            raise ValueError("processing root does not belong to the published result")


def empty_catalog() -> dict:
    return {"version": CATALOG_VERSION, "kind": "reader-catalog", "current": None, "generations": {}}


def _generation_id(payload: dict, parent: str | None) -> str:
    material = json.dumps({"payload": payload, "parent": parent},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def catalog_generation(sidecar: dict, parent: dict | None = None,
                       references: list[dict] | None = None,
                       now: str | None = None) -> tuple[dict, str]:
    if sidecar.get("v") != 1 or not isinstance(sidecar.get("f"), dict):
        raise ValueError("invalid sidecar for catalog generation")
    catalog = empty_catalog()
    if parent:
        if parent.get("version") != CATALOG_VERSION or not isinstance(parent.get("generations"), dict):
            raise ValueError("invalid Reader catalog")
        catalog = {"version": CATALOG_VERSION, "kind": "reader-catalog",
                   "current": parent.get("current"),
                   "generations": dict(parent.get("generations") or {})}
    previous = catalog.get("current") or {}
    parent_id = previous.get("generation") if isinstance(previous, dict) else None
    refs = sidecar_references(sidecar) if references is None else references
    normalized = sorted({(item["bucket"], item["path"]) for item in refs})
    digest = hashlib.sha256(json.dumps(sidecar, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    signature = _generation_id({"sidecar_sha256": digest, "references": normalized}, None)
    prior = catalog["generations"].get(parent_id, {})
    if prior.get("signature") == signature:
        return catalog, parent_id
    generation = _generation_id({"signature": signature}, parent_id)
    timestamp = now or now_iso()
    entry = {"generation": generation, "status": "current", "published_at": timestamp,
             "signature": signature, "sidecar_sha256": digest,
             "parent_generation": parent_id,
             "references": normalized,
             "acks": {"hf": False, "pages": False}}
    entry["references"] = [{"bucket": bucket, "path": path} for bucket, path in entry["references"]]
    if isinstance(previous, dict) and previous.get("generation") in catalog["generations"]:
        old = dict(catalog["generations"][previous["generation"]])
        old["status"] = "superseded"
        old["superseded_at"] = timestamp
        catalog["generations"][previous["generation"]] = old
    catalog["generations"][generation] = entry
    catalog["current"] = {"generation": generation, "published_at": timestamp}
    return catalog, generation


def sidecar_references(sidecar: dict) -> list[dict]:
    references = []
    for entry in sidecar.get("f", {}).values():
        if not isinstance(entry, dict):
            continue
        page_bucket = entry.get("b") or READER_ASSETS_BUCKET
        for field, bucket_field in (("p", "b"), ("o", "ob"), ("pd", "pdb"),
                                    ("c", "cb"), ("f", "fb")):
            path = entry.get(field)
            if isinstance(path, str):
                bucket = entry.get(bucket_field) or page_bucket
                if bucket not in {READER_ASSETS_BUCKET, PDF_PAGES_BUCKET, PDF_OCR_INPUT_BUCKET}:
                    raise ValueError(f"sidecar resource lacks current-bucket identity: {field}")
                references.append({"bucket": bucket, "path": path})
    return references


def generation_live(entry: dict, current: str | None, now=None, grace_days: int = 30) -> bool:
    if entry.get("generation") == current or entry.get("status") == "current":
        return True
    if not isinstance(entry.get("superseded_at"), str):
        return True
    try:
        superseded = datetime.fromisoformat(entry["superseded_at"].replace("Z", "+00:00"))
        if superseded.tzinfo is None:
            return True
    except (TypeError, ValueError):
        return True
    clock = now or datetime.now(timezone.utc)
    if clock - superseded < timedelta(days=grace_days):
        return True
    acks = entry.get("replacement_acks")
    return not (isinstance(acks, dict) and acks.get("hf") is True and acks.get("pages") is True)


def asset_record(result: dict) -> dict:
    return {
        "key": result["key"],
        "path": result["path"],
        "paths": list(result.get("bucket_paths") or [result["path"]]),
        "sha256": result.get("sha256", ""),
        "bytes": int(result.get("bytes") or 0),
        "source_sha256": result.get("source_sha256", ""),
        "source_revision": result.get("source_revision", ""),
        "profile": result.get("profile", ""),
        "phase": "final",
        "consumers": {},
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def merge(manifest: dict, updates: list[dict]) -> dict:
    if manifest.get("version") != LIFECYCLE_VERSION or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid Reader lifecycle manifest")
    result = {"version": LIFECYCLE_VERSION, "files": dict(manifest["files"]),
              "orphans": dict(manifest.get("orphans") or {})}
    for update in updates:
        key = update.get("key")
        if not key:
            continue
        previous = dict(result["files"].get(key) or {})
        merged = {**previous, **update, "updated_at": now_iso()}
        if previous.get("created_at"):
            merged["created_at"] = previous["created_at"]
        result["files"][key] = merged
    return result


def mark_orphans(manifest: dict, paths: set[str], today: str) -> dict:
    orphans = dict(manifest.get("orphans") or {})
    for path in paths:
        if path not in orphans:
            orphans[path] = {"since": today}
    for path in list(orphans):
        if path not in paths:
            orphans.pop(path, None)
    return {**manifest, "orphans": dict(sorted(orphans.items()))}
