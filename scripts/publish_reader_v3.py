#!/usr/bin/env python3
"""Stage verified immutable v3 candidates, then promote one serialized pointer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
import uuid

try:
    from . import pdf_reading_v3 as v3, pdf_text_layer as text, reader_lifecycle, shared, pdf_ocr, ocr_layout
    from .reader_bucket_store import HubBucketStore
except ImportError:
    import pdf_reading_v3 as v3
    import pdf_text_layer as text
    import reader_lifecycle
    import shared
    import pdf_ocr
    import ocr_layout
    from reader_bucket_store import HubBucketStore

ASSETS = shared.READER_ASSETS_BUCKET
POINTER = "reader-index/v3/current.json"
PROTOCOL = "central-reader-sidecar-v3-v1"


class PublicationReviewRequired(ValueError):
    pass


PUBLIC_OBJECT = re.compile(
    r"objects/[0-9a-f]{2}/[0-9a-f]{64}/[0-9a-f]{16}/(?:document\.pdf|"
    r"reading-manifest\.json|page-map\.json\.gz|text-layer-manifest\.json|"
    r"text/(?:page-[0-9]{6}\.json\.gz|book-text\.json\.gz|search-[0-9]{6}-[0-9]{6}\.json\.gz|partition-[0-9]{6}-[0-9]{6}-manifest\.json)|"
    r"preview/(?:page-[0-9]{6}\.(?:png|webp|jpeg)|partition-[0-9]{6}-[0-9]{6}-manifest\.json))")


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def metadata(bucket, path, raw, role="runtime"):
    return {"bucket": bucket, "path": path, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "role": role}


def index_resource(ref):
    if (not isinstance(ref, dict) or ref.get("bucket") != ASSETS
            or not isinstance(ref.get("path"), str)
            or not ref["path"].startswith("reader-index/v3/")
            or any(part in {"", ".", ".."} for part in ref["path"].split("/"))
            or "\\" in ref["path"] or any(ord(c) < 32 for c in ref["path"])
            or "?" in ref["path"] or "#" in ref["path"]
            or type(ref.get("bytes")) is not int or ref["bytes"] < 1):
        raise ValueError("invalid v3 catalog resource")
    text.sha(ref.get("sha256"))
    return ref


def validate_reading_ref(ref):
    v3.resource(ref)
    if (ref["bucket"] != shared.PDF_PAGES_BUCKET or ref["role"] != "runtime"
            or not re.fullmatch(r"objects/[0-9a-f]{2}/[0-9a-f]{64}/[0-9a-f]{16}/reading-manifest\.json", ref["path"])):
        raise ValueError("candidate does not name a public reading manifest")
    return ref


def read_index(store, ref):
    index_resource(ref)
    raw = store.read_bytes(ref["bucket"], ref["path"])
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("v3 catalog checksum mismatch")
    return json.loads(raw)


def validate_catalog(catalog):
    if (catalog.get("version") != 3 or catalog.get("kind") != "reader-reading-catalog"
            or not isinstance(catalog.get("files"), dict)):
        raise ValueError("invalid reading catalog")
    body = {k: v for k, v in catalog.items() if k != "generation"}
    if text.digest(body) != catalog.get("generation"):
        raise ValueError("reading catalog generation mismatch")
    for key, entry in catalog["files"].items():
        if not isinstance(key, str) or not key or not isinstance(entry, dict):
            raise ValueError("invalid reading catalog entry")
        validate_reading_ref(entry["resource"])
        text.sha(entry["source_sha256"])
        text.sha(entry["reading_generation"])
        if (entry["resource"]["bucket"] != shared.PDF_PAGES_BUCKET
                or not entry["resource"]["path"].startswith(
                    f"objects/{entry['source_sha256'][:2]}/{entry['source_sha256']}/")
                or not entry["resource"]["path"].endswith("/reading-manifest.json")):
            raise ValueError("catalog reading resource identity mismatch")
    return catalog


def current(store):
    try:
        pointer = json.loads(store.read_bytes(ASSETS, POINTER))
    except FileNotFoundError:
        return None, {"files": {}}
    if pointer.get("version") != 1 or pointer.get("kind") != "reader-reading-pointer":
        raise ValueError("invalid reading pointer")
    generation = text.sha(pointer.get("generation"))
    ref = index_resource(pointer["catalog"])
    if ref["path"] != f"reader-index/v3/generations/{generation}/catalog.json":
        raise ValueError("pointer catalog path mismatch")
    catalog = validate_catalog(read_index(store, ref))
    if catalog["generation"] != generation:
        raise ValueError("pointer/catalog generation mismatch")
    return pointer, catalog


def immutable_put(store, bucket, path, raw):
    try:
        prior = store.read_bytes(bucket, path)
    except FileNotFoundError:
        store.put_bytes(bucket, path, raw)
    else:
        if prior != raw:
            raise ValueError(f"immutable object conflict: {bucket}:{path}")
    actual = store.read_bytes(bucket, path)
    if actual != raw:
        raise ValueError("uploaded immutable object differs from candidate")


def assert_writer(store):
    store.assert_serialized_writer(PROTOCOL)


class CandidateReader:
    def __init__(self, store, bundle=None):
        self.store, self.bundle = store, bundle.resolve() if bundle else None
        self.refs, self.local = {}, {}

    def __call__(self, ref):
        v3.resource(ref)
        if ref["role"] == "runtime" and not (
                ref["bucket"] == shared.PDF_PAGES_BUCKET and PUBLIC_OBJECT.fullmatch(ref["path"]) or
                ref["bucket"] == shared.PDF_PAGES_BUCKET and re.fullmatch(
                    r"derived/[A-Za-z0-9._-]+/[a-z0-9]{32}/document\.pdf", ref["path"]) or
                ref["bucket"] == shared.READER_ASSETS_BUCKET and re.fullmatch(
                    r"documents/pdf/[a-z0-9_-]+/[0-9a-f]{64}/document\.pdf", ref["path"])):
            raise ValueError("runtime candidate resource lacks a public Reader path")
        key = (ref["bucket"], ref["path"])
        if key in self.refs and self.refs[key]["sha256"] != ref["sha256"]:
            raise ValueError("conflicting candidate reference digests")
        self.refs[key] = dict(ref)
        if self.bundle and ref["bucket"] == shared.PDF_PAGES_BUCKET:
            target = (self.bundle / ref["path"]).resolve()
            if not target.is_relative_to(self.bundle):
                raise ValueError("candidate path escapes bundle")
            if target.is_file():
                self.local[key] = target
                return target.read_bytes()
        return self.store.read_bytes(*key)

    def closure(self):
        """Verify provenance/review dependencies too, preserving all named evidence."""
        seen = set()
        while set(self.refs) - seen:
            key = min(set(self.refs) - seen)
            seen.add(key)
            ref = self.refs[key]
            raw = v3.verified_read(ref, self)
            if not ref["path"].endswith((".json", ".json.gz")):
                continue
            payload = v3.decode(raw)
            def walk(value):
                if isinstance(value, dict):
                    if {"bucket", "path", "sha256", "bytes", "role"} <= value.keys():
                        if value["path"].startswith("reader-index/v3/"):
                            index_resource(value)
                            read_index(self.store, value)
                        else:
                            self(value)
                    else:
                        for child in value.values():
                            walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)
            walk(payload)


def stage(store, ref, bundle=None, *, apply=False):
    validate_reading_ref(ref)
    reader = CandidateReader(store, bundle)
    manifest = v3.verify_reading(ref, reader)
    if not manifest["preview"]["complete"] or not manifest.get("text_layer"):
        raise ValueError("promotion candidate requires complete previews and text")
    reader.closure()
    body = {"version": 1, "kind": "reader-reading-candidate", "source_key": manifest["source_key"],
            "source_sha256": manifest["source_sha256"], "reading_generation": manifest["generation"],
            "reading": ref, "components": manifest["components"], "page_count": manifest["page_count"],
            "resources": [reader.refs[key] for key in sorted(reader.refs)]}
    identity = text.digest(body)
    path = f"reader-index/v3/candidates/{identity}.json"
    raw = encode(body)
    report = {"candidate": metadata(ASSETS, path, raw), "source_key": body["source_key"],
              "page_count": body["page_count"], "resources_verified": len(reader.refs),
              "local_objects": len(reader.local), "applied": apply}
    if not apply:
        return report
    assert_writer(store)
    roots = sorted({(bucket, "/".join(path.split("/")[:4])) for bucket, path in reader.local})
    protection_path = "reader-index/processing/v3-" + uuid.uuid4().hex + ".json"
    protection = reader_lifecycle.processing_record(
        [{"bucket": bucket, "root": root} for bucket, root in roots] or
        [{"bucket": ref["bucket"], "root": str(Path(ref["path"]).parent)}],
        "v3-candidate-upload", {"repository": os.environ.get("GITHUB_REPOSITORY", ""),
                                "run_id": os.environ.get("GITHUB_RUN_ID", "")})
    store.put_bytes(ASSETS, protection_path, encode(protection))
    for key in sorted(reader.local):
        immutable_put(store, *key, reader.local[key].read_bytes())
    # Remote readback must stand on its own, not on the local upload workspace.
    remote = CandidateReader(store)
    v3.verify_reading(ref, remote)
    remote.closure()
    immutable_put(store, ASSETS, path, raw)
    protection.update(status="uploaded", updated_at=reader_lifecycle.now_iso(), candidate=report["candidate"])
    store.put_bytes(ASSETS, protection_path, encode(protection))
    report["processing_root"] = protection_path
    return report


def import_primary(spec, workspace):
    """Materialize an original PDF from its immutable dataset revision."""
    origin = spec.get("primary_source")
    if origin is None:
        return
    ref = v3.resource(spec["primary"])
    source = text.sha(spec.get("source_sha256"))
    if (not isinstance(origin, dict) or not isinstance(origin.get("repo"), str)
            or not re.fullmatch(r"VoiceOfML/[A-Za-z0-9._-]+", origin["repo"])
            or not isinstance(origin.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", origin["revision"])
            or not isinstance(origin.get("path"), str)
            or any(part in {"", ".", ".."} for part in origin["path"].split("/"))
            or "\\" in origin["path"] or not origin["path"].lower().endswith(".pdf")
            or spec.get("source_key") != origin["repo"] + "\0" + origin["path"]
            or ref["bucket"] != shared.PDF_PAGES_BUCKET or ref["role"] != "runtime"
            or ref["sha256"] != source or ref["bytes"] > 512 * 1024 * 1024
            or not re.fullmatch(r"objects/" + source[:2] + "/" + source + r"/[0-9a-f]{16}/document\.pdf", ref["path"])):
        raise ValueError("primary import requires matching pinned original-source identity")
    from huggingface_hub import HfApi, hf_hub_download
    entries = HfApi().get_paths_info(origin["repo"], [origin["path"]], repo_type="dataset", revision=origin["revision"])
    if len(entries) != 1 or entries[0].size != ref["bytes"]:
        raise ValueError("pinned primary source size mismatch")
    cached = hf_hub_download(origin["repo"], origin["path"], repo_type="dataset", revision=origin["revision"])
    raw = Path(cached).read_bytes()
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != source:
        raise ValueError("pinned primary source checksum mismatch")
    target = workspace / ref["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def build_stage(store, spec, workspace, *, apply=False):
    """Explicit single-book generation from pinned PDF and complete OCR objects."""
    source = text.sha(spec.get("source_sha256"))
    key = spec.get("source_key")
    if not isinstance(key, str) or not key:
        raise ValueError("single-book source key required")
    if spec.get("reindex_reading"):
        reader = CandidateReader(store, workspace)
        previous = v3.verify_reading(spec["reindex_reading"], reader)
        if previous["source_key"] != key or previous["source_sha256"] != source:
            raise ValueError("reindex source identity mismatch")
        effective = v3.repartition_text_bundle(previous["text_layer"], reader, workspace)
        previews = [entry for part in previous["preview"]["partitions"]
                    for entry in v3.decode(v3.verified_read(part["resource"], reader))["pages"]]
        reading_spec = {"source_key": key, "source_sha256": source,
                        "primary": previous["primary"]["resource"], "previews": previews,
                        "text_layer": effective, "require_complete_preview": True}
        ref, manifest = v3.build_reading(reading_spec, reader, workspace)
        result = stage(store, ref, workspace, apply=apply)
        result.update(reading=ref, components=manifest["components"])
        return result
    import_primary(spec, workspace)
    reader = CandidateReader(store, workspace)
    if spec.get("rebuild_native_text"):
        if spec.get("primary", {}).get("path") is None:
            raise ValueError("native rebuild requires primary PDF")
        primary = workspace / spec["primary"]["path"]
        probe = pdf_ocr.probe_pdf(primary)
        if probe["classification"] != "native-text":
            raise PublicationReviewRequired("native rebuild source is not a complete native-text PDF")
        source = text.sha(spec["source_sha256"])
        identity = text.digest({"source": source, "native_rebuild": 1, "profile": pdf_ocr.asset_profile()})
        root = Path("objects") / source[:2] / source / identity[:16]
        pages = []
        for number, parsed in pdf_ocr.native_pages(primary, range(1, probe["page_count"] + 1)).items():
            payload = pdf_ocr.page_payload(number, parsed["width"], parsed["height"], parsed["blocks"], "native")
            payload.update(ocr_layout.arrange(payload["blocks"], payload["width"], payload["height"], {},
                                               include_writing_modes=True))
            raw_ref = v3.stored(workspace / root / "ocr" / f"page-{number:06d}.json.gz", workspace,
                                payload, compressed=True, role="provenance")
            pages.append({"p": number, "source": "native", "width": parsed["width"],
                          "height": parsed["height"], "chars": len(payload["text"]), "text": payload["text"],
                          "text_spans": payload["text_spans"], "layout": payload["layout"],
                          "o": raw_ref["path"], "os": raw_ref["sha256"], "ob": raw_ref["bytes"]})
        raw_ocr = {"kind": "pdf-ocr", "complete": True, "source_sha256": source,
                   "page_count": probe["page_count"], "language": "ch", "pages": pages}
    else:
        raw_ocr = v3.decode(v3.verified_read(spec["ocr_manifest"], reader))
    if raw_ocr.get("source_sha256") != source:
        raise ValueError("OCR/source identity mismatch")
    text_ref = raw_ocr.get("text_layer")
    if text_ref:
        v3.verify_text_bundle(text_ref, reader, source, raw_ocr["page_count"])
    else:
        text_ref = v3.backfill_text(raw_ocr, reader, workspace)
    if spec.get("require_usable_text"):
        text_index = v3.decode(v3.verified_read(text_ref, reader))
        book = v3.decode(v3.verified_read(text_index["book_text"], reader))
        if not any(char.isalnum() for page in book["pages"] for char in page["text"]):
            raise PublicationReviewRequired("derived document has no usable recognized text; inspect source conversion")
    reading_spec = {"source_key": key, "source_sha256": source, "primary": spec["primary"],
                    "text_layer": text_ref, "require_complete_preview": True}
    reading_spec = v3.generate_previews(reading_spec, reader, workspace,
                                        dpi=spec.get("dpi", 150), max_pixels=spec.get("max_pixels", 50000000))
    ref, manifest = v3.build_reading(reading_spec, reader, workspace)
    result = stage(store, ref, workspace, apply=apply)
    result["reading"] = ref
    result["components"] = manifest["components"]
    return result


def promote(store, candidate_ref, expected_parent, *, apply=False):
    candidate = read_index(store, candidate_ref)
    if candidate.get("version") != 1 or candidate.get("kind") != "reader-reading-candidate":
        raise ValueError("invalid staged reading candidate")
    if candidate_ref["path"] != f"reader-index/v3/candidates/{text.digest(candidate)}.json":
        raise ValueError("staged candidate path/content mismatch")
    validate_reading_ref(candidate["reading"])
    verified = CandidateReader(store)
    manifest = v3.verify_reading(candidate["reading"], verified)
    verified.closure()
    if (not manifest["preview"]["complete"] or not manifest.get("text_layer")
            or candidate["source_key"] != manifest["source_key"]
            or candidate["reading_generation"] != manifest["generation"]
            or candidate["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("staged candidate differs from verified reading generation")
    pointer, old = current(store)
    parent = pointer["generation"] if pointer else None
    key = manifest["source_key"]
    entry = {"resource": candidate["reading"], "source_sha256": manifest["source_sha256"],
             "reading_generation": manifest["generation"], "page_count": manifest["page_count"],
             "components": manifest["components"]}
    if old["files"].get(key) == entry:
        return {"generation": parent, "applied": apply, "unchanged": True, "pointer": pointer}
    if parent != expected_parent:
        raise ValueError("stale expected parent; replan against current catalog")
    files = {**old["files"], key: entry}
    catalog = {"version": 3, "kind": "reader-reading-catalog", "files": files,
               "parent_generation": parent}
    return commit_catalog(store, catalog, pointer, apply=apply)


def commit_catalog(store, catalog, pointer, *, apply=False):
    parent = pointer["generation"] if pointer else None
    if catalog.get("parent_generation") != parent:
        raise ValueError("catalog parent mismatch")
    generation = text.digest(catalog)
    catalog["generation"] = generation
    validate_catalog(catalog)
    raw = encode(catalog)
    catalog_ref = metadata(ASSETS, f"reader-index/v3/generations/{generation}/catalog.json", raw)
    new_pointer = {"version": 1, "kind": "reader-reading-pointer", "generation": generation,
                   "catalog": catalog_ref, "resources": [catalog_ref]}
    report = {"generation": generation, "parent_generation": parent, "applied": apply,
              "pointer": new_pointer, "files": len(catalog["files"])}
    if not apply:
        return report
    assert_writer(store)
    # Central Actions owns the serialized writer. Hub has no CAS operation.
    latest, _ = current(store)
    if latest != pointer:
        raise ValueError("reading pointer changed during verification")
    immutable_put(store, ASSETS, catalog_ref["path"], raw)
    history = {"version": 1, "kind": "reader-reading-promotion", "generation": generation,
               "parent_generation": parent, "resources": [catalog_ref],
               "acks": {"hf": False, "pages": False}, "retention_days": 30,
               "created_at": reader_lifecycle.now_iso()}
    history_path = f"reader-index/v3/promotions/{generation}.json"
    try:
        prior_history = json.loads(store.read_bytes(ASSETS, history_path))
        if prior_history.get("generation") != generation:
            raise ValueError("promotion record identity mismatch")
    except FileNotFoundError:
        store.put_bytes(ASSETS, history_path, encode(history))
    if current(store)[0] != pointer:
        raise ValueError("reading pointer changed before promotion")
    store.put_bytes(ASSETS, POINTER, encode(new_pointer))
    if current(store)[0] != new_pointer:
        raise ValueError("reading pointer verification failed")
    return report


def rollback(store, target_ref, expected_parent, *, apply=False):
    target = validate_catalog(read_index(store, target_ref))
    pointer, _ = current(store)
    if not pointer or pointer["generation"] != expected_parent:
        raise ValueError("rollback requires the current expected parent")
    for key, entry in target["files"].items():
        manifest = v3.verify_reading(entry["resource"], lambda ref: store.read_bytes(ref["bucket"], ref["path"]))
        if manifest["source_key"] != key or manifest["generation"] != entry["reading_generation"]:
            raise ValueError("rollback resource identity mismatch")
    catalog = {"version": 3, "kind": "reader-reading-catalog", "files": target["files"],
               "parent_generation": expected_parent, "rollback_target": target["generation"]}
    return commit_catalog(store, catalog, pointer, apply=apply)


def withdraw(store, options, expected_parent, *, apply=False):
    """Remove only one defective v3 override; retain the source and all history."""
    key = options.get("source_key")
    generation = text.sha(options.get("reading_generation"))
    if not isinstance(key, str) or not key:
        raise ValueError("withdrawal requires an exact source key")
    pointer, catalog = current(store)
    active = catalog["files"].get(key)
    if active is None:
        report = {"applied": apply, "unchanged": True}
    else:
        if not pointer or pointer["generation"] != expected_parent or active["reading_generation"] != generation:
            raise ValueError("stale withdrawal generation")
        actor = os.environ.get("GITHUB_ACTOR", "").strip()
        if not actor:
            raise ValueError("authenticated withdrawal actor required")
        files = {k: entry for k, entry in catalog["files"].items() if k != key}
        report = commit_catalog(store, {"version": 3, "kind": "reader-reading-catalog", "files": files,
            "parent_generation": expected_parent, "withdrawal": {"source_key": key,
            "reading_generation": generation, "resource": active["resource"], "actor": actor,
            "reason": "source-conversion-needs-review"}}, pointer, apply=apply)
    if apply:
        try:
            from .reader_v3_automation import load_state, save_state
        except ImportError:
            from reader_v3_automation import load_state, save_state
        state = load_state(store)
        for task in state["tasks"].values():
            if task["spec"]["source_key"] == key:
                task.update(status="needs-review", error_type="PublicationReviewRequired")
        save_state(store, state)
    return report


def acknowledge(store, surface, receipt, *, apply=False):
    if surface not in {"hf", "pages"}:
        raise ValueError("invalid consumer surface")
    pointer, catalog = current(store)
    if (not pointer or receipt.get("version") != 1 or receipt.get("surface") != surface
            or receipt.get("active") is not True or receipt.get("generation") != pointer["generation"]
            or receipt.get("catalog_sha256") != pointer["catalog"]["sha256"]
            or receipt.get("files") != len(catalog["files"])):
        raise ValueError("consumer did not observe current reading generation")
    path = f"reader-index/v3/promotions/{pointer['generation']}.json"
    history = json.loads(store.read_bytes(ASSETS, path))
    if history.get("generation") != pointer["generation"]:
        raise ValueError("consumer acknowledgment promotion mismatch")
    updated = copy.deepcopy(history)
    updated["acks"][surface] = True
    updated.setdefault("receipts", {})[surface] = copy.deepcopy(receipt)
    if apply:
        assert_writer(store)
        if receipt.get("projection_verified") is not True or not receipt.get("checked_url"):
            raise ValueError("consumer acknowledgment requires read-only projection acceptance")
        if current(store)[0] != pointer:
            raise ValueError("consumer receipt became stale")
        store.put_bytes(ASSETS, path, encode(updated))
    return {"generation": pointer["generation"], "surface": surface, "applied": apply,
            "acks": updated["acks"]}


def observe_consumer(store, surface, client):
    """Read deployed status AND projection; a successful upload is not a receipt."""
    import gzip
    pointer, catalog = current(store)
    if not pointer:
        raise ValueError("no v3 generation to accept")
    endpoints = {
        "hf": ("https://voiceofml-search.hf.space/api/reader-v3-status",
               "https://voiceofml-search.hf.space/api/reader-assets"),
        "pages": ("https://vomebook.github.io/search/data/reader_v3_receipt.json",
                  "https://vomebook.github.io/search/data/reader_assets.json.gz"),
    }
    if surface not in endpoints:
        raise ValueError("invalid consumer surface")
    status_url, sidecar_url = endpoints[surface]
    def fetch(url, limit):
        with client.stream("GET", url, headers={"Cache-Control": "no-cache"}) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > limit:
                    raise ValueError("consumer acceptance response too large")
                chunks.append(chunk)
            return b"".join(chunks)
    receipt = json.loads(fetch(status_url, 1024 * 1024))
    raw = fetch(sidecar_url, 64 * 1024 * 1024)
    if surface == "pages":
        if (receipt.get("reader_contract") != "pdf-reading-v3-v1"
                or hashlib.sha256(raw).hexdigest() != receipt.get("sidecar_sha256")):
            raise ValueError("Pages deployment receipt/projection mismatch")
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            expanded = stream.read(64 * 1024 * 1024 + 1)
        if len(expanded) > 64 * 1024 * 1024:
            raise ValueError("consumer sidecar expands beyond limit")
        sidecar = json.loads(expanded)
    else:
        sidecar = json.loads(raw)
    if sidecar.get("v") != 1 or not isinstance(sidecar.get("f"), dict):
        raise ValueError("invalid deployed consumer sidecar")
    for key, entry in catalog["files"].items():
        expected = {"s": 2, "m": "p", "p": entry["resource"]["path"], "b": shared.PDF_PAGES_BUCKET}
        if sidecar["f"].get(key) != expected:
            raise ValueError("deployed consumer resolves a different reading manifest")
    acknowledge(store, surface, receipt)
    return {**receipt, "checked_url": status_url, "checked_at": reader_lifecycle.now_iso(),
            "projection_verified": True}


def project_sidecar(sidecar, catalog):
    validate_catalog(catalog)
    if sidecar.get("v") != 1 or not isinstance(sidecar.get("f"), dict):
        raise ValueError("invalid legacy sidecar")
    result = copy.deepcopy(sidecar)
    for key, entry in catalog["files"].items():
        result["f"][key] = {"s": 2, "m": "p", "p": entry["resource"]["path"],
                            "b": shared.PDF_PAGES_BUCKET}
    raw = encode(catalog)
    result["v3"] = {"generation": catalog["generation"], "catalog_sha256": hashlib.sha256(raw).hexdigest(),
                    "files": len(catalog["files"])}
    return result


class CentralHubStore(HubBucketStore):
    def assert_serialized_writer(self, protocol):
        try:
            from .pdf_worker_lanes import load_config
        except ImportError:
            from pdf_worker_lanes import load_config
        expected = load_config()["publisher_repository"]
        if (protocol != PROTOCOL or os.environ.get("GITHUB_ACTIONS") != "true"
                or os.environ.get("GITHUB_REPOSITORY") != expected
                or not os.environ.get("GITHUB_RUN_ID")
                or os.environ.get("GITHUB_WORKFLOW") != "Publish v3 Reading Generation"
                or os.environ.get("GITHUB_WORKFLOW_REF") != expected + "/.github/workflows/reader-v3-publish.yml@refs/heads/main"
                or os.environ.get("READER_V3_WRITE_PROTOCOL") != PROTOCOL):
            raise RuntimeError("v3 mutation requires the central serialized workflow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-stage", "stage", "promote", "rollback", "withdraw", "ack", "ack-all", "inspect", "project", "auto", "correct", "accept", "reject", "retry"))
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--resource-json")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--expected-parent", default="none")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--surface", choices=("hf", "pages"))
    args = parser.parse_args()
    store = CentralHubStore()
    if args.command == "withdraw":
        report = withdraw(store, json.loads(args.resource_json or "{}"),
                          None if args.expected_parent == "none" else text.sha(args.expected_parent), apply=args.apply)
    elif args.command == "retry":
        try:
            from .reader_v3_automation import retry
        except ImportError:
            from reader_v3_automation import retry
        report = retry(store, json.loads(args.resource_json or "{}"), apply=args.apply)
    elif args.command == "auto":
        try:
            from .reader_v3_automation import run
        except ImportError:
            from reader_v3_automation import run
        options = json.loads(args.resource_json or "{}")
        priority = options.get("source_sha256")
        if priority is not None:
            text.sha(priority)
        report = run(store, apply=args.apply, priority_source=priority)
    elif args.command in {"correct", "accept", "reject"}:
        try:
            from .reader_v3_correction import operate
        except ImportError:
            from reader_v3_correction import operate
        report = operate(store, args.command, json.loads(args.resource_json or "{}"), apply=args.apply)
    elif args.command == "build-stage":
        if not args.resource and not args.resource_json:
            parser.error("build-stage requires a pinned single-book spec JSON")
        spec = json.loads(args.resource_json if args.resource_json else args.resource.read_text())
        with tempfile.TemporaryDirectory(prefix="reader-v3-candidate-") as directory:
            report = build_stage(store, spec, Path(directory), apply=args.apply)
    elif args.command == "ack-all":
        import httpx
        import time
        pending = {"hf", "pages"}
        report = {"applied": args.apply, "consumers": {}}
        deadline = time.monotonic() + 1200
        if current(store)[0] is not None:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                while pending:
                    for surface in sorted(pending):
                        try:
                            receipt = observe_consumer(store, surface, client)
                            report["consumers"][surface] = acknowledge(store, surface, receipt, apply=args.apply)
                            pending.remove(surface)
                        except (ValueError, httpx.HTTPError):
                            if time.monotonic() >= deadline:
                                raise RuntimeError("consumer projection acceptance deadline exceeded") from None
                    if pending:
                        time.sleep(20)
    elif args.command == "ack":
        if not args.surface:
            parser.error("ack requires --surface")
        import httpx
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            receipt = observe_consumer(store, args.surface, client)
        report = acknowledge(store, args.surface, receipt, apply=args.apply)
    elif args.command in {"stage", "promote", "rollback"}:
        if not args.resource and not args.resource_json:
            parser.error("stage/promote require a qualified --resource JSON")
        ref = json.loads(args.resource_json if args.resource_json else args.resource.read_text())
        if args.command == "stage":
            report = stage(store, ref, args.bundle, apply=args.apply)
        else:
            operation = promote if args.command == "promote" else rollback
            report = operation(store, ref, None if args.expected_parent == "none" else text.sha(args.expected_parent), apply=args.apply)
    elif args.command == "inspect":
        pointer, catalog = current(store)
        report = {"pointer": pointer, "files": len(catalog["files"])}
    else:
        try:
            from .reader_bucket import read_bytes
        except ImportError:
            from reader_bucket import read_bytes
        import gzip
        pointer, catalog = current(store)
        base = json.loads(gzip.decompress(read_bytes("reader-index/reader_assets.json.gz", os.environ.get("HF_TOKEN"))))
        report = project_sidecar(base, catalog) if pointer else base
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode(report))
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
