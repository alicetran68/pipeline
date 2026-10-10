#!/usr/bin/env python3
"""Durable, budgeted OCR admission under the existing central v3 writer lock."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit

try:
    from . import publish_reader_v3 as publication, pdf_text_layer as text, shared
except ImportError:
    import publish_reader_v3 as publication
    import pdf_text_layer as text
    import shared

STATE = "reader-index/v3/automation.json"
REGISTRY = "reader-index/pdf_ocr_manifest.json"
DAILY_ATTEMPTS = 2
DAILY_PAGES = 1000
MAX_ATTEMPTS = 3


def clock():
    return datetime.now(timezone.utc)


def load_state(store):
    try:
        state = json.loads(store.read_bytes(publication.ASSETS, STATE))
    except FileNotFoundError:
        return {"version": 1, "kind": "reader-v3-automation", "tasks": {}, "days": {}}
    if (state.get("version") != 1 or state.get("kind") != "reader-v3-automation"
            or not isinstance(state.get("tasks"), dict) or not isinstance(state.get("days"), dict)):
        raise ValueError("invalid v3 automation state")
    return state


def save_state(store, state):
    publication.assert_writer(store)
    raw = publication.encode(state)
    store.put_bytes(publication.ASSETS, STATE, raw)
    if store.read_bytes(publication.ASSETS, STATE) != raw:
        raise ValueError("automation checkpoint readback mismatch")


def retry(store, options, *, apply=False):
    """Explicit operator retry preserves history and the already charged day budget."""
    queue = options.get("queue", "build")
    if queue == "correction":
        try:
            from . import reader_v3_correction as correction
        except ImportError:
            import reader_v3_correction as correction
        state, persist = correction.load_state(store), correction.save
    elif queue == "build":
        state, persist = load_state(store), save_state
    else:
        raise ValueError("unknown retry queue")
    ids = options.get("task_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("explicit retry task IDs required")
    import os
    actor = os.environ.get("GITHUB_ACTOR", "").strip()
    if not actor:
        raise ValueError("authenticated retry actor required")
    tasks = [state["tasks"][text.sha(identity)] for identity in ids]
    if any(t["status"] not in {"failed", "retry"} for t in tasks):
        raise ValueError("only failed/retry tasks may be retried")
    if apply:
        publication.assert_writer(store)
        for task in tasks:
            task.setdefault("retries", []).append({"actor": actor, "at": clock().isoformat(),
                "attempts": task["attempts"], "failures": task.get("failures", 0), "error_type": task.get("error_type")})
            task.update(status="pending", attempts=0, failures=0)
            for name in ("retry_at", "retry_day", "error_type"):
                task.pop(name, None)
        persist(store, state)
    return {"applied": apply, "queue": queue, "retried": ids}


def spec_for(key, entry):
    """Admit only pinned originals or checksum-identified current-bucket PDFs."""
    source = text.sha(entry.get("source_sha256"))
    size, count = entry.get("source_bytes"), entry.get("page_count")
    if (entry.get("status") != "ready" or type(count) is not int or count < 1
            or type(size) is not int or not 1 <= size <= 512 * 1024 * 1024):
        raise ValueError("OCR entry lacks bounded complete source metadata")
    ocr = {"bucket": shared.PDF_PAGES_BUCKET, "path": entry["ocr_manifest"],
           "sha256": entry["ocr_manifest_sha256"], "bytes": entry["ocr_manifest_bytes"], "role": "provenance"}
    publication.v3.resource(ocr)
    if not re.fullmatch(r"objects/" + source[:2] + "/" + source + r"/[0-9a-f]{16}/ocr-manifest\.json", ocr["path"]):
        raise ValueError("OCR resource/source mismatch")
    spec = {"source_key": key, "source_sha256": source, "ocr_manifest": ocr}
    if entry.get("source_kind", "upstream") == "upstream":
        origin = {"repo": entry.get("repo"), "path": entry.get("path"), "revision": entry.get("source_revision")}
        if (not isinstance(origin["repo"], str) or not re.fullmatch(r"VoiceOfML/[A-Za-z0-9._-]+", origin["repo"])
                or not isinstance(origin["path"], str) or not origin["path"].lower().endswith(".pdf")
                or any(p in {"", ".", ".."} for p in origin["path"].split("/"))
                or "\\" in origin["path"] or any(ord(c) < 32 for c in origin["path"])
                or not isinstance(origin["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", origin["revision"])
                or key != origin["repo"] + "\0" + origin["path"]):
            raise ValueError("original source is not pinned")
        root = f"objects/{source[:2]}/{source}/{text.digest({'primary': 1})[:16]}"
        spec.update(primary_source=origin, primary={"bucket": shared.PDF_PAGES_BUCKET,
                    "path": root + "/document.pdf", "sha256": source, "bytes": size, "role": "runtime"})
    else:
        url = urlsplit(entry.get("source_url", ""))
        match = re.fullmatch(r"/buckets/(vomebook/(?:pdf-pages-v2|reader-assets-v2))/resolve/(.+)", url.path)
        if not entry.get("source_url"):
            bucket, path = entry.get("reader_assets_bucket"), entry.get("reader_assets_path")
        elif url.scheme == "https" and url.netloc == "huggingface.co" and not url.query and not url.fragment and match:
            bucket, path = match[1], unquote(match[2])
        else:
            raise ValueError("derived primary is not a current-bucket resource")
        primary = {"bucket": bucket, "path": path, "sha256": source, "bytes": size, "role": "runtime"}
        publication.v3.resource(primary)
        if bucket not in {shared.PDF_PAGES_BUCKET, shared.READER_ASSETS_BUCKET} or not (publication.PUBLIC_OBJECT.fullmatch(primary["path"]) or re.fullmatch(
                r"derived/[A-Za-z0-9._-]+/[a-z0-9]{32}/document\.pdf", primary["path"]) or re.fullmatch(
                r"documents/pdf/[a-z0-9_-]+/[0-9a-f]{64}/document\.pdf", primary["path"])):
            raise ValueError("derived primary path is not published")
        spec["primary"] = primary
        spec["require_usable_text"] = True
    return spec, count


def discover(state, registry):
    if registry.get("version") != 1 or not isinstance(registry.get("files"), dict):
        raise ValueError("invalid OCR registry")
    ignored = {}
    for key, entry in sorted(registry["files"].items()):
        if entry.get("status") != "ready":
            continue
        try:
            spec, count = spec_for(key, entry)
        except (ValueError, KeyError, TypeError) as error:
            ignored[key] = type(error).__name__
            continue
        identity = text.digest(spec)
        state["tasks"].setdefault(identity, {"spec": spec, "page_count": count, "status": "pending", "attempts": 0})
    return ignored


def run(store, *, apply=False, priority_source=None, build=None, now=None):
    if apply:
        publication.assert_writer(store)
    state = copy.deepcopy(load_state(store))
    ignored = discover(state, json.loads(store.read_bytes(publication.ASSETS, REGISTRY)))
    now = now or clock()
    day = now.date().isoformat()
    budget = state["days"].setdefault(day, {"attempts": 0, "pages": 0})
    _, catalog = publication.current(store)
    eligible = []
    for identity, task in state["tasks"].items():
        spec = task["spec"]
        active = catalog["files"].get(spec["source_key"])
        if active:
            # A different raw recipe must never replace accepted or already published text.
            task["status"] = "published" if active["source_sha256"] == spec["source_sha256"] else "protected"
            task["active_generation"] = active["reading_generation"]
            continue
        if task["status"] in {"published", "protected", "failed", "needs-review"}:
            continue
        if not task.get("candidate") and task["attempts"] >= MAX_ATTEMPTS:
            task["status"] = "failed"
            continue
        if task.get("retry_at") and task["retry_at"] > now.isoformat():
            continue
        if task.get("candidate"):
            eligible.append((identity, task))
        elif (task["attempts"] < MAX_ATTEMPTS and budget["attempts"] < DAILY_ATTEMPTS
              and budget["pages"] + task["page_count"] <= DAILY_PAGES):
            eligible.append((identity, task))
    eligible.sort(key=lambda pair: (pair[1]["spec"]["source_sha256"] != priority_source,
                                    not bool(pair[1].get("candidate")), pair[1]["page_count"], pair[0]))
    report = {"applied": apply, "tasks": len(state["tasks"]), "ignored": ignored,
              "eligible": len(eligible), "budget": budget, "processed": []}
    if not apply:
        report["next"] = eligible[0][0] if eligible else None
        return report
    save_state(store, state)
    if not eligible:
        return report
    identity, task = eligible[0]
    try:
        if not task.get("candidate"):
            task.update(status="building", attempts=task["attempts"] + 1, started_at=now.isoformat())
            budget["attempts"] += 1
            budget["pages"] += task["page_count"]
            save_state(store, state)  # Charge before work, including interrupted attempts.
            with tempfile.TemporaryDirectory(prefix="reader-v3-auto-") as directory:
                built = (build or publication.build_stage)(store, task["spec"], Path(directory), apply=True)
            task.update(status="staged", candidate=built["candidate"])
            save_state(store, state)
        pointer, latest = publication.current(store)
        if task["spec"]["source_key"] in latest["files"]:
            task["status"] = "protected"
        else:
            promoted = publication.promote(store, task["candidate"], pointer["generation"] if pointer else None, apply=True)
            task.update(status="published", catalog_generation=promoted["generation"])
        task.pop("retry_at", None)
        task.pop("error_type", None)
    except Exception as error:
        task["error_type"] = type(error).__name__
        task["failures"] = task.get("failures", 0) + 1
        task["status"] = ("needs-review" if isinstance(error, publication.PublicationReviewRequired) else
                          "failed" if task["failures"] >= MAX_ATTEMPTS else "retry")
        task["retry_at"] = (now + timedelta(hours=2 ** min(task["failures"], 5))).isoformat()
    save_state(store, state)
    report["processed"].append({"id": identity, "status": task["status"], "error_type": task.get("error_type")})
    return report
