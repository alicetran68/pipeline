"""Independent, versioned Reader text layers and explicit correction acceptance."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata

VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA-256")
    return value


def number(value, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError("invalid geometry number")
    return value


def box(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(number(x) > 1 for x in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise ValueError("invalid normalized text box")
    return value


def quad(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(not isinstance(point, list) or len(point) != 2
                   or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 or n > 1
                          for n in point) for point in value)):
        raise ValueError("invalid normalized text quadrilateral")
    area = sum(value[i][0] * value[(i + 1) % 4][1]
               - value[(i + 1) % 4][0] * value[i][1] for i in range(4)) / 2
    if abs(area) <= 1e-9:
        raise ValueError("degenerate normalized text quadrilateral")
    return value


def direction(text):
    strong = {unicodedata.bidirectional(c) for c in text} & {"L", "R", "AL"}
    if "L" in strong and strong & {"R", "AL"}:
        return "mixed"
    if strong & {"R", "AL"}:
        return "rtl"
    return "ltr" if "L" in strong else "neutral"


def scripts(text):
    result = set()
    for char in text:
        name = unicodedata.name(char, "")
        for prefix, script in (("CJK", "Hani"), ("HIRAGANA", "Hira"), ("KATAKANA", "Kana"),
                               ("HANGUL", "Hang"), ("ARABIC", "Arab"), ("HEBREW", "Hebr"),
                               ("CYRILLIC", "Cyrl"), ("GREEK", "Grek"), ("LATIN", "Latn"),
                               ("DEVANAGARI", "Deva"), ("THAI", "Thai")):
            if name.startswith(prefix):
                result.add(script)
                break
    return sorted(result)


def from_page(payload, source_sha256, language="und", raw_resource=None, *, layout_options=None):
    """Retain engine text verbatim; expose only offsets actually matching it."""
    sha(source_sha256)
    if payload.get("kind") != "pdf-ocr-page" or payload.get("source") not in {"native", "ocr"}:
        raise ValueError("invalid raw page")
    page = payload.get("page")
    if type(page) is not int or page < 1:
        raise ValueError("invalid page number")
    width, height = number(payload["width"], True), number(payload["height"], True)
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("invalid page text")
    layout = payload.get("layout", {})
    mode = layout.get("writing_mode", "auto")
    flags = set(layout.get("review", []))
    options = dict(layout_options or {})
    if options:
        try:
            from . import ocr_layout
        except ImportError:
            import ocr_layout
        options = ocr_layout.validate_options(options)
    regions = []
    blocks = {b.get("id", index): b for index, b in enumerate(payload.get("blocks", []))}
    spans = payload.get("text_spans", [])
    if text and not spans:
        flags.add("text-without-positioned-spans")
    cursor = 0
    for index, span in enumerate(spans):
        start, end = span.get("start"), span.get("end")
        block = blocks.get(span.get("block"))
        if (type(start) is not int or type(end) is not int or not cursor <= start < end <= len(text)
                or block is None or text[start:end] != block.get("t")):
            flags.add("text-span-mismatch")
            continue
        try:
            bounds = box(list(span["box"]))
            polygon = block.get("q") or [[bounds[0], bounds[1]], [bounds[2], bounds[1]],
                                         [bounds[2], bounds[3]], [bounds[0], bounds[3]]]
            if (len(polygon) != 4 or any(len(p) != 2 for p in polygon)
                    or any(number(n) > 1 for p in polygon for n in p)):
                raise ValueError("invalid text quadrilateral")
            confidence = number(block.get("c", 0))
            if confidence > 1:
                raise ValueError("invalid confidence")
        except (ValueError, KeyError, TypeError):
            flags.add("invalid-text-geometry")
            continue
        content = text[start:end]
        bidi = direction(content)
        detected_scripts = scripts(content)
        if bidi in {"rtl", "mixed"}:
            flags.add("bidi-order-needs-review")
        if payload["source"] == "ocr" and confidence < .90:
            flags.add("low-recognition-confidence")
        if payload["source"] == "ocr":
            supported = ({"Hani", "Latn"} if language in {"ch", "chinese_cht"} else
                         {"Latn"} if language == "en" else
                         {"Hani", "Hira", "Kana", "Latn"} if language == "japan" else None)
            if supported is not None and set(detected_scripts) - supported:
                flags.add("recognition-language-script-mismatch")
        region_mode = span.get("writing_mode", mode)
        if "writing_mode" not in span and options:
            region_mode = options.get("writing_mode", mode)
            region_mode = mode if region_mode == "auto" else region_mode
            center_x, center_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            for override in options.get("regions", []):
                original_box = ocr_layout.map_box_back(override["box"], options.get("rotation", 0))
                if original_box[0] <= center_x <= original_box[2] and original_box[1] <= center_y <= original_box[3]:
                    region_mode = override.get("writing_mode", region_mode)
                    if region_mode == "auto":
                        region_mode = mode
                        flags.add("region-writing-mode-unverified")
                    break
        if region_mode.startswith("vertical"):
            flags.add("vertical-recognition-needs-sample-validation")
        regions.append({"id": f"b{span['block']}", "order": len(regions), "start": start, "end": end,
                        "text": content, "box": bounds, "quad": copy.deepcopy(polygon),
                        "mapping_precision": "block", "source": payload["source"],
                        "confidence": confidence, "language_hint": language,
                        "scripts": detected_scripts, "direction": bidi,
                        "writing_mode": region_mode, "layout_region": span.get("region", index)})
        cursor = end
    if payload["source"] == "ocr" and not text.strip():
        # An empty detector output cannot establish that the source page is blank.
        flags.add("empty-recognition-unverified")
    geometry = {"width": width, "height": height, "coordinate_space": "normalized-page",
                "ocr_rotation_clockwise": layout.get("rotation_clockwise", 0)}
    result = {"version": VERSION, "kind": "pdf-text-layer", "source_sha256": source_sha256,
              "page": page, "page_identity": digest({"source": source_sha256, "page": page,
                                                     "geometry": geometry}),
              "geometry": geometry, "offset_unit": "unicode-codepoint", "text": text,
              "regions": regions, "raw_sha256": digest(payload), "revision": "raw",
              "processing": "processed" if text.strip() else "processed-empty",
              "quality": "unreviewed", "review_flags": sorted(flags),
              "provenance": {"source": payload["source"], "language_hint": language}}
    if raw_resource is not None:
        result["provenance"]["raw_resource"] = copy.deepcopy(raw_resource)
    result["generation"] = digest(result)
    validate(result)
    return result


def validate(layer):
    if layer.get("version") != VERSION or layer.get("kind") != "pdf-text-layer":
        raise ValueError("invalid text layer schema")
    for field in ("source_sha256", "page_identity", "raw_sha256", "generation"):
        sha(layer.get(field))
    if type(layer.get("page")) is not int or layer["page"] < 1:
        raise ValueError("invalid text layer page")
    if layer.get("offset_unit") != "unicode-codepoint" or not isinstance(layer.get("text"), str):
        raise ValueError("invalid text offset contract")
    if (layer.get("revision") not in {"raw", "accepted"}
            or layer.get("quality") not in {"unreviewed", "partially-reviewed", "accepted"}
            or layer.get("processing") not in {"processed", "processed-empty"}):
        raise ValueError("invalid text layer state")
    if layer["revision"] == "raw" and layer["quality"] != "unreviewed":
        raise ValueError("raw text is not reviewed text")
    if layer["revision"] == "accepted":
        sha(layer.get("parent_generation"))
        if layer["quality"] != "partially-reviewed" or not layer.get("acceptances"):
            raise ValueError("accepted text lacks explicit review audit")
        for audit in layer["acceptances"]:
            sha(audit.get("proposal_sha256"))
            if not isinstance(audit.get("actor"), str) or not audit["actor"].strip():
                raise ValueError("invalid correction audit actor")
    geometry = layer["geometry"]
    number(geometry["width"], True)
    number(geometry["height"], True)
    if geometry.get("coordinate_space") != "normalized-page":
        raise ValueError("unsupported coordinate space")
    if geometry.get("ocr_rotation_clockwise") not in (0, 90, 180, 270):
        raise ValueError("invalid OCR rotation")
    if layer["page_identity"] != digest({"source": layer["source_sha256"], "page": layer["page"],
                                         "geometry": geometry}):
        raise ValueError("page identity mismatch")
    previous, ids = 0, set()
    for index, region in enumerate(layer["regions"]):
        start, end = region["start"], region["end"]
        if (type(start) is not int or type(end) is not int
                or not previous <= start < end <= len(layer["text"])
                or layer["text"][start:end] != region["text"]
                or region["id"] in ids or region["order"] != index):
            raise ValueError("invalid text region offsets or identity")
        box(region["box"])
        quad(region["quad"])
        if (region["direction"] not in {"ltr", "rtl", "mixed", "neutral"}
                or region["writing_mode"] not in {"auto", "horizontal-ltr", "horizontal-rtl",
                                                   "vertical-rl", "vertical-lr"}
                or region["mapping_precision"] not in {"block", "corrected-block"}):
            raise ValueError("invalid region writing contract")
        if (number(region["confidence"]) > 1 or region["source"] not in {"native", "ocr"}
                or not isinstance(region.get("language_hint"), str)):
            raise ValueError("invalid region recognition provenance")
        previous = end
        ids.add(region["id"])
    content = {k: v for k, v in layer.items() if k != "generation"}
    if digest(content) != layer["generation"]:
        raise ValueError("text generation mismatch")
    return layer


def review_task(layer, text_resource, evidence=None):
    validate(layer)
    if not layer["review_flags"]:
        return None
    body = {"source_sha256": layer["source_sha256"], "page": layer["page"],
            "page_identity": layer["page_identity"], "raw_sha256": layer["raw_sha256"],
            "base_generation": layer["generation"], "issues": layer["review_flags"],
            "region_ids": [r["id"] for r in layer["regions"]], "text_layer": text_resource,
            "evidence": evidence or [], "status": "pending"}
    return {"id": digest(body), **body}


def accept_proposal(layer, proposal, *, actor):
    """Explicit trusted acceptance; proposals never mutate their raw baseline."""
    validate(layer)
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("acceptance actor required")
    if (proposal.get("kind") != "pdf-text-correction" or proposal.get("version") != 1
            or proposal.get("base_generation") != layer["generation"]
            or proposal.get("raw_sha256") != layer["raw_sha256"]
            or proposal.get("page_identity") != layer["page_identity"]):
        raise ValueError("stale or mismatched correction")
    replacements = proposal.get("replacements")
    requested_order = proposal.get("region_order", [])
    original_order = [r["id"] for r in layer["regions"]]
    if (not isinstance(requested_order, list) or any(not isinstance(i, str) for i in requested_order)
            or requested_order and (len(requested_order) != len(original_order) or set(requested_order) != set(original_order))):
        raise ValueError("correction order must name every region exactly once")
    reorder = bool(requested_order and requested_order != original_order)
    if not isinstance(replacements, list) or not replacements and not reorder:
        raise ValueError("empty correction proposal")
    known = {r["id"]: r for r in layer["regions"]}
    changes = {}
    for item in replacements:
        target = item.get("region_id")
        if (target not in known or target in changes or item.get("before") != known[target]["text"]
                or not isinstance(item.get("after"), str) or not item["after"]
                or len(item["after"]) > 10000):
            raise ValueError("invalid correction region or text")
        changes[target] = item["after"]
    result = copy.deepcopy(layer)
    if reorder:
        cursor = 0
        for region in layer["regions"]:
            if layer["text"][cursor:region["start"]].strip():
                raise ValueError("cannot reorder text with unpositioned content")
            cursor = region["end"]
        if layer["text"][cursor:].strip():
            raise ValueError("cannot reorder text with unpositioned content")
        by_id = {r["id"]: r for r in result["regions"]}
        result["regions"] = [by_id[identity] for identity in requested_order]
    parts, position, length = [], 0, 0
    for index, region in enumerate(result["regions"]):
        old_start, old_end = region["start"], region["end"]
        gap = ("\n" if index else "") if reorder else layer["text"][position:old_start]
        parts.append(gap)
        length += len(gap)
        if region["id"] in changes:
            region["text"] = changes[region["id"]]
            region["mapping_precision"] = "corrected-block"
            region["scripts"] = scripts(region["text"])
            region["direction"] = direction(region["text"])
        region["start"] = length
        region["order"] = index
        length += len(region["text"])
        region["end"] = length
        parts.append(region["text"])
        position = old_end
    if not reorder:
        parts.append(layer["text"][position:])
    result["text"] = "".join(parts)
    result.update(revision="accepted", quality="partially-reviewed", parent_generation=layer["generation"])
    result.setdefault("acceptances", []).append({"proposal_sha256": digest(proposal), "actor": actor,
                                                "region_ids": sorted(changes),
                                                **({"region_order": requested_order} if reorder else {})})
    result["generation"] = digest({k: v for k, v in result.items() if k != "generation"})
    return validate(result)
