"""Conservative reading order and offset mapping for image OCR.

Geometry can establish column order, but cannot prove character order inside a
recognizer's string. Never reverse a recognized string to guess an old heading.
Uncertain layout keeps separators and records review flags instead.
"""

from __future__ import annotations

import copy
import re
import statistics

VERSION = "layout-v1"
MODES = {"auto", "horizontal-ltr", "horizontal-rtl", "vertical-rl", "vertical-lr"}


def validate_options(options):
    options = dict(options or {})
    if set(options) - {"writing_mode", "rotation", "join_soft_lines", "regions"}:
        raise ValueError("unknown OCR layout option")
    if options.get("writing_mode", "auto") not in MODES:
        raise ValueError("unsupported writing mode")
    if options.get("rotation", 0) not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180 or 270 clockwise")
    if not isinstance(options.get("join_soft_lines", True), bool):
        raise ValueError("join_soft_lines must be boolean")
    for region in options.get("regions", []):
        box = region.get("box", [])
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError("invalid layout region")
        if region.get("writing_mode", "auto") not in MODES:
            raise ValueError("invalid region writing mode")
    return options


def map_point_back(x, y, clockwise):
    if clockwise == 90:
        return y, 1 - x
    if clockwise == 180:
        return 1 - x, 1 - y
    if clockwise == 270:
        return 1 - y, x
    return x, y


def map_box_back(box, clockwise):
    points = [map_point_back(x, y, clockwise) for x, y in
              ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))]
    return [min(x for x, y in points), min(y for x, y in points),
            max(x for x, y in points), max(y for x, y in points)]


def physical_ratio(block, width, height):
    b = block["b"]
    return (b[2] - b[0]) * width / max(1e-9, (b[3] - b[1]) * height)


def infer_mode(blocks, width, height):
    substantial = [b for b in blocks if len(b["t"].strip()) >= 3]
    if substantial and sum(physical_ratio(b, width, height) < .6 for b in substantial) / len(substantial) >= .75:
        return "vertical-rl", "geometry"
    return "horizontal-ltr", "default"


def gap_split(blocks, axis, minimum):
    """Find a whitespace cut crossed by no text block."""
    ordered = sorted(blocks, key=lambda b: (b["b"][axis], b["id"]))
    end = ordered[0]["b"][axis + 2]
    best = None
    for index, block in enumerate(ordered[1:], 1):
        gap = block["b"][axis] - end
        if gap >= minimum and (best is None or gap > best[0]):
            best = (gap, index)
        end = max(end, block["b"][axis + 2])
    if best:
        return ordered[:best[1]], ordered[best[1]:]
    return None


def groups(blocks, mode, width, height, depth=0):
    if not blocks:
        return []
    vertical = mode.startswith("vertical")
    if vertical:
        # Group tall/single-character fragments sharing a column; do not read
        # a row across adjacent vertical columns.
        columns = []
        for block in sorted(blocks, key=lambda b: (b["b"][0], b["b"][1], b["id"])):
            box = block["b"]
            selected = None
            for column in columns:
                left, right = column[0]["b"][0], column[0]["b"][2]
                overlap = min(right, box[2]) - max(left, box[0])
                if overlap >= .5 * min(right - left, box[2] - box[0]):
                    selected = column
                    break
            if selected is None:
                columns.append([block])
            else:
                selected.append(block)
        columns.sort(key=lambda c: c[0]["b"][0], reverse=mode == "vertical-rl")
        return [sorted(c, key=lambda b: (b["b"][1], b["id"])) for c in columns]
    if len(blocks) > 1 and depth < 12:
        line_height = statistics.median(max(1e-6, b["b"][3] - b["b"][1]) for b in blocks)
        single_glyph_row = (all(len(b["t"]) == 1 for b in blocks)
                            and max(b["b"][1] for b in blocks) - min(b["b"][1] for b in blocks) < .5 * line_height)
        cut = None if single_glyph_row else gap_split(blocks, 0, max(.018, .8 * line_height * height / width))
        if cut:
            if mode == "horizontal-rtl":
                cut = cut[::-1]
            return [g for part in cut for g in groups(part, mode, width, height, depth + 1)]
        # Wide headings may bridge columns: split off a distinctly separated
        # horizontal band, then reconsider columns inside each band.
        cut = gap_split(blocks, 1, max(.012, .7 * line_height))
        if cut:
            return [g for part in cut for g in groups(part, mode, width, height, depth + 1)]
    rows = []
    for block in sorted(blocks, key=lambda b: (b["b"][1], b["b"][0], b["id"])):
        box = block["b"]
        if rows:
            anchor = rows[-1][0]["b"]
            overlap = min(anchor[3], box[3]) - max(anchor[1], box[1])
            if overlap >= .5 * min(anchor[3] - anchor[1], box[3] - box[1]):
                rows[-1].append(block)
                continue
        rows.append([block])
    return [[b for row in rows for b in sorted(row, key=lambda b: b["b"][0],
                                              reverse=mode == "horizontal-rtl")]]


def cjk(char):
    return bool(char and re.fullmatch(r"[\u3400-\u9fff\uf900-\ufaff]", char))


def soft_boundary(previous, current, group, mode):
    # Preserve punctuation and join only geometrically continuous CJK text.
    if not cjk(previous["t"][-1:]) or not cjk(current["t"][:1]):
        return False
    a, b = previous["b"], current["b"]
    if mode.startswith("vertical"):
        aw, bw = a[2] - a[0], b[2] - b[0]
        return (abs(a[0] - b[0]) < .25 * max(aw, bw)
                and 0 <= b[1] - a[3] <= .5 * max(aw, bw))
    if mode.startswith("horizontal") and len(previous["t"]) == len(current["t"]) == 1:
        h = max(a[3] - a[1], b[3] - b[1])
        gap = a[0] - b[2] if mode == "horizontal-rtl" else b[0] - a[2]
        return abs(a[1] - b[1]) < .25 * h and 0 <= gap <= 1.25 * h
    if mode != "horizontal-ltr" or len(previous["t"]) < 4 or len(current["t"]) < 4:
        return False
    ah, bh = a[3] - a[1], b[3] - b[1]
    left = min(x["b"][0] for x in group)
    right = max(x["b"][2] for x in group)
    return (0 <= b[1] - a[3] <= .6 * max(ah, bh)
            and abs(a[0] - left) <= .25 * ah and abs(b[0] - left) <= .25 * bh
            and abs(a[2] - right) <= .5 * ah
            and .75 <= ah / max(bh, 1e-9) <= 1.33)


def inline_boundary(previous, current, mode, width, height):
    if mode.startswith("vertical"):
        return None
    a, b = previous["b"], current["b"]
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    row_height = min(a[3] - a[1], b[3] - b[1])
    if overlap < .5 * row_height:
        return None
    gap = a[0] - b[2] if mode == "horizontal-rtl" else b[0] - a[2]
    if gap < -.25 * row_height * height / width or gap > 1.5 * row_height * height / width:
        return None
    left, right = previous["t"][-1], current["t"][0]
    if (cjk(left) or left in "，。！？；：、」』）") and cjk(right):
        return ""
    if cjk(left) and right in "，。！？；：、」』）":
        return ""
    return " "


def arrange(blocks, width, height, options=None, *, include_writing_modes=False):
    options = validate_options(options)
    original = copy.deepcopy(blocks)
    working = [{**copy.deepcopy(b), "id": index} for index, b in enumerate(blocks) if b.get("t")]
    mode = options.get("writing_mode", "auto")
    inferred, evidence = infer_mode(working, width, height)
    mode = inferred if mode == "auto" else mode
    evidence = evidence if options.get("writing_mode", "auto") == "auto" else "override"
    review = []
    if evidence == "default":
        review.append("reading-direction-assumed-ltr")
    if mode.startswith("vertical"):
        review.append("vertical-recognition-needs-sample-validation")
    if mode == "horizontal-rtl" and any(len(b["t"]) > 1 for b in working):
        review.append("within-block-character-order-unverified")
    partitions = []
    remaining = list(working)
    for region in options.get("regions", []):
        x0, y0, x1, y1 = region["box"]
        selected = [b for b in remaining if x0 <= (b["b"][0] + b["b"][2]) / 2 <= x1
                    and y0 <= (b["b"][1] + b["b"][3]) / 2 <= y1]
        remaining = [b for b in remaining if b not in selected]
        region_mode = region.get("writing_mode", mode)
        if region_mode == "auto":
            region_mode = infer_mode(selected, width, height)[0]
        partitions.append((selected, region_mode))
    if remaining:
        partitions.append((remaining, mode))
        if options.get("regions"):
            review.append("blocks-outside-explicit-regions")
    text, spans, ordered, boundaries = "", [], [], []
    group_index = 0
    for partition, partition_mode in partitions:
        for group in groups(partition, partition_mode, width, height):
            previous = None
            for block in group:
                if text:
                    inline = inline_boundary(previous, block, partition_mode, width, height) if previous else None
                    join = bool(inline is None and previous and options.get("join_soft_lines", True)
                                and soft_boundary(previous, block, group, partition_mode))
                    separator = inline if inline is not None else ("" if join else ("\n" if previous else "\n\n"))
                    boundaries.append({"offset": len(text), "kind": "inline" if inline is not None else
                                       ("soft-line" if join else ("line" if previous else "region")),
                                       "separator": separator})
                    text += separator
                start = len(text)
                text += block["t"]
                spans.append({"start": start, "end": len(text), "block": block["id"],
                               "box": block["b"], "precision": "block", "region": group_index,
                               **({"writing_mode": partition_mode} if include_writing_modes else {})})
                ordered.append(block)
                previous = block
            group_index += 1
    return {"text": text, "blocks": ordered, "raw_blocks": original, "text_spans": spans,
            "layout": {"version": VERSION, "writing_mode": mode, "evidence": evidence,
                       "review": review, "regions": group_index, "boundaries": boundaries,
                       "offset_unit": "unicode-codepoint", "mapping_precision": "block"}}


def restore_coordinates(payload, clockwise):
    if not clockwise:
        return payload
    for field in ("blocks", "raw_blocks"):
        for block in payload[field]:
            block["b"] = map_box_back(block["b"], clockwise)
            if "q" in block:
                block["q"] = [list(map_point_back(x, y, clockwise)) for x, y in block["q"]]
    for span in payload["text_spans"]:
        span["box"] = map_box_back(span["box"], clockwise)
    payload["layout"]["rotation_clockwise"] = clockwise
    return payload
