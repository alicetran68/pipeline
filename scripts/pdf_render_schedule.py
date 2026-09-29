"""Time-weighted render scheduling, independent of image and OCR formats."""

import math

try:
    from . import shared
except ImportError:
    import shared


DEFAULT_PAGE_SECONDS = 3.0
MAX_RENDER_SHARDS = 256


def finite_seconds(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        finite = math.isfinite(value)
    except OverflowError:
        return None
    if not finite or value < 0 or (positive and value == 0):
        return None
    return float(value)


def parse_range(key, page_count):
    try:
        first, last = key.split("-")
        start, end = int(first), int(last)
    except (AttributeError, TypeError, ValueError):
        return None
    if key != f"{start:06d}-{end:06d}" or not 1 <= start <= end <= page_count:
        return None
    return start, end


def history_cost(identity, progress, image_rendered=None):
    if not isinstance(progress, dict) or any(progress.get(k) != v for k, v in identity.items()):
        return None
    samples = []
    timings = progress.get("range_timings", {})
    ranges = progress.get("ranges", {})
    if not isinstance(timings, dict) or not isinstance(ranges, dict):
        return None
    for key, timing in timings.items():
        bounds = parse_range(key, identity["page_count"])
        if bounds is None or not isinstance(timing, dict) or key not in ranges:
            continue
        if image_rendered is not None and timing.get("image_rendered") != image_rendered:
            continue
        count = bounds[1] - bounds[0] + 1
        seconds = finite_seconds(timing.get("page_seconds"), positive=True)
        setup = finite_seconds(timing.get("setup_seconds"))
        if seconds is not None and setup is not None and timing.get("page_count") == count:
            samples.append((seconds / count, setup))
    if not samples:
        return None
    # Upper quartiles damp fast-range bias without letting one outlier dominate.
    index = math.ceil(.75 * len(samples)) - 1
    return {"seconds_per_page": sorted(s[0] for s in samples)[index],
            "setup_seconds": sorted(s[1] for s in samples)[index],
            "source": "history", "samples": len(samples)}


def missing_ranges(page_count, completed, size):
    cursor = 1
    missing = []
    for start, end in sorted(completed):
        if start > cursor:
            missing.extend((n, min(n + size - 1, start - 1)) for n in range(cursor, start, size))
        cursor = max(cursor, end + 1)
    missing.extend((n, min(n + size - 1, page_count)) for n in range(cursor, page_count + 1, size))
    return missing


def range_size(cost, target_seconds, maximum):
    rate = finite_seconds(cost.get("seconds_per_page"), positive=True) or DEFAULT_PAGE_SECONDS
    setup = finite_seconds(cost.get("setup_seconds")) or 0
    return max(1, min(maximum, int(max(0, target_seconds / 2 - setup) / rate)))


def task_seconds(task):
    cost = task.get("_render_cost", {})
    rate = finite_seconds(cost.get("seconds_per_page"), positive=True) or DEFAULT_PAGE_SECONDS
    setup = finite_seconds(cost.get("setup_seconds")) or 0
    return (task["end"] - task["start"] + 1) * rate + setup


def balance(tasks, target_seconds):
    target = finite_seconds(target_seconds, positive=True)
    if target is None:
        raise ValueError("render target seconds must be finite and positive")
    if not tasks:
        return []
    count = min(MAX_RENDER_SHARDS, len(tasks), max(1, math.ceil(sum(map(task_seconds, tasks)) / target)))
    return shared.weighted_shards(tasks, count, weight=task_seconds,
                                  order=lambda t: (-task_seconds(t), t["key"], t["start"]))
