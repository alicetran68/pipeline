#!/usr/bin/env python3
"""Split the scanned Reader Assets queue into conversion shard files."""

import json
import os
from pathlib import Path

try:
    from . import scan_reader_assets
except ImportError:
    import scan_reader_assets


def prepare(queue_path: Path = Path("output/reader-assets/queue.json")) -> tuple[str, int, int, bool]:
    output_root = queue_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    items = queue["items"]
    extension = items[0]["extension"] if items else ""
    items = [item for item in items if item["extension"] == extension]
    queue["items"] = items
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")

    force_rebuild = os.environ.get("FORCE_REBUILD") == "true"
    scoped = bool(os.environ.get("INPUT_PATH") or os.environ.get("INPUT_REPO")
                  or os.environ.get("INPUT_EXTENSION") or os.environ.get("INPUT_PATH_PREFIX"))
    shard_count = 1 if force_rebuild or scoped else 10
    for shard in range(shard_count):
        shard_items = [
            item for item in items
            if scan_reader_assets.shard_for_key(item["key"], shard_count) == shard
        ]
        (output_root / f"queue-{shard}.json").write_text(
            json.dumps({**queue, "items": shard_items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_root / "snapshot.json").write_text(
        json.dumps({**queue, "items": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return extension, len(items), len(queue.get("stale_keys", [])), queue.get("authoritative_snapshot") is True


def main() -> int:
    extension, count, stale_count, authoritative = prepare()
    shard_count = 1 if (os.environ.get("FORCE_REBUILD") == "true"
                        or os.environ.get("INPUT_PATH") or os.environ.get("INPUT_REPO")
                        or os.environ.get("INPUT_EXTENSION") or os.environ.get("INPUT_PATH_PREFIX")) else 10
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"extension={extension}\n")
        output.write(f"count={count}\n")
        output.write(f"stale_count={stale_count}\n")
        output.write(f"authoritative={str(authoritative).lower()}\n")
        output.write(f"shards={'[0]' if shard_count == 1 else '[0,1,2,3,4,5,6,7,8,9]'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
