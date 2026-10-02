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
    bucket_migration = queue.get("bucket_migration") is True
    extension = "static" if bucket_migration and items else (items[0]["extension"] if items else "")
    if not bucket_migration:
        items = [item for item in items if item["extension"] == extension]
    queue["items"] = items
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")

    force_rebuild = os.environ.get("FORCE_REBUILD") == "true"
    force_single_shard = force_rebuild and not bucket_migration
    scoped = bool(os.environ.get("INPUT_PATH") or os.environ.get("INPUT_REPO")
                  or os.environ.get("INPUT_EXTENSION"))
    shard_count = 1 if force_single_shard or (scoped and not bucket_migration) else 20
    active_shards = []
    for shard in range(shard_count):
        shard_items = [
            item for item in items
            if scan_reader_assets.shard_for_key(item["key"], shard_count) == shard
        ]
        if shard_items:
            active_shards.append(shard)
        (output_root / f"queue-{shard}.json").write_text(
            json.dumps({**queue, "items": shard_items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_root / "shards.json").write_text(json.dumps(active_shards), encoding="utf-8")
    (output_root / "snapshot.json").write_text(
        json.dumps({**queue, "items": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return extension, len(items), len(queue.get("stale_keys", [])), queue.get("authoritative_snapshot") is True


def main() -> int:
    extension, count, stale_count, authoritative = prepare()
    shards = json.loads(Path("output/reader-assets/shards.json").read_text(encoding="utf-8"))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"extension={extension}\n")
        output.write(f"count={count}\n")
        output.write(f"stale_count={stale_count}\n")
        output.write(f"authoritative={str(authoritative).lower()}\n")
        output.write(f"shards={json.dumps(shards, separators=(',', ':'))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
