#!/usr/bin/env python3
"""Report or remove the retired PDF structure-optimization bucket."""

import argparse
import os

from huggingface_hub import batch_bucket_files, list_bucket_tree

try:
    from .shared import PDF_RANGE_BUCKET
except ImportError:
    from shared import PDF_RANGE_BUCKET


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    paths = [item.path for item in list_bucket_tree(PDF_RANGE_BUCKET, recursive=True, token=token)
             if item.type == "file"]
    if len(paths) > args.limit:
        raise RuntimeError(f"refusing to process {len(paths)} objects above limit {args.limit}")
    print(f"found {len(paths)} retired PDF range object(s)")
    for path in paths:
        print(path)
    if args.apply and paths:
        batch_bucket_files(PDF_RANGE_BUCKET, delete=paths, token=token)
        print(f"deleted {len(paths)} retired PDF range object(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
