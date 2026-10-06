#!/usr/bin/env python3
"""Merge ebook shard indexes into one canonical index per extension."""

from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config


EXTENSIONS = ("epub", "mobi", "azw3", "fb2", "chm")


def main() -> int:
    namespace, bucket = os.environ.get("HF_S3_BUCKET", "vomebook/reader-assets-v2").rsplit("/", 1)
    client = boto3.client(
        "s3", endpoint_url=f"https://s3.hf.co/{namespace}",
        aws_access_key_id=os.environ["HF_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["HF_S3_SECRET_ACCESS_KEY"],
        config=Config(region_name="us-east-1", s3={"addressing_style": "path"},
                      retries={"mode": "adaptive", "max_attempts": 8}),
    )
    for extension in EXTENSIONS:
        root = f"chapters/ebook/{extension}"
        keys = []
        for page in client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=f"{root}/index-"):
            keys.extend(item["Key"] for item in page.get("Contents", [])
                        if item["Key"].endswith(".json"))
        if not keys:
            continue
        files = {}
        failures = {}
        for key in sorted(keys):
            payload = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
            for entry in payload.get("files", []):
                if isinstance(entry, dict) and entry.get("key"):
                    files[entry["key"]] = entry
            for entry in payload.get("failures", []):
                if isinstance(entry, dict) and entry.get("key"):
                    failures[entry["key"]] = entry
        body = (json.dumps({
            "version": 1, "kind": "ebook-chapter-stream-index",
            "files": [files[key] for key in sorted(files)],
            "failures": [failures[key] for key in sorted(failures) if key not in files],
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        client.put_object(Bucket=bucket, Key=f"{root}/index.json", Body=body,
                          ContentType="application/json")
        print(f"{extension}: merged {len(files)} files, {len(failures)} failures from {len(keys)} shard indexes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
