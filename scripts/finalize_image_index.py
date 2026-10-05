#!/usr/bin/env python3
"""Merge image shard indexes into the canonical image index."""

from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config


def main() -> int:
    namespace, bucket = os.environ.get("HF_S3_BUCKET", "vomebook/reader-assets-v2").rsplit("/", 1)
    client = boto3.client(
        "s3", endpoint_url=f"https://s3.hf.co/{namespace}",
        aws_access_key_id=os.environ["HF_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["HF_S3_SECRET_ACCESS_KEY"],
        config=Config(region_name="us-east-1", s3={"addressing_style": "path"}),
    )
    prefix = "pages/image/index"
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []) if item["Key"].endswith(".json"))
    merged = {}
    for key in sorted(keys):
        payload = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
        for entry in payload.get("files", []):
            if isinstance(entry, dict) and entry.get("key"):
                merged[entry["key"]] = entry
    body = (json.dumps({"version": 1, "kind": "image-page-stream-index",
                       "files": [merged[key] for key in sorted(merged)]},
                       ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    client.put_object(Bucket=bucket, Key="pages/image/index.json", Body=body,
                      ContentType="application/json")
    print(f"merged {len(merged)} image stream mapping(s) from {len(keys)} index file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
