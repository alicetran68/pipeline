#!/usr/bin/env python3
"""Report/apply GC only for static PDF conversion objects in reader-assets-v2."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import boto3
from botocore.config import Config


BUCKET = "vomebook/reader-assets-v2"
ROOT = "documents/pdf/"
LIFECYCLE = "reader-index/static_pdf_lifecycle.json"


def client():
    ns, bucket = BUCKET.split("/", 1)
    return boto3.client("s3", endpoint_url=f"https://s3.hf.co/{ns}",
                        aws_access_key_id=os.environ["HF_S3_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["HF_S3_SECRET_ACCESS_KEY"],
                        config=Config(region_name="us-east-1", s3={"addressing_style": "path"})), bucket


def read_json(c, bucket, path, default):
    try:
        return json.loads(c.get_object(Bucket=bucket, Key=path)["Body"].read())
    except Exception:
        return default


def main():
    p = argparse.ArgumentParser(); p.add_argument("--apply", action="store_true"); p.add_argument("--grace-days", type=int, default=0)
    a = p.parse_args(); c, bucket = client(); files = set()
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=ROOT):
        files.update(x["Key"] for x in page.get("Contents", []))
    refs = {x for x in files if x.endswith("/index.json")}
    for index in sorted(refs):
        data = read_json(c, bucket, index, {})
        for entry in data.get("files", []):
            for field in ("object", "manifest"):
                value = entry.get(field) if isinstance(entry, dict) else None
                if isinstance(value, str):
                    refs.add(value)
                    prefix = value.rsplit("/", 1)[0] + "/"
                    refs.update(x for x in files if x.startswith(prefix))
    candidates = sorted(files - refs)
    lifecycle = read_json(c, bucket, LIFECYCLE, {"version": 1, "orphans": {}})
    today = date.today().isoformat(); orphans = lifecycle.get("orphans", {})
    for path in candidates: orphans.setdefault(path, {"since": today})
    for path in list(orphans):
        if path not in candidates: del orphans[path]
    cutoff = date.today() - timedelta(days=a.grace_days)
    expired = [path for path, item in orphans.items() if date.fromisoformat(item["since"]) <= cutoff]
    lifecycle = {"version": 1, "orphans": dict(sorted(orphans.items()))}
    c.put_object(Bucket=bucket, Key=LIFECYCLE, Body=(json.dumps(lifecycle, sort_keys=True, indent=2) + "\n").encode(), ContentType="application/json")
    print(f"bucket={BUCKET} files={len(files)} references={len(refs)} orphans={len(candidates)} expired={len(expired)}")
    if a.apply and expired:
        c.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": x} for x in expired], "Quiet": True}); print(f"deleted={len(expired)}")
    elif a.apply: print("deleted=0")


if __name__ == "__main__": main()
