#!/usr/bin/env python3
"""S3 adapter for current Reader buckets."""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import posixpath
import tempfile
from pathlib import Path
from urllib.parse import quote


class S3NotFound(FileNotFoundError):
    pass


class HubBucketStore:
    """Current bucket API access with independent credentials for OCR inputs."""

    def __init__(self):
        from huggingface_hub import HfApi
        self.api = HfApi(token=os.environ.get("HF_TOKEN"))

    def token(self, bucket):
        if bucket.startswith("melsm/"):
            return os.environ.get("HF_INPUT_TOKEN")
        return os.environ.get("HF_TOKEN")

    def list_files(self, bucket: str, prefixes: tuple[str, ...]) -> set[str]:
        return {item.path for prefix in prefixes for item in self.api.list_bucket_tree(
            bucket, prefix=prefix or None, recursive=True, token=self.token(bucket))
                if item.type == "file"}

    def read_bytes(self, bucket: str, path: str) -> bytes:
        from huggingface_hub.utils import get_session, hf_raise_for_status
        token = self.token(bucket)
        response = get_session().get(
            f"https://huggingface.co/buckets/{bucket}/resolve/{quote(path, safe='/')}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            follow_redirects=True, timeout=120)
        if response.status_code == 404:
            raise FileNotFoundError(path)
        hf_raise_for_status(response)
        return response.content

    def put_json(self, bucket: str, path: str, payload: dict) -> None:
        from huggingface_hub import batch_bucket_files
        with tempfile.TemporaryDirectory(prefix="reader-gc-state-") as directory:
            target = Path(directory) / "state.json"
            target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            batch_bucket_files(bucket, add=[(str(target), path)], token=self.token(bucket))


class S3BucketStore:
    def __init__(self) -> None:
        self._access_key = os.environ.get("HF_S3_ACCESS_KEY_ID")
        self._secret_key = os.environ.get("HF_S3_SECRET_ACCESS_KEY")
        if not self._access_key or not self._secret_key:
            raise RuntimeError("HF_S3_ACCESS_KEY_ID and HF_S3_SECRET_ACCESS_KEY are required")
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError("boto3 is required for Reader bucket operations") from error
        self._boto3 = boto3
        self._config = Config(region_name="us-east-1", s3={"addressing_style": "path"},
                              retries={"mode": "adaptive", "max_attempts": 8},
                              max_pool_connections=64)
        self._namespace = os.environ.get("HF_S3_NAMESPACE", "vomebook")
        self._input_namespace = os.environ.get("HF_S3_INPUT_NAMESPACE", "melsm")
        self._input_bucket = os.environ.get("HF_S3_INPUT_BUCKET", "pdf-archive-v2")
        self._clients = {}

    def _location(self, bucket: str) -> tuple[str, str]:
        if "/" in bucket:
            return bucket.rsplit("/", 1)
        if bucket == self._input_bucket:
            return self._input_namespace, bucket
        return self._namespace, bucket

    def _client(self, namespace: str):
        separate = namespace == self._input_namespace and namespace != self._namespace
        key = os.environ.get("HF_S3_INPUT_ACCESS_KEY_ID") if separate else self._access_key
        secret = os.environ.get("HF_S3_INPUT_SECRET_ACCESS_KEY") if separate else self._secret_key
        if not key or not secret:
            raise RuntimeError(f"S3 credentials are required for namespace {namespace}")
        cache_key = (namespace, key)
        if cache_key not in self._clients:
            self._clients[cache_key] = self._boto3.client(
                "s3", endpoint_url=f"https://s3.hf.co/{namespace}",
                aws_access_key_id=key, aws_secret_access_key=secret, config=self._config)
        return self._clients[cache_key]

    def list_files(self, bucket: str, prefixes: tuple[str, ...]) -> set[str]:
        namespace, name = self._location(bucket)
        client = self._client(namespace)
        def one(prefix):
            result = set()
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=name, Prefix=prefix):
                result.update(item["Key"] for item in page.get("Contents", []))
            return result
        files = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(prefixes)))) as pool:
            for result in pool.map(one, prefixes):
                files.update(result)
        return files

    def read_bytes(self, bucket: str, path: str) -> bytes:
        namespace, name = self._location(bucket)
        try:
            response = self._client(namespace).get_object(Bucket=name, Key=path)
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}:
                raise S3NotFound(path) from error
            raise
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def put_json(self, bucket: str, path: str, payload: dict) -> None:
        namespace, name = self._location(bucket)
        self._client(namespace).put_object(
            Bucket=name, Key=path,
            Body=(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
            ContentType="application/json")

    def existing_files(self, bucket: str, paths: set[str]) -> set[str]:
        namespace, name = self._location(bucket)
        client = self._client(namespace)
        def exists(path):
            try:
                client.head_object(Bucket=name, Key=path)
                return path
            except Exception as error:
                response = getattr(error, "response", {})
                code = response.get("Error", {}).get("Code")
                status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                    return None
                raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(64, max(1, len(paths)))) as pool:
            return {path for path in pool.map(exists, sorted(paths)) if path}

    def manifest_references(self, bucket: str, paths: set[str]) -> dict[str, set[str]]:
        manifests = {path for path in paths if path.endswith(("/page-manifest.json", "/chapter-manifest.json"))}
        def load(path):
            raw = self.read_bytes(bucket, path)
            payload = json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)
            root = posixpath.dirname(path)
            if path.endswith("/page-manifest.json"):
                count = payload.get("page_count")
                if type(count) is not int or count < 1:
                    raise ValueError(f"invalid page manifest: {path}")
                return path, {f"{root}/pages/page-{number:06d}.webp" for number in range(1, count + 1)}
            chapters = payload.get("chapters")
            if not isinstance(chapters, list):
                raise ValueError(f"invalid chapter manifest: {path}")
            refs = {posixpath.normpath(posixpath.join(root, item["path"]))
                    for item in chapters if isinstance(item, dict) and isinstance(item.get("path"), str)}
            search = payload.get("search_index")
            if isinstance(search, dict) and isinstance(search.get("path"), str):
                refs.add(posixpath.normpath(posixpath.join(root, search["path"])))
            return path, refs
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(manifests)))) as pool:
            return dict(pool.map(load, sorted(manifests)))

    def delete(self, bucket: str, paths: list[str]) -> None:
        namespace, name = self._location(bucket)
        client = self._client(namespace)
        for start in range(0, len(paths), 1000):
            response = client.delete_objects(Bucket=name, Delete={
                "Objects": [{"Key": path} for path in paths[start:start + 1000]], "Quiet": True})
            if response.get("Errors"):
                raise RuntimeError(f"S3 delete failed for {len(response['Errors'])} object(s)")
