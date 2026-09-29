#!/usr/bin/env python3
"""Publish the compact Reader Assets sidecar to HF Search and GitHub Pages."""

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

try:
    from .reader_assets import READER_ASSETS_REPO
    from .reader_bucket import INDEX_FILES, read_bytes as read_bucket_bytes
except ImportError:
    from reader_assets import READER_ASSETS_REPO
    from reader_bucket import INDEX_FILES, read_bytes as read_bucket_bytes

SIDECAR_NAME = "reader_assets.json.gz"


def run(args: list[str], cwd: str | None = None, env: dict | None = None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(args, cwd=cwd, env=merged, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_auth_env(token: str) -> dict[str, str]:
    credentials = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credentials}",
        "GIT_TERMINAL_PROMPT": "0",
    }


def publish_to_pages(source: Path, pages_repo: str, token: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    if not pages_repo or not token:
        raise RuntimeError("PAGES_REPO and PAGES_TOKEN are required")
    with tempfile.TemporaryDirectory(prefix="reader-index-pages-") as root:
        auth = git_auth_env(token)
        ret, out, err = run(["git", "clone", "--depth", "1", f"https://github.com/{pages_repo}.git", root], env=auth)
        if ret:
            raise RuntimeError(f"Pages clone failed: {err or out}")
        target = Path(root) / "data" / SIDECAR_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        run(["git", "config", "user.name", "github-actions[bot]"], cwd=root)
        run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=root)
        run(["git", "add", f"data/{SIDECAR_NAME}"], cwd=root)
        ret, status, err = run(["git", "status", "--porcelain"], cwd=root)
        if ret:
            raise RuntimeError(f"Pages status failed: {err}")
        if not status:
            return
        ret, out, err = run(["git", "commit", "-m", "chore: update reader assets index"], cwd=root)
        if ret:
            raise RuntimeError(f"Pages commit failed: {err or out}")
        ret, out, err = run(["git", "push"], cwd=root, env=auth)
        if ret:
            raise RuntimeError(f"Pages push failed: {err or out}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-repo", default=os.environ.get("READER_ASSETS_REPO", READER_ASSETS_REPO))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=hf_token)
    source = None
    temporary_source = False
    if type(api) is HfApi:
        try:
            with tempfile.NamedTemporaryFile(prefix="reader-index-", suffix=".gz", delete=False) as handle:
                handle.write(read_bucket_bytes(INDEX_FILES["sidecar"], hf_token))
                source = Path(handle.name)
                temporary_source = True
        except (FileNotFoundError, OSError, ValueError):
            source = None
    if source is None:
        source = Path(api.hf_hub_download(
            repo_id=args.assets_repo, repo_type="dataset", filename=SIDECAR_NAME,
        ))
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError("Reader Assets sidecar is empty")
    try:
        publish_to_pages(
            source, os.environ.get("PAGES_REPO", ""), os.environ.get("PAGES_TOKEN", ""),
            dry_run=args.dry_run,
        )
    finally:
        if temporary_source:
            source.unlink(missing_ok=True)
    print("validated reader assets index" if args.dry_run else "published reader assets index to GitHub Pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
