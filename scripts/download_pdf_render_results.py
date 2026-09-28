#!/usr/bin/env python3
"""Download render result artifacts serially for large publish/recovery runs."""

import argparse
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(NoRedirect)
NAME = re.compile(r"pdf-render-results-(\d+)")
API = "https://api.github.com"


def api_get(path: str, token: str, *, zip_archive: bool = False) -> bytes:
    request = urllib.request.Request(API + path, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "pdf-render-publisher",
    })
    for attempt in range(9):
        try:
            with OPENER.open(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 302 and zip_archive:
                location = exc.headers.get("Location", "")
                if urlsplit(location).scheme != "https":
                    raise ValueError("artifact redirect must use HTTPS") from exc
                # Signed blob links are already authenticated; never send the API token there.
                with urllib.request.urlopen(urllib.request.Request(location, headers={
                        "User-Agent": "pdf-render-publisher"}), timeout=120) as response:
                    return response.read()
            message = exc.read().decode("utf-8", errors="replace")
            secondary = exc.code == 403 and "secondary rate limit" in message.lower()
            if not (secondary or exc.code == 429 or 500 <= exc.code < 600) or attempt == 8:
                raise RuntimeError(f"GitHub API {exc.code}: {message[:300]}") from exc
            retry_after = exc.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 0
            delay = min(180, max(delay, 10 * 2 ** attempt))
            print(f"GitHub artifact rate limit; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("GitHub artifact download retry limit reached")


def result_artifacts(repo: str, run_id: int, token: str) -> list[dict]:
    selected = {}
    page = 1
    while True:
        payload = json.loads(api_get(
            f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}", token))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or not isinstance(payload.get("total_count"), int):
            raise ValueError("invalid GitHub artifact list")
        for artifact in artifacts:
            match = NAME.fullmatch(str(artifact.get("name", "")))
            if not match:
                continue
            shard = int(match.group(1))
            if shard in selected or artifact.get("expired") or not isinstance(artifact.get("id"), int):
                raise ValueError("duplicate or unavailable render result artifact")
            selected[shard] = artifact
        if page * 100 >= payload["total_count"]:
            break
        if not artifacts:
            raise ValueError("incomplete GitHub artifact listing")
        page += 1
    if not selected:
        raise ValueError("no render result artifacts found")
    return [selected[shard] for shard in sorted(selected)]


def unpack_result(archive: bytes, shard: int, output: Path) -> None:
    expected = f"results-{shard}.json"
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        matches = [item for item in zipped.infolist() if not item.is_dir()
                   and Path(item.filename).name == expected]
        if len(matches) != 1 or matches[0].file_size > 16 * 1024 * 1024:
            raise ValueError(f"invalid render result ZIP for shard {shard}")
        item = matches[0]
        if item.filename.startswith("/") or ".." in Path(item.filename).parts:
            raise ValueError("unsafe render result ZIP path")
        content = zipped.read(item)
    payload = json.loads(content)
    if payload.get("version") != 1 or not isinstance(payload.get("results"), list):
        raise ValueError(f"invalid render results for shard {shard}")
    destination = output / f"pdf-render-results-{shard}" / expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo) or args.run_id < 1:
        raise ValueError("invalid GitHub repository or run ID")
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        raise ValueError("GH_TOKEN is required")
    artifacts = result_artifacts(args.repo, args.run_id, token)
    for index, artifact in enumerate(artifacts, 1):
        shard = int(NAME.fullmatch(artifact["name"]).group(1))
        unpack_result(api_get(f"/repos/{args.repo}/actions/artifacts/{artifact['id']}/zip",
                              token, zip_archive=True), shard, args.output)
        if index % 25 == 0 or index == len(artifacts):
            print(f"downloaded {index}/{len(artifacts)} render results", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
