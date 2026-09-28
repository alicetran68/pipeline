#!/usr/bin/env python3
"""Dispatch one large-PDF render book only when the workflow is idle."""

import json
import io
import os
import zipfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


ACTIVE = frozenset({"pending", "requested", "queued", "in_progress", "waiting"})
WORKFLOW = "pdf-render-inputs.yml"


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        redirected = super().redirect_request(request, file, code, message, headers, new_url)
        if redirected and urlparse(request.full_url).netloc != urlparse(new_url).netloc:
            redirected.headers.pop("Authorization", None)
            redirected.unredirected_hdrs.pop("Authorization", None)
        return redirected


def urlopen(request, timeout=30):
    return build_opener(_SafeRedirectHandler).open(request, timeout=timeout)


def queue_has_books(repo, token, run_id):
    if not str(run_id).isdigit():
        raise ValueError("SOURCE_RUN must be a numeric workflow run id")
    base = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    with urlopen(Request(base + "/artifacts?per_page=100", headers=headers), timeout=30) as response:
        payload = json.load(response)
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    queue = next((artifact for artifact in artifacts or []
                  if isinstance(artifact, dict) and artifact.get("name") == "pdf-render-queue"
                  and artifact.get("expired") is False), None)
    if not queue or not isinstance(queue.get("archive_download_url"), str):
        raise ValueError("serial render queue artifact is missing")
    with urlopen(Request(queue["archive_download_url"], headers=headers), timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = [name for name in bundle.namelist() if name.endswith("queue.json")]
        if len(names) != 1:
            raise ValueError("serial render queue artifact is invalid")
        data = json.loads(bundle.read(names[0]).decode("utf-8"))
    books = data.get("books") if isinstance(data, dict) else None
    if not isinstance(books, list):
        raise ValueError("serial render queue is invalid")
    return bool(books)


def dispatch(repo, token, source_run=None, retry_failed=True):
    if not repo or not token:
        raise ValueError("REPO and GH_TOKEN are required")
    endpoint = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    with urlopen(Request(endpoint + "/runs?per_page=100", headers=headers), timeout=30) as response:
        payload = json.load(response)
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or any(not isinstance(run, dict) or not isinstance(run.get("status"), str)
                                         for run in runs):
        raise ValueError("invalid render workflow run list")
    if any(run["status"] in ACTIVE for run in runs):
        print("Large PDF render already pending or active; skipping dispatch.")
        return False
    if source_run and not queue_has_books(repo, token, source_run):
        print("Serial large PDF render queue is empty; stopping the chain.")
        return False
    body = json.dumps({"ref": "main", "inputs": {"limit": "1", "checkpoint": "0",
                                                  "retry_failed": "true" if retry_failed else "false",
                                                  "continue_queue": "true"}}).encode()
    request = Request(endpoint + "/dispatches", data=body,
                      headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        if response.status != 204:
            raise ValueError(f"unexpected render dispatch status: {response.status}")
    print("Dispatched large PDF render from current main.")
    return True


if __name__ == "__main__":
    retry_failed = os.environ.get("RETRY_FAILED", "true").lower() in {"1", "true", "yes"}
    dispatch(os.environ.get("REPO"), os.environ.get("GH_TOKEN"),
             os.environ.get("SOURCE_RUN") or None, retry_failed)
