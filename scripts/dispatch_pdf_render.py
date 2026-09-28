#!/usr/bin/env python3
"""Dispatch one large-PDF render book only when the workflow is idle."""

import json
import os
from urllib.request import Request, urlopen


ACTIVE = frozenset({"pending", "requested", "queued", "in_progress", "waiting"})
WORKFLOW = "pdf-render-inputs.yml"


def dispatch(repo, token):
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
    body = json.dumps({"ref": "main", "inputs": {"limit": "1", "checkpoint": "0",
                                                  "retry_failed": "true",
                                                  "continue_queue": "true"}}).encode()
    request = Request(endpoint + "/dispatches", data=body,
                      headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        if response.status != 204:
            raise ValueError(f"unexpected render dispatch status: {response.status}")
    print("Dispatched large PDF render from current main.")
    return True


if __name__ == "__main__":
    dispatch(os.environ.get("REPO"), os.environ.get("GH_TOKEN"))
