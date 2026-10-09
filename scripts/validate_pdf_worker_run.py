#!/usr/bin/env python3
"""Validate a completed account worker before central artifact publication."""

import argparse
import json
import os
import subprocess
import time

try:
    from . import pdf_worker_lanes
except ImportError:
    import pdf_worker_lanes


def validate_run(config, repository, run):
    owner, separator, name = repository.partition("/")
    if not separator or repository.lower() != pdf_worker_lanes.repository_for_owner(config, owner).lower():
        raise ValueError("PDF worker must run in its configured account repository")
    pdf_worker_lanes.lane_for_owner(config, owner)
    if (run.get("repository", {}).get("full_name", "").lower() != repository.lower()
            or run.get("head_branch") != "main"
            or run.get("path") != ".github/workflows/pdf-account-worker.yml"
            or run.get("event") not in {"workflow_dispatch", "schedule"}
            or run.get("status") != "completed"
            or run.get("conclusion") not in {"success", "failure"}):
        raise ValueError("PDF worker run identity or completion mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()
    config = pdf_worker_lanes.load_config()
    if os.environ.get("GITHUB_REPOSITORY", "").lower() != config["publisher_repository"].lower():
        parser.error("only the configured central repository may publish worker registries")
    owner, separator, name = args.repository.partition("/")
    if not separator or args.repository.lower() != pdf_worker_lanes.repository_for_owner(config, owner).lower() or args.run_id < 1:
        parser.error("invalid worker repository or run ID")
    lane = pdf_worker_lanes.lane_for_owner(config, owner)
    for attempt in range(31):
        completed = subprocess.run(["gh", "api", f"repos/{args.repository}/actions/runs/{args.run_id}"],
                                   check=True, capture_output=True, text=True, timeout=30)
        run = json.loads(completed.stdout)
        if run.get("status") == "completed":
            break
        if attempt == 30:
            raise TimeoutError("PDF worker has not finished after notification")
        time.sleep(10)
    validate_run(config, args.repository, run)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"owner={owner.lower()}\nlane={lane}\n")


if __name__ == "__main__":
    main()
