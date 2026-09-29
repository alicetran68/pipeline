"""Persistent PDF assessment state shared by all asset publishers."""
import json
from pathlib import Path
from huggingface_hub.errors import HfHubHTTPError

try:
    from . import shared
    from .reader_bucket import INDEX_FILES, read_json as read_bucket_json
except ImportError:
    import shared
    from reader_bucket import INDEX_FILES, read_json as read_bucket_json

MANIFEST_NAME = "pdf_range_manifest.json"


def empty_state():
    return {"version": 1, "artifact_bucket": shared.PDF_RANGE_BUCKET, "files": {}, "inventories": {}}


def artifact_bucket(state: dict, entry: dict) -> str:
    return entry.get("artifact_bucket") or state.get("artifact_bucket") or "vomebook/Reader-Assets"


def has_legacy_artifacts(state: dict) -> bool:
    return any(entry.get("status") == "optimized" and artifact_bucket(state, entry) != shared.PDF_RANGE_BUCKET
               for entry in (state or {}).get("files", {}).values() if isinstance(entry, dict))


def remote_state(api, repo, revision=None):
    if type(api).__name__ == "HfApi":
        try:
            state = read_bucket_json(INDEX_FILES["range"])
            if state.get("version") != 1 or not isinstance(state.get("files"), dict):
                raise ValueError("invalid PDF range assessment state")
            return state
        except (FileNotFoundError, OSError, ValueError):
            pass
    try:
        path = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=MANIFEST_NAME,
                                   revision=revision)
    except HfHubHTTPError as error:
        if getattr(error.response, "status_code", None) == 404:
            return empty_state()
        raise
    state = json.loads(Path(path).read_text())
    if state.get("version") != 1 or not isinstance(state.get("files"), dict):
        raise ValueError("invalid PDF range assessment state")
    return state


def apply_optimized(files, manifest, state):
    # PDF structure optimization is retired. Page streams/OCR are the only
    # generated PDF delivery route; assessment state remains historical data
    # until the dedicated optimized bucket is retired.
    return
