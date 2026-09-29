#!/usr/bin/env python3
"""
每天拉取 9 个 HF 仓库的 直接目录.txt，生成 search_data.json。
仅当有仓库 commit SHA 变化时才重新生成。

用法:
  python scripts/fetch_and_parse.py
  可选环境变量: HF_TOKEN（提升 API 频率限制）
  FORCE_SYNC=true 可在来源未变化时强制重新生成
"""

import json
import os
import sys
import time
import gzip
import urllib.parse
import posixpath
import urllib.error
import urllib.request
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

REPOS = [
    "VoiceOfML/VOMEBOOK",
    "VoiceOfML/SovMaterials",
    "VoiceOfML/GPCREducation",
    "VoiceOfML/Teachers",
    "VoiceOfML/MLMRL-Library",
    "VoiceOfML/Japanese-Materials",
    "VoiceOfML/A-Historical-Learning-Data",
    "VoiceOfML/MLMRL-Hub",
    "VoiceOfML/Omnibus",
]

STATE_FILE = Path("state/commits.json")
OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "search_data.json"
FOLDER_TREE_FILE = OUTPUT_DIR / "folder_tree.json"
FOLDER_BROWSER_FILE = OUTPUT_DIR / "folder_browser.json"

API_DATASETS = "https://huggingface.co/api/datasets"
RAW_BASE = "https://huggingface.co/datasets"

RECORD_KEY_MAP = {
    "Repo": "r",
    "File": "f",
    "Extension": "e",
    "Folder": "d",
    "Size": "s",
    "HasTxt": "t",
}

TREE_KEY_MAP = {
    "name": "n",
    "count": "c",
    "hasDirectFiles": "df",
    "children": "ch",
}

BROWSER_KEY_MAP = {
    "folders": "d",
    "files": "f",
    "name": "n",
    "count": "c",
    "ext": "e",
    "hasTxt": "t",
    "size": "s",
}

SEARCH_DATA_VERSION = 2
TRANSIENT_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
HTTP_RETRY_DELAYS = (1, 2, 4)


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def _make_request(url: str, token: str) -> urllib.request.Request:
    """构造带 User-Agent 和可选 Bearer token 的 HTTP 请求。"""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "VoiceOfML-Search-Pipeline/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def _retry_delay(attempt: int) -> int:
    return HTTP_RETRY_DELAYS[min(attempt, len(HTTP_RETRY_DELAYS) - 1)]


def _open_with_retries(req: urllib.request.Request, timeout: int):
    """Open an HF request, retrying only errors that may resolve by themselves."""
    for attempt in range(len(HTTP_RETRY_DELAYS) + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_HTTP_CODES or attempt >= len(HTTP_RETRY_DELAYS):
                raise
            delay = _retry_delay(attempt)
            print(f"  ⚠ HTTP {error.code}，{delay}s 后重试 ({attempt + 1}/{len(HTTP_RETRY_DELAYS)})...")
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt >= len(HTTP_RETRY_DELAYS):
                raise
            delay = _retry_delay(attempt)
            print(f"  ⚠ 网络请求失败 [{error}]，{delay}s 后重试 ({attempt + 1}/{len(HTTP_RETRY_DELAYS)})...")
            time.sleep(delay)


def http_get_json(url: str, token: str = "") -> dict | list:
    """GET JSON, retrying transient HF/API failures before returning empty data."""
    try:
        with _open_with_retries(_make_request(url, token), timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as error:
        print(f"  ⚠ HTTP GET JSON 失败 [{url}]: {error}")
        return {}


def http_get_text(url: str, token: str = "") -> str:
    """GET text, retrying transient HF/API failures before returning empty text."""
    try:
        with _open_with_retries(_make_request(url, token), timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as error:
        print(f"  ⚠ HTTP GET TEXT 失败 [{url}]: {error}")
        return ""


def encode_url_path(raw_path: str) -> str:
    """
    对路径进行 URL 百分号编码 (UTF-8)，保留 '/' 不编码。

    示例:
      •重要资料/《国际歌》.pdf
      → %E2%80%A2%E9%87%8D%E8%A6%81%E8%B5%84%E6%96%99/%E3%80%8A%E5%9B%BD%E9%99%85%E6%AD%8C%E3%80%8B.pdf
    """
    return urllib.parse.quote(raw_path, safe="/")


def build_relative_path(record: dict) -> str:
    filename = record.get("File", "")
    extension = record.get("Extension", "")
    full_name = f"{filename}.{extension}" if extension else filename
    folders = record.get("Folder", []) or []
    return posixpath.join(*folders, full_name) if folders else full_name


def encode_record(record: dict) -> dict:
    return {
        RECORD_KEY_MAP["Repo"]: record.get("Repo", ""),
        RECORD_KEY_MAP["File"]: record.get("File", ""),
        RECORD_KEY_MAP["Extension"]: record.get("Extension", ""),
        RECORD_KEY_MAP["Folder"]: record.get("Folder", []) or [],
        RECORD_KEY_MAP["Size"]: record.get("Size", ""),
        RECORD_KEY_MAP["HasTxt"]: record.get("HasTxt", False),
    }


def encode_search_payload(records: list[dict]) -> dict:
    repo_ids = {}
    repos = []
    folder_ids = {}
    folders = []
    encoded_records = []

    for record in records:
        repo = record.get("Repo", "")
        if repo not in repo_ids:
            repo_ids[repo] = len(repos)
            repos.append(repo)

        folder_tuple = tuple(record.get("Folder", []) or [])
        if folder_tuple not in folder_ids:
            folder_ids[folder_tuple] = len(folders)
            folders.append(list(folder_tuple))

        encoded_records.append([
            repo_ids[repo],
            record.get("File", ""),
            record.get("Extension", ""),
            folder_ids[folder_tuple],
            record.get("Size", ""),
            1 if record.get("HasTxt", False) else 0,
        ])

    return {
        "v": SEARCH_DATA_VERSION,
        "rp": repos,
        "fd": folders,
        "rc": encoded_records,
    }


def encode_tree_node(node: dict) -> dict:
    encoded = {
        TREE_KEY_MAP["name"]: node.get("name", ""),
        TREE_KEY_MAP["count"]: node.get("count", 0),
        TREE_KEY_MAP["hasDirectFiles"]: node.get("hasDirectFiles", False),
        TREE_KEY_MAP["children"]: [encode_tree_node(child) for child in node.get("children", [])],
    }
    return encoded


def encode_folder_tree(tree_by_repo: dict) -> dict:
    return {
        repo: [encode_tree_node(node) for node in nodes]
        for repo, nodes in tree_by_repo.items()
    }


def encode_browser_folder_item(item: dict) -> dict:
    return {
        BROWSER_KEY_MAP["name"]: item.get("name", ""),
        BROWSER_KEY_MAP["count"]: item.get("count", 0),
    }


def encode_browser_file_item(item: dict) -> dict:
    return {
        BROWSER_KEY_MAP["name"]: item.get("name", ""),
        BROWSER_KEY_MAP["ext"]: item.get("ext", ""),
        BROWSER_KEY_MAP["hasTxt"]: item.get("hasTxt", False),
        BROWSER_KEY_MAP["size"]: item.get("size", ""),
    }


def encode_folder_browser(browser_by_repo: dict) -> dict:
    encoded = {}
    for repo, repo_browser in browser_by_repo.items():
        encoded[repo] = {}
        for path, entry in repo_browser.items():
            encoded[repo][path] = {
                BROWSER_KEY_MAP["folders"]: [encode_browser_folder_item(item) for item in entry.get("folders", [])],
                BROWSER_KEY_MAP["files"]: [encode_browser_file_item(item) for item in entry.get("files", [])],
            }
    return encoded

def batch_get_sizes(repo: str, revision: str, paths: list[str], token: str, max_bytes: int = 80000) -> dict:
    url = f"https://huggingface.co/api/datasets/{repo}/paths-info/{revision}"
    size_map = {}
    total = len(paths)
    idx = 0
    batch_num = 0

    while idx < total:
        # 动态取一批，控制 payload 不超过 max_bytes
        batch = []
        payload_size = 0
        overhead = 30  # {"paths":[]} 的固定开销

        while idx < total and payload_size + overhead + len(paths[idx].encode('utf-8')) + 5 < max_bytes:
            batch.append(paths[idx])
            payload_size += len(paths[idx].encode('utf-8')) + 3  # 每条加引号和逗号
            idx += 1

        if not batch:
            # 单条路径就超长（罕见），强制单独发送
            batch = [paths[idx]]
            idx += 1

        batch_num += 1
        payload = json.dumps({"paths": batch}).encode("utf-8")

        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("User-Agent", "VoiceOfML-Search-Pipeline/1.0")
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        try:
            with _open_with_retries(req, timeout=60) as resp:
                results = json.loads(resp.read().decode("utf-8"))
                for item in results:
                    p = item.get("path", "")
                    s = item.get("size", 0)
                    if p and s:
                        size_map[p] = s
        except Exception as e:
            print(f"  ⚠ paths-info 失败 (batch {batch_num}, {len(batch)}条): {e}")

        if idx < total:
            time.sleep(0.3)

    return size_map

def parse_one_line(line: str, repo: str, size_map: dict) -> dict | None:
    """
    解析 txt 中的一行路径，返回一条 JSON 记录。不合法则返回 None。

    输入示例:
      ./•重要资料/《国际歌》在中国——国际歌的译本、底本与传播.pdf

    返回:
      {
        "Repo": "VoiceOfML/MLMRL-Hub",
        "File": "《国际歌》在中国——国际歌的译本、底本与传播",
        "Extension": "pdf",
        "Link": "https://huggingface.co/datasets/VoiceOfML/MLMRL-Hub/resolve/main/%E2%80%A2...",
        "Path": "https://huggingface.co/datasets/VoiceOfML/MLMRL-Hub/blob/main/%E2%80%A2...",
        "Folder": ["•重要资料"],
        "Size": 12345,
        "HasTxt": false
      }
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # 过滤 Git LFS 指针行
    if line.startswith("version https://git-lfs") or line.startswith("oid sha256:") or line.startswith("size "):
        return None
    # 去掉开头的 ./
    if line.startswith("./"):
        rel_path = line[2:]
    elif line.startswith("/"):
        rel_path = line[1:]
    else:
        rel_path = line

    if not rel_path:
        return None

    # 拆分文件夹部分和文件名部分
    parts = rel_path.split("/")
    filename_part = parts[-1] if parts else ""
    folder_parts = parts[:-1] if len(parts) > 1 else []

    # 提取文件名和扩展名
    if "." in filename_part and not filename_part.startswith("."):
        # 普通文件: name.ext
        last_dot = filename_part.rfind(".")
        file_name = filename_part[:last_dot]
        extension = filename_part[last_dot + 1:]
    elif filename_part.startswith(".") and filename_part.count(".") >= 2:
        # 隐藏文件带扩展名: .gitignore
        last_dot = filename_part.rfind(".")
        file_name = filename_part[:last_dot]
        extension = filename_part[last_dot + 1:]
    else:
        # 无后缀文件（纯文件名 或 只以点开头的隐藏文件如 .hidden）
        file_name = filename_part
        extension = ""

    # 文件大小（可能为空字符串）
    size = size_map.get(rel_path, "")

    return {
        "Repo": repo,
        "File": file_name,
        "Extension": extension,
        "Folder": folder_parts,
        "Size": size,
        "HasTxt": False,
    }


def get_repo_sha(repo: str, token: str) -> str:
    """
    通过 HF API 获取仓库最新 commit SHA。
    返回空字符串表示获取失败。
    """
    url = f"{API_DATASETS}/{repo}"
    data = http_get_json(url, token)
    if isinstance(data, dict):
        return data.get("sha", "")
    return ""


def build_folder_tree(records: list[dict]) -> dict:
    tree_by_repo = {}
    browser_by_repo = {}

    for rec in records:
        repo = rec.get("Repo", "")
        folders = rec.get("Folder", []) or []
        dir_path = "/".join(folders)

        if repo not in tree_by_repo:
            short = repo.split("/")[-1] if repo else ""
            tree_by_repo[repo] = [{
                "name": short,
                "path": "",
                "count": 0,
                "hasDirectFiles": False,
                "hasChildren": False,
                "showSelfToggle": False,
                "children": [],
            }]
            browser_by_repo[repo] = {}

        root = tree_by_repo[repo][0]
        root["count"] += 1

        if dir_path not in browser_by_repo[repo]:
            browser_by_repo[repo][dir_path] = {"folders": [], "files": [], "current_path": dir_path}

        browser_by_repo[repo][dir_path]["files"].append({
            "name": rec.get("File", ""),
            "ext": rec.get("Extension", ""),
            "hasTxt": rec.get("HasTxt", False),
            "size": rec.get("Size", ""),
        })

        node = root
        parent_path = ""
        for depth, part in enumerate(folders, start=1):
            path = "/".join(folders[:depth])
            child = next((c for c in node["children"] if c["path"] == path), None)
            if child is None:
                child = {
                    "name": part,
                    "path": path,
                    "count": 0,
                    "hasDirectFiles": False,
                    "hasChildren": False,
                    "showSelfToggle": False,
                    "children": [],
                }
                node["children"].append(child)
                parent_entry = browser_by_repo[repo].setdefault(parent_path, {"folders": [], "files": [], "current_path": parent_path})
                parent_entry["folders"].append({"name": part, "path": path, "count": 0})
            child["count"] += 1
            node = child
            parent_path = path

        if folders:
            node["hasDirectFiles"] = True
        else:
            root["hasDirectFiles"] = True

    for repo, roots in tree_by_repo.items():
        def finalize(node: dict) -> None:
            node["children"].sort(key=lambda x: x["name"])
            node["hasChildren"] = len(node["children"]) > 0
            node["showSelfToggle"] = bool(node["hasDirectFiles"] and node["hasChildren"] and node["path"] != "")
            browser_entry = browser_by_repo[repo].setdefault(node["path"], {"folders": [], "files": [], "current_path": node["path"]})
            browser_entry["folders"].sort(key=lambda x: x["name"])
            browser_entry["files"].sort(key=lambda x: x["name"])
            for folder_item in browser_entry["folders"]:
                child_node = next((c for c in node["children"] if c["path"] == folder_item["path"]), None)
                if child_node:
                    folder_item["count"] = child_node["count"]
            for child in node["children"]:
                finalize(child)

        finalize(roots[0])

    return {"tree": tree_by_repo, "browser": browser_by_repo}


def write_json_gz(path: Path, data) -> None:
    json_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(json_text, encoding="utf-8")
    path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(json_text.encode("utf-8"), compresslevel=9, mtime=0))


def write_action_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")

# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    token = os.environ.get("HF_TOKEN", "")
    force_sync = os.environ.get("FORCE_SYNC", "false").lower() == "true"

    print("=" * 60)
    print("📂 VoiceOfML Search Pipeline — fetch_and_parse")
    print("=" * 60)

    # ── 1. 读取上次 commit 状态 ────────────────────────
    old_state = {}
    if STATE_FILE.exists():
        try:
            old_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            print(f"📖 已读取上次状态: {len(old_state)} 个仓库")
        except json.JSONDecodeError:
            print("⚠ 状态文件损坏，将全量重新生成")
    else:
        print("📖 无历史状态文件，将全量生成")

    # ── 2. 检查每个仓库的最新 commit SHA ───────────────
    new_state = {}
    changed_repos = []
    failed_repos = []

    for repo in REPOS:
        print(f"\n🔍 检查仓库: {repo}")
        old_sha = old_state.get(repo, "")
        sha = get_repo_sha(repo, token)
        new_state[repo] = sha or old_sha

        if not sha:
            print(f"  ⚠ 无法获取 SHA")
            failed_repos.append(repo)
            continue

        if sha != old_sha or not old_sha:
            print(f"  🔄 有变更: {old_sha[:8] if old_sha else '(新)'} → {sha[:8]}")
            changed_repos.append(repo)
        else:
            print(f"  ✅ 无变更: {sha[:8]}")

    # ── 3. 判断是否需要重新生成 ─────────────────────────
    if failed_repos:
        write_action_output("data_changed", "false")
        print("\n❌ 无法确认全部来源版本，停止同步: " + ", ".join(failed_repos))
        return 1

    data_changed = bool(changed_repos) or force_sync
    write_action_output("data_changed", "true" if data_changed else "false")

    if not data_changed:
        print("\n✅ 所有仓库无变更，跳过生成。")
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(new_state, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return 0

    if force_sync and not changed_repos:
        print("\n🔄 已要求强制同步，重新生成当前数据...")
    else:
        print(f"\n🔄 {len(changed_repos)} 个仓库有变更，开始重新生成...")

    # ── 4. 拉取 txt 并解析 ─────────────────────────────
    all_records = []
    failed_catalogs = []

    for repo in REPOS:
        print(f"\n📥 处理仓库: {repo}")

        # 1. 下载 txt
        revision = new_state[repo]
        txt_url = f"{RAW_BASE}/{repo}/resolve/{revision}/{urllib.parse.quote('直接目录.txt')}"
        print(f"   📄 下载: {txt_url}")
        txt_content = http_get_text(txt_url, token)
        if not txt_content:
            print(f"   ❌ 无法下载，停止生成")
            failed_catalogs.append(repo)
            continue

        # 2. 先解析（Size 暂时为空）
        lines = txt_content.split("\n")
        repo_records = []
        all_paths = []
        for line in lines:
            rec = parse_one_line(line, repo, {})  # 空 size_map
            if rec:
                repo_records.append(rec)
                # 提取相对路径用于 paths-info 请求
                rel_path = build_relative_path(rec)
                all_paths.append(rel_path)

        print(f"   ✅ 解析到 {len(repo_records)} 条记录")

        # 3. 批量获取大小
        print(f"   📏 批量获取文件大小 ({len(all_paths)} 个文件)...")
        size_map = batch_get_sizes(repo, revision, all_paths, token)
        
        # 4. 补全 Size
        filled = 0
        for i, rec in enumerate(repo_records):
            path_key = all_paths[i]
            sz = size_map.get(path_key, "")
            if sz:
                rec["Size"] = sz
                filled += 1
        print(f"   ✅ 补全了 {filled} 个文件的大小")

        all_records.extend(repo_records)
        time.sleep(0.5)
    if failed_catalogs:
        print("\n❌ 无法生成完整搜索数据: " + ", ".join(failed_catalogs))
        return 1
    # ── 5. 写入 output/*.json(+gz) ─────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(encode_search_payload(all_records), ensure_ascii=False, separators=(",", ":"))
    OUTPUT_FILE.write_text(json_text, encoding="utf-8")
    OUTPUT_FILE.with_suffix(".json.gz").write_bytes(gzip.compress(json_text.encode("utf-8"), compresslevel=9, mtime=0))

    folder_meta = build_folder_tree(all_records)
    write_json_gz(FOLDER_TREE_FILE, encode_folder_tree(folder_meta["tree"]))
    write_json_gz(FOLDER_BROWSER_FILE, encode_folder_browser(folder_meta["browser"]))

    file_size_mb = len(json_text.encode("utf-8")) / 1024 / 1024
    print(f"💾 已写入 {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"💾 已写入 {FOLDER_TREE_FILE} / {FOLDER_TREE_FILE.name}.gz")
    print(f"💾 已写入 {FOLDER_BROWSER_FILE} / {FOLDER_BROWSER_FILE.name}.gz")

    # ── 6. 更新 state/commits.json ─────────────────────
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(new_state, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"💾 已更新 {STATE_FILE}")

    print("\n✅ fetch_and_parse 完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
