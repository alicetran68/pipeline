#!/usr/bin/env python3
"""Shared contracts for incremental VoiceOfML reader assets."""

import json
import hashlib
import posixpath
import re
import urllib.parse
from pathlib import Path

MANIFEST_VERSION = 1
CHAPTER_MANIFEST_VERSION = 1
READER_ASSETS_REPO = "vomebook/Reader-Assets"
READER_ASSETS_BUCKET = "vomebook/reader-assets-v2"
MANIFEST_NAME = "manifest.json"
# Chapter manifests make multi-file books cheap to open: the Reader fetches the
# manifest and nearby chapters instead of downloading the complete archive.
# Native ebook chapter bundles include an on-demand full-text search index.
CHM_CHAPTER_SPLIT_BYTES = 16 * 1024 * 1024
EPUB_CHAPTER_BUNDLE_DIR = "epub-chapters"
EPUB_CHAPTER_PROFILE = "epub-chapters-v8-bucket"
SPREADSHEET_HTML_PROFILE = "libreoffice-native-html-spreadsheet-v1"
# Retained for validation of already-published image-stream manifests.
SPREADSHEET_PAGE_PROFILE = "libreoffice-html-pages-spreadsheet-v6"
SPREADSHEET_EXTENSIONS = frozenset({"xls", "xlsx", "csv", "ods"})
NATIVE_MEDIA_PROFILE = "native-media-cdn-v1"
BUCKET_NATIVE_MEDIA_EXTENSIONS = {
    **{extension: (NATIVE_MEDIA_PROFILE, "audio", f"audio.{extension}") for extension in (
        "mp3", "wav", "m4a", "flac", "mpga",
    )},
    **{extension: (NATIVE_MEDIA_PROFILE, "video", f"video.{extension}") for extension in (
        "mp4", "mov",
    )},
}
BUCKET_NATIVE_EXTENSIONS = {
    "txt": ("native-text-v1", "text", "document.txt"),
    "md": ("native-markdown-v1", "markdown", "document.md"),
    "markdown": ("native-markdown-v1", "markdown", "document.md"),
    "jpg": ("native-image-webp-v1", "image", "document.webp"),
    "jpeg": ("native-image-webp-v1", "image", "document.webp"),
    "png": ("native-image-webp-v1", "image", "document.webp"),
    "bmp": ("native-image-webp-v1", "image", "document.webp"),
    "webp": ("native-image-webp-v1", "image", "document.webp"),
    "vcf": ("native-text-v1", "text", "document.txt"),
    "ini": ("native-text-v1", "text", "document.txt"),
}
CONVERTIBLE_EXTENSIONS = {
    "doc": ("libreoffice-docx-v2", "docx", "document.docx"),
    "docx": ("docx-native-v2", "docx", "document.docx"),
    "epub": ("foliate-original-v1", "foliate", "document.epub"),
    "htm": ("sanitized-html-v5", "html", "document.html"),
    "html": ("sanitized-html-v5", "html", "document.html"),
    "mobi": ("foliate-original-v1", "foliate", "document.mobi"),
    "azw3": ("foliate-original-v1", "foliate", "document.azw3"),
    "fb2": ("foliate-original-v1", "foliate", "document.fb2"),
    "odt": ("calibre-odt-html-v1", "html", "document.html"),
    "rtf": ("calibre-rtf-html-v1", "html", "document.html"),
    "chm": ("calibre-chm-epub-v3", "epub", "document.epub"),
    "tif": ("pillow-pdf-v2", "pdf", "document.pdf"),
    "tiff": ("pillow-pdf-v2", "pdf", "document.pdf"),
    "djvu": ("djvulibre-pdf-v2", "pdf", "document.pdf"),
    "ppt": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "pptx": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "pps": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "odp": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "xls": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "xlsx": ("libreoffice-pdf-office-xlsx-v3", "pdf", "document.pdf"),
    "csv": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "ods": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "wps": ("libreoffice-pdf-office-v2", "pdf", "document.pdf"),
    "mht": ("sanitized-mhtml-v6", "html", "document.html"),
    "mhtml": ("sanitized-mhtml-v6", "html", "document.html"),
    "ps": ("ghostscript-pdf-v5", "pdf", "document.pdf"),
    "caj": ("caj-family-pdf-v1", "pdf", "document.pdf"),
    "kdh": ("caj-family-pdf-v1", "pdf", "document.pdf"),
    "ape": ("ffmpeg-audio-mp3-v1", "audio", "audio.mp3"),
    "wma": ("ffmpeg-audio-mp3-v1", "audio", "audio.mp3"),
    "amr": ("ffmpeg-audio-mp3-v1", "audio", "audio.mp3"),
    "asx": ("ffmpeg-media-auto-v1", "video", "video.mp4"),
    "flac": ("ffmpeg-audio-mp3-v1", "audio", "audio.mp3"),
    "m4a": ("ffmpeg-audio-mp3-v1", "audio", "audio.mp3"),
    "mpga": ("ffmpeg-audio-mp3-v1", "audio", "audio.mp3"),
    "wav": ("ffmpeg-audio-mp3-v1", "audio", "audio.mp3"),
    "flv": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "f4v": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "rm": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "rmvb": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "mkv": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "avi": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "mpg": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "mpeg": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "mts": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "ts": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "wmv": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "swf": ("native-swf-ruffle-v1", "swf", "document.swf"),
    "mov": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
    "mp4": ("ffmpeg-video-mp4-h264-aac-v1", "video", "video.mp4"),
}
PROTECTED_PDF_CONTRACT = ("qpdf-decrypted-v1", "pdf", "document.pdf")
GBK_PDF_CONTRACT = ("gbk-font-repair-v1", "pdf", "document.pdf")
GBK_PDF_FOLDER = "A4 毛泽东主席/03-03 建国以来毛泽东文稿 林一章版"
GBK_PDF_REPAIR_FOLDERS = (
    "A1 马克思&恩格斯/01-04 马克思恩格斯全集 林一章新版",
    "A1 马克思&恩格斯/02-05 马克思恩格斯选集 林一章版",
    "A2 列宁/02-06 列宁选集 林一章版（博览网版）",
    "A3 斯大林/01-04 斯大林全集 林一章旧版",
    "A3 斯大林/02-01 斯大林选集 林一章旧版（博览网版）",
    "A4 毛泽东主席/02-03 毛泽东选集 林一章版 第一版",
    "A4 毛泽东主席/02-05 毛泽东选集 林一章版（博览网版）第二版",
    "其他/鲁迅/鲁迅全集 林一章版",
)
KNOWN_GBK_PDFS = {
    ("VoiceOfML/Teachers", f"{GBK_PDF_FOLDER}/{name}.pdf")
    for name in (
        "合订本", "第1册 (1949.9-1950.12)", "第2册 (1951.1-1951.12)",
        "第3册 (1952.1-1952.12)", "第4册 (1953.1-1954.12)",
        "第5册 (1955.1-1955.12)", "第6册 (1956.1-1957.12)",
        "第7册 (1958.1-1958.12)", "第8册 (1959.1-1959.12)",
        "第9册 (1960.1-1961.12)", "第10册 (1962.1-1963.12)",
        "第11册 (1964.1-1965.12)",
    )
}


def known_gbk_pdf(repo: str, path: str) -> bool:
    return (repo, path) in KNOWN_GBK_PDFS or (
        repo == "VoiceOfML/Teachers" and isinstance(path, str) and path.lower().endswith(".pdf")
        and any(path.startswith(folder + "/") for folder in GBK_PDF_REPAIR_FOLDERS)
    )


PASSWORD_RE = re.compile(
    r"(?:密码|口令|password|passwd)\s*(?:[：:=]\s*|(?=[A-Za-z0-9]))"
    r"([^\s\]〕】）)},，；;]+)",
    re.IGNORECASE,
)
KNOWN_SOURCE_PASSWORDS = {
    ("VoiceOfML/MLMRL-Library", "基础入门书单/入门答疑/风正集.pdf"): "230505",
    ("VoiceOfML/MLMRL-Hub", "000269/1870520043_3072_风正集230220.pdf"): "230220",
    ("VoiceOfML/MLMRL-Hub", "001346/1870520043_15344_风正集230123.pdf"): "230123",
    ("VoiceOfML/MLMRL-Hub", "001346/1870520043_15348_风正集230505.pdf"): "230505",
}


def conversion_contract(extension: str, key: str = "") -> tuple[str, str, str]:
    return CONVERTIBLE_EXTENSIONS[extension]


def needs_epub_chapters(extension: str, reader_mode: str, source_bytes: int) -> bool:
    """All supported native ebook formats use the independently fetched chapter stream."""
    # CHM navigation includes groups and repeated fragment targets. The flat
    # chapter bundle cannot represent that tree; keep its repaired native EPUB.
    return ((extension in {"epub", "mobi", "azw3", "fb2"} and reader_mode == "foliate")
            or (extension == "chm" and reader_mode == "epub"))


def source_password(repo: str, path: str) -> str:
    match = PASSWORD_RE.search(path)
    if match:
        value = match.group(1)
        suffix = Path(path).suffix
        return value[:-len(suffix)] if suffix and value.lower().endswith(suffix.lower()) else value
    return KNOWN_SOURCE_PASSWORDS.get((repo, path), "")


def reusable_object_key(source_sha256: str, profile: str, *, extension: str = "",
                        source_revision: str = "", key: str = "") -> str:
    identity = f"{source_sha256}\0{profile}"
    if extension in {"htm", "html"}:
        identity += f"\0{source_revision}\0{key}"
    return identity


def object_profile_path(profile: str, *, extension: str = "", source_revision: str = "", key: str = "") -> str:
    if extension not in {"htm", "html"}:
        return profile
    context = hashlib.sha256(f"{source_revision}\0{key}".encode("utf-8")).hexdigest()[:16]
    return f"{profile}-{context}"


def validate_object_path(path: str) -> str:
    if (not isinstance(path, str) or not path.startswith("objects/") or "\\" in path
            or path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise ValueError("invalid reader asset object path")
    return path


def validate_storage_path(path: str) -> str:
    """Validate immutable Reader objects, chapter objects, or lifecycle staging."""
    if isinstance(path, str) and path.startswith("staging/"):
        if "\\" in path or path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
            raise ValueError("invalid reader staging path")
        return path
    if isinstance(path, str) and path.startswith("ebook-chapters/"):
        validate_object_path(path[len("ebook-chapters/"):])
        return path
    return validate_object_path(path)


def validate_chapter_manifest_path(path: str) -> str:
    """Accept chapter manifests stored under the legacy ebook-chapters prefix."""
    if isinstance(path, str) and path.startswith("ebook-chapters/"):
        validate_object_path(path[len("ebook-chapters/"):])
        return path
    return validate_object_path(path)


def validate_chapter_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("version") != CHAPTER_MANIFEST_VERSION:
        raise ValueError("unsupported EPUB chapter manifest version")
    chapters = manifest.get("chapters")
    if manifest.get("kind") != "epub-chapters" or not isinstance(chapters, list) or not chapters:
        raise ValueError("invalid EPUB chapter manifest")
    seen = set()
    for index, chapter in enumerate(chapters, 1):
        if not isinstance(chapter, dict) or chapter.get("index") != index:
            raise ValueError("EPUB chapters must be ordered")
        path = chapter.get("path")
        if (not isinstance(path, str) or path.startswith("/") or "\\" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or not path.startswith("chapters/") or path in seen):
            raise ValueError("invalid EPUB chapter path")
        seen.add(path)
        if not isinstance(chapter.get("title", ""), str) or not isinstance(chapter.get("bytes"), int) or chapter["bytes"] <= 0:
            raise ValueError("invalid EPUB chapter metadata")
        if not re.fullmatch(r"[0-9a-f]{64}", str(chapter.get("sha256", ""))):
            raise ValueError("invalid EPUB chapter digest")
    toc = manifest.get("toc")
    if toc is not None:
        if not isinstance(toc, list):
            raise ValueError("invalid EPUB TOC")
        for entry in toc:
            if (not isinstance(entry, dict) or not isinstance(entry.get("title"), str)
                    or not entry["title"].strip()
                    or type(entry.get("chapter")) is not int
                    or not 1 <= entry["chapter"] <= len(chapters)
                    or type(entry.get("depth")) is not int or not 0 <= entry["depth"] <= 64
                    or not isinstance(entry.get("fragment"), str)):
                raise ValueError("invalid EPUB TOC entry")
    search_index = manifest.get("search_index")
    if search_index is not None:
        if not isinstance(search_index, dict):
            raise ValueError("invalid EPUB search index metadata")
        search_path = search_index.get("path")
        if (not isinstance(search_path, str) or search_path != "epub-search-index.json.gz"
                or search_index.get("bytes", 0) <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(search_index.get("sha256", "")))):
            raise ValueError("invalid EPUB search index metadata")
    if manifest.get("fallback") is not None:
        validate_object_path(manifest["fallback"])
    return manifest


def source_conversion_contract(repo: str, path: str, extension: str, source_bytes: int = 0):
    if extension in CONVERTIBLE_EXTENSIONS:
        return CONVERTIBLE_EXTENSIONS[extension]
    if extension == "pdf" and known_gbk_pdf(repo, path):
        return GBK_PDF_CONTRACT
    if extension == "pdf" and source_password(repo, path):
        return PROTECTED_PDF_CONTRACT
    return None


def bucket_conversion_contract(repo: str, path: str, extension: str, source_bytes: int = 0):
    """Return the contract used when moving every static Reader format to the bucket."""
    if extension == "pdf":
        special = source_conversion_contract(repo, path, extension, source_bytes)
        if special:
            return special
        return ("pdf-staging-v1", "pdf", "document.pdf")
    if extension in BUCKET_NATIVE_EXTENSIONS:
        return BUCKET_NATIVE_EXTENSIONS[extension]
    if extension in BUCKET_NATIVE_MEDIA_EXTENSIONS:
        return BUCKET_NATIVE_MEDIA_EXTENSIONS[extension]
    if extension in {"xls", "xlsx", "csv", "ods"}:
        return (SPREADSHEET_HTML_PROFILE, "html", "document.html")
    contract = source_conversion_contract(repo, path, extension, source_bytes)
    if contract and contract[1] in {"docx", "html", "text", "markdown", "image", "pdf", "foliate", "epub", "swf", "audio", "video"}:
        return contract
    return None


def conversion_dependencies(extension: str, reader_mode: str) -> dict:
    """Describe local-only conversion inputs that never become bucket objects."""
    if extension == "chm":
        return {"kind": "chm-expanded-html", "lifetime": "runner", "final_mode": reader_mode}
    if extension in {"mobi", "azw3", "fb2"} and reader_mode == "foliate":
        return {"kind": "temporary-epub", "lifetime": "runner", "final_mode": reader_mode}
    return {"kind": "none", "lifetime": "none", "final_mode": reader_mode}


def empty_manifest() -> dict:
    return {"version": MANIFEST_VERSION, "files": {}}


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("unsupported reader manifest version")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("reader manifest files must be an object")
    if "orphans" in manifest and not isinstance(manifest["orphans"], dict):
        raise ValueError("reader manifest orphans must be an object")
    for key, entry in manifest["files"].items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ValueError("reader manifest file entries must be objects")
        status = entry.get("status", "ready")
        if status not in {"ready", "failed"}:
            raise ValueError("reader manifest file entry has invalid status")
        if status == "ready":
            validate_storage_path(entry.get("path"))
            if "chapter_manifest" in entry:
                validate_chapter_manifest_path(entry["chapter_manifest"])
            if "fallback_path" in entry:
                validate_object_path(entry["fallback_path"])
            if entry.get("chapter_bucket") not in {None, READER_ASSETS_BUCKET}:
                raise ValueError("invalid chapter bucket")
            if "chapter_manifest" in entry and not entry["chapter_manifest"].endswith("/chapter-manifest.json"):
                raise ValueError("invalid chapter manifest path")
            if "chapter_manifest" in entry and entry.get("reader_mode") not in {"epub", "foliate", "pdf"}:
                raise ValueError("chapter manifest requires EPUB or PDF reader mode")
            if "reader_mode" in entry and entry.get("reader_mode") not in {"pdf", "epub", "foliate", "docx", "html", "text", "markdown", "image", "audio", "video", "swf"}:
                raise ValueError("reader manifest ready entry has invalid reader mode")
            if "bytes" in entry and (not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0):
                raise ValueError("reader manifest ready entry has invalid byte count")
            if "sha256" in entry and not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
                raise ValueError("reader manifest ready entry has invalid digest")
    for path, entry in manifest.get("orphans", {}).items():
        validate_object_path(path)
        if isinstance(entry, dict) and entry.get("path") not in {None, path}:
            raise ValueError("reader asset orphan path mismatch")
    return manifest


def decode_search_payload(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    repos, folders, records = data.get("rp", []), data.get("fd", []), []
    for item in data.get("rc", []):
        if not isinstance(item, list) or len(item) < 6:
            continue
        repo = repos[item[0]] if isinstance(item[0], int) and 0 <= item[0] < len(repos) else ""
        folder = folders[item[3]] if isinstance(item[3], int) and 0 <= item[3] < len(folders) else []
        records.append({"Repo": repo, "File": item[1], "Extension": item[2], "Folder": folder, "Size": item[4]})
    return records


def relative_path(record: dict) -> str:
    name, extension = str(record.get("File") or ""), str(record.get("Extension") or "")
    filename = f"{name}.{extension}" if extension else name
    folders = [str(part) for part in (record.get("Folder") or [])]
    return posixpath.join(*folders, filename) if folders else filename


def asset_key(repo: str, path: str) -> str:
    return f"{repo}\0{path}"


def source_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path, safe='/')}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(data: dict, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    else:
        text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")
