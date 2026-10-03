#!/usr/bin/env python3
"""Download and convert one Reader Assets queue into a publishable bundle."""

import argparse
import concurrent.futures
import base64
import copy
import csv
import email.policy
import hashlib
import html
import io
import mimetypes
import json
import re
import shutil
import subprocess
import tempfile
import threading
import traceback
import time
import http.client
import urllib.error
import urllib.request
import urllib.parse
import zipfile
import os
import posixpath
import xml.etree.ElementTree as ET
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from lxml import html as lxml_html

import bleach
import tinycss2
import esprima
from bleach.css_sanitizer import CSSSanitizer
from PIL import Image, ImageSequence

try:
    from .reader_assets import (
        EPUB_CHAPTER_BUNDLE_DIR, EPUB_CHAPTER_PROFILE, GBK_PDF_CONTRACT, PASSWORD_RE, canonical_json, load_json, needs_epub_chapters,
        object_profile_path, reusable_object_key, source_password, validate_storage_path,
    )
except ImportError:
    from reader_assets import (
        EPUB_CHAPTER_BUNDLE_DIR, EPUB_CHAPTER_PROFILE, GBK_PDF_CONTRACT, PASSWORD_RE, canonical_json, load_json, needs_epub_chapters,
        object_profile_path, reusable_object_key, source_password, validate_storage_path,
    )

try:
    from . import shared
except ImportError:
    import shared

MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_HTML_RESOURCE_BYTES = 16 * 1024 * 1024
MAX_HTML_RESOURCE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_HTML_RESOURCES = 64
MAX_EPUB_MEMBERS = 10000
MAX_EPUB_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_EPUB_MEMBER_BYTES = 256 * 1024 * 1024
MAX_MHTML_RESOURCE_BYTES = 128 * 1024 * 1024
MAX_MHTML_SOURCE_BYTES = 256 * 1024 * 1024
MAX_CHM_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
HTML_TAGS = {
    "a", "abbr", "address", "article", "aside", "b", "bdi", "bdo", "blockquote", "body", "br",
    "caption", "cite", "code", "col", "colgroup", "dd", "del", "details", "dfn", "div", "dl", "dt",
    "em", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header",
    "hgroup", "hr", "html", "i", "img", "ins", "kbd", "li", "link", "main", "mark", "meta", "nav",
    "ol", "p", "picture", "pre", "q", "rb", "rp", "rt", "rtc", "ruby", "s", "samp", "section", "small", "source",
    "span", "strong", "style", "sub", "summary", "sup", "table", "tbody", "td", "tfoot", "th", "thead",
    "time", "title", "tr", "u", "ul", "var", "wbr",
}
GLOBAL_HTML_ATTRIBUTES = {"class", "dir", "id", "lang", "title"}
TAG_HTML_ATTRIBUTES = {
    "a": {"href", "name"}, "col": {"span", "width"}, "colgroup": {"span", "width"},
    "img": {"alt", "height", "src", "width"}, "link": {"href", "media", "rel", "type"},
    "meta": {"charset"}, "source": {"src", "type"}, "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan", "scope"}, "time": {"datetime"},
}
CSS_SANITIZER = CSSSanitizer()
SVG_TAGS = {
    "circle", "clippath", "defs", "desc", "ellipse", "g", "image", "line", "lineargradient",
    "mask", "path", "pattern", "polygon", "polyline", "radialgradient", "rect", "stop", "svg",
    "symbol", "text", "title", "tspan", "use",
}
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
OLE_SIGNATURE = bytes.fromhex("d0cf11e0a1b11ae1")
MIN_PAGE_CONTENT_RATIO = 0.0005
COMMAND_TIMEOUT_SECONDS = int(os.environ.get("READER_CONVERSION_COMMAND_TIMEOUT", "120"))
SPREADSHEET_RENDER_COMMAND_TIMEOUT_SECONDS = max(
    1800, int(os.environ.get("READER_SPREADSHEET_RENDER_TIMEOUT", str(COMMAND_TIMEOUT_SECONDS)))
)
SPREADSHEET_FULL_PAGE_MAX_PIXELS = 24_000_000
SPREADSHEET_FULL_PAGE_MAX_EDGE = 16_000
EPUB_COMMAND_TIMEOUT_SECONDS = max(1800, int(os.environ.get(
    "READER_EPUB_COMMAND_TIMEOUT", str(COMMAND_TIMEOUT_SECONDS),
)))
CHM_COMMAND_TIMEOUT_SECONDS = max(900, int(os.environ.get(
    "READER_CHM_COMMAND_TIMEOUT", str(COMMAND_TIMEOUT_SECONDS),
)))
DJVU_COMMAND_TIMEOUT_SECONDS = max(600, int(os.environ.get(
    "READER_DJVU_COMMAND_TIMEOUT", str(COMMAND_TIMEOUT_SECONDS),
)))
MEDIA_COMMAND_TIMEOUT_SECONDS = int(os.environ.get(
    "READER_MEDIA_COMMAND_TIMEOUT", str(max(7200, COMMAND_TIMEOUT_SECONDS)),
))
POSTSCRIPT_COMMAND_TIMEOUT_SECONDS = int(os.environ.get(
    "READER_POSTSCRIPT_COMMAND_TIMEOUT", str(max(600, COMMAND_TIMEOUT_SECONDS)),
))
CONVERSION_WORKERS = max(1, int(os.environ.get("READER_CONVERSION_WORKERS", "1")))
READER_ASSETS_REPO = os.environ.get("READER_ASSETS_REPO", "vomebook/Reader-Assets")
CAJ2PDF_DIR = Path(os.environ.get("CAJ2PDF_DIR", "/opt/caj2pdf"))
ARTIFACT_LOCKS = {}
ARTIFACT_LOCKS_GUARD = threading.Lock()


def download_source(url: str, target: Path, *, max_bytes: int = MAX_SOURCE_BYTES) -> tuple[str, int]:
    for attempt in range(3):
        offset = target.stat().st_size if target.exists() else 0
        headers = {"User-Agent": "VoiceOfML-Reader-Assets/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        digest, size = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                resumed = bool(offset and response.status == 206)
                if not offset and response.status == 206:
                    raise http.client.IncompleteRead(b"unexpected partial response")
                if resumed:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                    if not match or int(match.group(1)) != offset:
                        target.unlink(missing_ok=True)
                        raise http.client.IncompleteRead(b"invalid Content-Range")
                if offset and not resumed:
                    offset = 0
                mode = "ab" if resumed else "wb"
                if not resumed:
                    digest = hashlib.sha256()
                if resumed:
                    with target.open("rb") as existing:
                        while chunk := existing.read(1024 * 1024):
                            digest.update(chunk)
                    size = offset
                with target.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise RuntimeError("download exceeds size limit")
                        digest.update(chunk)
                        output.write(chunk)
            return digest.hexdigest(), size
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 429} and exc.code < 500:
                raise
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            error = exc
        if attempt == 2:
            raise error
        time.sleep(attempt + 1)
    raise RuntimeError("source download retry limit reached")


def decode_html_source(source: Path) -> str:
    data = source.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    probe = data[:8192].decode("latin-1")
    match = re.search(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", probe, re.IGNORECASE)
    declared = None
    if match:
        declared = match.group(1).lower().replace("_", "-")
        declared = {
            "gb2312": "gb18030", "gb-2312": "gb18030", "gbk": "gb18030",
            "x-gbk": "gb18030", "x-sjis": "shift-jis", "windows-31j": "shift-jis",
            "ks_c_5601-1987": "euc-kr", "euc-cn": "gb18030", "x-euc-tw": "big5",
        }.get(declared, declared)
    if declared:
        try:
            return data.decode(declared)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        # Valid UTF-8 is unambiguous; legacy Chinese pages fall through to
        # the GB18030-first candidates below.
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    candidates = [encoding for encoding in (declared, "gb18030", "big5",
                  "shift-jis", "euc-kr", "latin-1") if encoding]
    best_text, best_score = "", float("-inf")
    for encoding in dict.fromkeys(candidates):
        try:
            text = data.decode(encoding, errors="replace")
        except (LookupError, UnicodeError):
            continue
        replacement = text.count("\ufffd")
        controls = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text))
        readable = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ffA-Za-z0-9]", text))
        score = readable - replacement * 80 - controls * 20
        if encoding == declared:
            score += 20
        if score > best_score:
            best_text, best_score = text, score
    return best_text


def expand_document_writes(document: str) -> tuple[str, bool]:
    """Extract top-level literal writes without executing or retaining script code."""
    if "document" not in document or "write" not in document:
        return document, False
    try:
        tree = esprima.parseScript(document)
    except (esprima.Error, RecursionError):
        return document, False

    def literal(node):
        if node.type == "Literal" and isinstance(node.value, str):
            return node.value
        if node.type == "BinaryExpression" and node.operator == "+":
            left, right = literal(node.left), literal(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    output = []
    for statement in tree.body:
        if statement.type != "ExpressionStatement":
            continue
        call = statement.expression
        if call.type != "CallExpression":
            continue
        callee = call.callee
        if (callee.type != "MemberExpression" or callee.computed
                or callee.object.type != "Identifier" or callee.object.name != "document"
                or callee.property.name not in {"write", "writeln"}):
            continue
        values = [literal(argument) for argument in call.arguments]
        if values and all(value is not None for value in values):
            output.append("".join(values) + ("\n" if callee.property.name == "writeln" else ""))
    return ("".join(output), True) if output else (document, False)


def inline_html_resources(source: Path, source_url: str, work: Path) -> Path:
    """Inline safe same-tree images and stylesheets before the HTML is published."""
    text = decode_html_source(source)
    text = re.sub(
        r"(charset\s*=\s*)([\"']?)[^\s\"'/>;]+\2",
        lambda match: match.group(1) + ('"utf-8"' if match.group(2) else "utf-8"),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(<\?xml\b[^>]*\bencoding\s*=\s*)[\"'][^\"']+[\"']", r'\1"utf-8"', text, flags=re.IGNORECASE)
    text = re.sub(
        r"<meta\b(?=[^>]*\bhttp-equiv\b)(?=[^>]*\bcharset\s*=)[^>]*>",
        '<meta charset="utf-8">', text, flags=re.IGNORECASE,
    )
    if not re.search(r"<meta\b[^>]*\bcharset\s*=", text, re.IGNORECASE):
        text = re.sub(r"(<head\b[^>]*>)", r'\1<meta charset="utf-8">', text, count=1, flags=re.IGNORECASE)
        if not re.search(r"<meta\b[^>]*\bcharset\s*=", text, re.IGNORECASE):
            text = '<meta charset="utf-8">' + text
    base = source_url.rsplit("/", 1)[0] + "/"
    root = urllib.parse.urlsplit(base)
    root_path = root.path.rstrip("/") + "/"
    total = 0
    count = 0
    cache = {}

    def load_resource(raw: str, allowed: set[str]):
        nonlocal total, count
        raw = html.unescape(raw).strip()
        if not raw or raw.startswith(("#", "data:", "mailto:", "javascript:")):
            return None
        parsed = urllib.parse.urlsplit(urllib.parse.urljoin(base, raw))
        if (parsed.scheme != "https" or parsed.netloc != root.netloc or parsed.query or
                not parsed.path.startswith(root_path) or parsed.path == root_path):
            return None
        path = parsed.path
        if path in cache:
            return cache[path]
        if count >= MAX_HTML_RESOURCES:
            return None
        suffix = Path(path).suffix.lower()
        mime = mimetypes.guess_type(path)[0] or ""
        if suffix not in allowed and mime not in allowed:
            return None
        target = work / "html-resources" / str(count)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            digest, size = download_source(
                urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
                target, max_bytes=MAX_HTML_RESOURCE_BYTES,
            )
        except Exception:
            target.unlink(missing_ok=True)
            return None
        if size > MAX_HTML_RESOURCE_BYTES or total + size > MAX_HTML_RESOURCE_TOTAL_BYTES:
            target.unlink(missing_ok=True)
            return None
        data = target.read_bytes()
        total += size
        count += 1
        encoded = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        cache[path] = encoded
        return encoded

    def image(match):
        value = match.group(2)
        encoded = load_resource(value, {"image/gif", "image/jpeg", "image/png", "image/webp", ".gif", ".jpg", ".jpeg", ".png", ".webp"})
        return match.group(1) + (encoded or value) + match.group(3)

    text = re.sub(r"(\b(?:src|data-src)\s*=\s*[\"'])([^\"']+)([\"'])", image, text, flags=re.IGNORECASE)

    # Replace local stylesheet links with their downloaded, resource-checked CSS.
    def stylesheet_tag(match):
        value = match.group(2)
        raw = html.unescape(value).strip()
        parsed = urllib.parse.urlsplit(urllib.parse.urljoin(base, raw))
        if parsed.path not in cache:
            load_resource(raw, {"text/css", ".css"})
        css_data = cache.get(parsed.path)
        if not css_data:
            return ""
        try:
            css = base64.b64decode(css_data.split(",", 1)[1]).decode("utf-8", "replace")
        except Exception:
            return ""
        css = re.sub(r"@import[^;]+;|url\s*\([^)]*\)", "", css, flags=re.IGNORECASE)
        return f"<style>{css}</style>"

    text = re.sub(r"<link\b[^>]*\brel\s*=\s*[\"']stylesheet[\"'][^>]*\bhref\s*=\s*([\"'])([^\"']+)\1[^>]*>\s*", stylesheet_tag, text, flags=re.IGNORECASE)
    text = re.sub(r"<link\b[^>]*\bhref\s*=\s*([\"'])([^\"']+)\1[^>]*\brel\s*=\s*[\"']stylesheet[\"'][^>]*>\s*", stylesheet_tag, text, flags=re.IGNORECASE)
    text = sanitize_html(text)
    output = work / "inlined.html"
    output.write_text(text, encoding="utf-8")
    return output


def inline_local_html_resources(document: str, root: Path, base: Path, *, allow_relative: bool = False) -> str:
    """Inline local images and stylesheets before sanitizing a generated HTML file."""
    cache = {}

    def resolve_resource(raw: str) -> Path | None:
        value = urllib.parse.unquote(raw)
        path = (base / value).resolve()
        root_path = root.resolve()
        try:
            relative = path.relative_to(root_path)
        except ValueError:
            return None
        if path.is_file():
            return path
        current = root_path
        for part in relative.parts:
            try:
                matches = [child for child in current.iterdir() if child.name.casefold() == part.casefold()]
            except OSError:
                return None
            if len(matches) != 1:
                return None
            current = matches[0]
        return current if current.is_file() else None

    def script_body(match):
        expanded, was_static = expand_document_writes(match.group(1))
        return expanded if was_static else ""

    # Bleach's strip mode removes a script element but may keep its text. Drop
    # dynamic script bodies entirely, while retaining safe HTML produced by
    # literal document.write calls.
    document = re.sub(
        r"<script\b[^>]*>(.*?)</script\s*>", script_body, document,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def resource(raw: str, *, css: bool = False) -> str | None:
        value = html.unescape(raw).strip().split("#", 1)[0]
        if not value or re.match(r"^(?:data:|https?:|//|#|javascript:)", value, re.IGNORECASE):
            return None
        path = resolve_resource(value)
        if path is None:
            return None
        try:
            if path.stat().st_size > MAX_HTML_RESOURCE_BYTES:
                return None
        except OSError:
            return None
        key = str(path)
        if key not in cache:
            data = path.read_bytes()
            if css:
                cache[key] = f"<style>{sanitize_css(data.decode('utf-8', 'replace'))}</style>"
            else:
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if mime not in {"image/gif", "image/jpeg", "image/png", "image/webp", "image/svg+xml"}:
                    return None
                cache[key] = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        return cache[key]

    def image(match):
        value = resource(match.group(2))
        return match.group(1) + (value or match.group(2)) + match.group(3)

    document = re.sub(r"(\b(?:src|data-src|poster)\s*=\s*[\"'])([^\"']+)([\"'])", image, document, flags=re.IGNORECASE)

    def stylesheet(match):
        value = resource(match.group(2), css=True)
        return value or ""

    document = re.sub(
        r"<link\b[^>]*\brel\s*=\s*[\"']stylesheet[\"'][^>]*\bhref\s*=\s*([\"'])([^\"']+)\1[^>]*>\s*",
        stylesheet, document, flags=re.IGNORECASE,
    )
    return sanitize_html(document, allow_relative=allow_relative)


def download_existing(url: str, target: Path, expected_sha256: str) -> None:
    temporary = target.with_name(f".{target.name}.download")
    try:
        digest, _ = download_source(url, temporary)
        if digest != expected_sha256:
            raise RuntimeError("reusable reader artifact digest mismatch")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    return shared.hash_file(path)[0]


def artifact_lock(path: Path) -> threading.Lock:
    with ARTIFACT_LOCKS_GUARD:
        return ARTIFACT_LOCKS.setdefault(str(path), threading.Lock())


class ReaderConversionTimeout(RuntimeError):
    pass


class ReaderConversionCommandError(RuntimeError):
    pass


def conversion_error_class(exc: Exception) -> str:
    if isinstance(exc, ReaderConversionTimeout):
        return "conversion-timeout"
    if isinstance(exc, ReaderConversionCommandError):
        return "conversion-command-failed"
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead)):
        return "source-download-failed"
    return type(exc).__name__


def run_checked(command: list[str], *, timeout_seconds: int | None = None, cwd: Path | None = None) -> None:
    timeout_seconds = COMMAND_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds, cwd=cwd,
            env={key: value for key, value in os.environ.items() if key not in {
                "HF_TOKEN", "PAGES_TOKEN", "GH_PAT", "READER_CONVERSION_PASSWORD",
            }},
        )
    except subprocess.TimeoutExpired as exc:
        raise ReaderConversionTimeout(f"conversion command timed out: {Path(command[0]).name}") from exc
    if result.returncode:
        detail = result.stderr.strip().replace("\n", " ")[-500:]
        raise ReaderConversionCommandError(f"conversion command failed: {Path(command[0]).name}: {detail}")


def command_output(command: list[str], *, timeout_seconds: int | None = None) -> str:
    timeout_seconds = COMMAND_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds,
            env={key: value for key, value in os.environ.items() if key not in {
                "HF_TOKEN", "PAGES_TOKEN", "GH_PAT", "READER_CONVERSION_PASSWORD",
            }},
        )
    except subprocess.TimeoutExpired as exc:
        raise ReaderConversionTimeout(f"validation command timed out: {Path(command[0]).name}") from exc
    if result.returncode:
        raise ReaderConversionCommandError(f"validation command failed: {Path(command[0]).name}: {result.stderr[-500:]}")
    return result.stdout


def media_probe(path: Path) -> dict:
    try:
        return json.loads(command_output([
            "ffprobe", "-v", "error", "-show_format", "-show_streams",
            "-print_format", "json", str(path),
        ]))
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("media probe returned invalid data") from exc


def source_media_mode(path: Path) -> str:
    probe = media_probe(path)
    streams = probe.get("streams") if isinstance(probe, dict) else None
    if not isinstance(streams, list):
        raise RuntimeError("source media has no stream metadata")
    if any(stream.get("codec_type") == "video" for stream in streams):
        return "video"
    if any(stream.get("codec_type") == "audio" for stream in streams):
        return "audio"
    raise RuntimeError("source media has no audio or video stream")


def validate_media_output(path: Path, reader_mode: str) -> None:
    probe = media_probe(path)
    streams = probe.get("streams") if isinstance(probe, dict) else None
    media_format = probe.get("format") if isinstance(probe, dict) else None
    if not isinstance(streams, list) or not isinstance(media_format, dict):
        raise RuntimeError("media output has no stream metadata")
    try:
        duration = float(media_format.get("duration") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("media output has invalid duration") from exc
    if not 0 < duration <= 24 * 60 * 60:
        raise RuntimeError("media output duration is outside limits")
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    unexpected = [stream for stream in streams if stream.get("codec_type") not in {"audio", "video"}]
    format_names = set(str(media_format.get("format_name") or "").split(","))
    if unexpected:
        raise RuntimeError("media output contains unsupported streams")
    if reader_mode == "audio":
        if len(audio) != 1 or video or audio[0].get("codec_name") != "mp3" or "mp3" not in format_names:
            raise RuntimeError("conversion output is not compatible MP3 audio")
        return
    if reader_mode != "video" or len(video) != 1 or len(audio) > 1:
        raise RuntimeError("conversion output has invalid video streams")
    width, height = int(video[0].get("width") or 0), int(video[0].get("height") or 0)
    if (video[0].get("codec_name") != "h264" or video[0].get("pix_fmt") != "yuv420p" or "mp4" not in format_names
            or not 0 < width <= 1920 or not 0 < height <= 1080
            or (audio and audio[0].get("codec_name") != "aac")):
        raise RuntimeError("conversion output is not compatible H.264/AAC video")


def embedded_pdf_fonts(path: Path) -> list[str]:
    fonts = command_output(["pdffonts", str(path)]).splitlines()[2:]
    return [line for line in fonts if len(line.split()) >= 6 and line.split()[-5].lower() == "yes"]


def outline_pdf_fonts(path: Path, work: Path) -> None:
    rewritten = work / "outlined-fonts.pdf"
    run_checked([
        "gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.7",
        "-dNoOutputFonts=true",
        f"-sOutputFile={rewritten}", str(path),
    ])
    if not rewritten.is_file():
        raise RuntimeError("Ghostscript produced no PDF")
    shutil.move(rewritten, path)


def convert_tiff(source: Path, target: Path, work: Path) -> None:
    pages = work / "tiff-pages"
    pages.mkdir()
    outputs = []
    with Image.open(source) as image:
        for index, frame in enumerate(ImageSequence.Iterator(image), start=1):
            output = pages / f"{index:08d}.pdf"
            converted = frame.copy().convert("RGB")
            converted.save(output, "PDF", resolution=150.0)
            converted.close()
            outputs.append(output)
    if not outputs:
        raise RuntimeError("TIFF has no frames")
    if len(outputs) == 1:
        shutil.move(outputs[0], target)
    else:
        run_checked(["pdfunite", *(str(output) for output in outputs), str(target)])


def mhtml_to_html(source: Path, target: Path) -> None:
    message = BytesParser(policy=email.policy.default).parsebytes(source.read_bytes())
    parts = list(message.walk()) if message.is_multipart() else [message]
    html_part = next((part for part in parts if part.get_content_type() == "text/html"), None)
    if html_part is None:
        raise RuntimeError("CHM MHTML has no HTML body")
    payload = html_part.get_payload(decode=True) or b""
    charset = html_part.get_content_charset() or "utf-8"
    try:
        document = payload.decode(charset)
    except (LookupError, UnicodeDecodeError):
        document = payload.decode("gb18030", "replace")
    base = str(html_part.get("Content-Location") or "")

    def safe_join(root: str, value: str) -> str:
        root, value = root.replace("\\", "/"), value.replace("\\", "/")
        root = re.sub(r"^file://([A-Za-z]:/)", r"file:///\1", root)
        if re.match(r"^[A-Za-z]:/", root):
            root = "file:///" + root
        try:
            return urllib.parse.urljoin(root, value)
        except ValueError:
            return ""
    resources = {}
    resource_bytes = 0
    for part in parts:
        if part is html_part or part.is_multipart():
            continue
        data = part.get_payload(decode=True)
        if not data:
            continue
        resource_bytes += len(data)
        if resource_bytes > MAX_MHTML_RESOURCE_BYTES:
            raise RuntimeError("MHTML embedded resources exceed size limit")
        encoded = f"data:{part.get_content_type()};base64,{base64.b64encode(data).decode('ascii')}"
        location = str(part.get("Content-Location") or "")
        content_id = str(part.get("Content-ID") or "").strip("<>")
        if location:
            resources[location] = encoded
            joined = safe_join(base, location)
            if joined:
                resources[joined] = encoded
        if content_id:
            resources[f"cid:{content_id}"] = encoded

    def inline_resource(match):
        value = html.unescape(match.group(2)).strip()
        replacement = resources.get(value) or resources.get(safe_join(base, value))
        return match.group(1) + (replacement or value) + match.group(3)

    document = re.sub(
        r"(\b(?:src|href|poster)\s*=\s*[\"'])([^\"']+)([\"'])",
        inline_resource,
        document,
        flags=re.IGNORECASE,
    )
    document = sanitize_chm_html(document)
    document = re.sub(r"<meta\b[^>]*charset[^>]*>", "", document, flags=re.IGNORECASE)
    document = re.sub(r"(<head\b[^>]*>)", r'\1<meta charset="utf-8">', document, count=1, flags=re.IGNORECASE)
    target.write_text(document, encoding="utf-8")


def css_token_has_external_url(token) -> bool:
    if token.type == "url":
        value = token.value
    elif token.type == "function" and token.name.lower() == "url":
        value = tinycss2.serialize(token.arguments).strip(" \t\r\n\"'")
    else:
        children = getattr(token, "arguments", None) or getattr(token, "content", None) or []
        return any(css_token_has_external_url(child) for child in children)
    normalized = re.sub(r"[\x00-\x20]+", "", value).lower()
    parsed = urllib.parse.urlsplit(normalized)
    return bool(parsed.scheme or parsed.netloc or normalized.startswith("//"))


def sanitize_css_declarations(document: str) -> str:
    sanitized = CSS_SANITIZER.sanitize_css(document)
    declarations = []
    for declaration in tinycss2.parse_declaration_list(sanitized, skip_comments=True, skip_whitespace=True):
        if declaration.type == "declaration" and not any(css_token_has_external_url(token) for token in declaration.value):
            declarations.append(declaration)
    return tinycss2.serialize(declarations)


def sanitize_css(document: str) -> str:
    output = []
    for rule in tinycss2.parse_stylesheet(document, skip_comments=True, skip_whitespace=True):
        if rule.type != "qualified-rule":
            continue
        selector = tinycss2.serialize(rule.prelude).strip()
        declarations = sanitize_css_declarations(tinycss2.serialize(rule.content))
        declarations = re.sub(r"\bexpression\s*\([^)]*\)", "", declarations, flags=re.IGNORECASE)
        if selector and declarations.strip():
            output.append(f"{selector}{{{declarations}}}")
    return "".join(output)


def safe_embedded_url(name: str, value: str, *, allow_relative: bool) -> bool:
    value = html.unescape(value).strip()
    normalized = re.sub(r"[\x00-\x20]+", "", value).lower()
    if name == "src" and re.match(r"^data:image/(?:gif|jpeg|png|webp);base64,", normalized):
        return True
    if name == "href" and normalized.startswith("#"):
        return True
    parsed = urllib.parse.urlsplit(normalized)
    return allow_relative and not parsed.scheme and not parsed.netloc and not normalized.startswith("//")


def sanitize_html(document: str, *, allow_relative: bool = False) -> str:
    # HTML accepts numeric references without a semicolon; Bleach otherwise
    # escapes the ampersand and turns a source space into visible entity text.
    document = re.sub(r"&#[xX][0-9a-fA-F]+(?![0-9a-fA-F;])|&#[0-9]+(?![0-9;])",
                      lambda match: match.group(0) + ";", document)
    styles = []

    def extract_style(match):
        token = f"READER_STYLE_{len(styles)}_{hashlib.sha256(match.group(0).encode()).hexdigest()}"
        styles.append((token, f"<style>{sanitize_css(match.group(1))}</style>"))
        return token

    document = re.sub(
        r"<style\b[^>]*>(.*?)</style\s*>", extract_style, document,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def allowed_attribute(tag: str, name: str, value: str) -> bool:
        if name not in GLOBAL_HTML_ATTRIBUTES | TAG_HTML_ATTRIBUTES.get(tag, set()):
            return False
        if name not in {"href", "src"}:
            return True
        return safe_embedded_url(name, value, allow_relative=allow_relative)

    cleaner = bleach.Cleaner(
        tags=HTML_TAGS,
        attributes=allowed_attribute,
        protocols={"data", "http", "https"},
        css_sanitizer=CSS_SANITIZER,
        strip=True,
        strip_comments=True,
    )
    cleaned = cleaner.clean(document)
    for token, style in styles:
        cleaned = cleaned.replace(token, style)
    return cleaned


def sanitize_xml_document(document: str) -> str:
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise RuntimeError("EPUB XML content is malformed") from exc
    root_kind = root.tag.rsplit("}", 1)[-1].lower()
    allowed_tags = SVG_TAGS if root_kind == "svg" else {tag.lower() for tag in HTML_TAGS if tag != "style"}

    def clean(parent):
        for child in list(parent):
            local_tag = child.tag.rsplit("}", 1)[-1].lower() if isinstance(child.tag, str) else ""
            if local_tag not in allowed_tags:
                if (root_kind != "svg"
                        and local_tag not in {"script", "style", "noscript", "iframe", "object", "embed",
                                              "applet", "form", "input", "button", "select", "textarea",
                                              "template", "base", "svg", "math",
                                              "font", "center", "big", "tt", "strike"}):
                    # Malformed books can wrap prose in passive custom tags.
                    # Preserve their text/anchors in an inert HTML container.
                    namespace = child.tag.rsplit("}", 1)[0] + "}" if "}" in child.tag else ""
                    child.tag = namespace + "span"
                    clean(child)
                    continue
                # Removing an element must not remove the following text node.
                # Legacy presentational containers are unwrapped, active ones
                # are discarded together with their contents.
                index = list(parent).index(child)
                if root_kind != "svg" and local_tag in {"font", "center", "big", "tt", "strike"}:
                    clean(child)
                    if child.text:
                        if index:
                            previous = parent[index - 1]
                            previous.tail = (previous.tail or "") + child.text
                        else:
                            parent.text = (parent.text or "") + child.text
                    for grandchild in list(child):
                        parent.insert(index, grandchild)
                        index += 1
                if child.tail:
                    if index:
                        previous = parent[index - 1]
                        previous.tail = (previous.tail or "") + child.tail
                    else:
                        parent.text = (parent.text or "") + child.tail
                parent.remove(child)
                continue
            clean(child)
        local_tag = parent.tag.rsplit("}", 1)[-1].lower() if isinstance(parent.tag, str) else ""
        for attribute, value in list(parent.attrib.items()):
            name = attribute.rsplit("}", 1)[-1].lower()
            if name.startswith("on"):
                del parent.attrib[attribute]
                continue
            if root_kind == "svg":
                if name == "style":
                    parent.attrib[attribute] = sanitize_css_declarations(value)
                elif name in {"href", "src"} and not safe_embedded_url(name, value, allow_relative=True):
                    del parent.attrib[attribute]
                elif re.search(r"(?:javascript|vbscript|https?:|//)", html.unescape(value), re.IGNORECASE):
                    del parent.attrib[attribute]
                continue
            if name not in GLOBAL_HTML_ATTRIBUTES | TAG_HTML_ATTRIBUTES.get(local_tag, set()):
                del parent.attrib[attribute]
            elif name == "style":
                parent.attrib[attribute] = sanitize_css_declarations(value)
            elif name in {"href", "src"} and not safe_embedded_url(name, value, allow_relative=True):
                del parent.attrib[attribute]

    clean(root)
    return ET.tostring(root, encoding="unicode")


sanitize_chm_html = sanitize_html


def sanitize_chm_epub(path: Path, work: Path) -> None:
    rewritten = work / "sanitized-chm.epub"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(rewritten, "w") as target:
        infos = sorted(source.infolist(), key=lambda info: info.filename != "mimetype")
        if (len(infos) > MAX_EPUB_MEMBERS or any(info.file_size > MAX_EPUB_MEMBER_BYTES for info in infos)
                or sum(info.file_size for info in infos) > MAX_EPUB_EXPANDED_BYTES):
            raise RuntimeError("EPUB expanded content exceeds limits")
        for info in infos:
            data = source.read(info.filename)
            suffix = Path(info.filename).suffix.lower()
            if suffix in {".htm", ".html", ".xhtml", ".css", ".svg"}:
                text = data.decode("utf-8", "replace")
                if suffix == ".css":
                    text = sanitize_css(text)
                else:
                    text = sanitize_xml_document(text)
                data = text.encode("utf-8")
            if info.filename == "mimetype":
                info.compress_type = zipfile.ZIP_STORED
            target.writestr(info, data)
    shutil.move(rewritten, path)


def repair_chm_epub_images(source: Path, path: Path, work: Path) -> dict:
    """Add CHM images omitted by Calibre because Windows paths are case-insensitive."""
    extracted = work / "chm-image-assets"
    extracted.mkdir(parents=True, exist_ok=True)
    run_checked(
        ["7z", "x", "-y", f"-o{extracted}", str(source)],
        timeout_seconds=CHM_COMMAND_TIMEOUT_SECONDS,
    )
    source_files = {
        item.relative_to(extracted).as_posix().casefold(): item
        for item in extracted.rglob("*") if item.is_file()
    }
    source_pages = {
        key: item for key, item in source_files.items()
        if item.suffix.lower() in {".htm", ".html", ".xhtml"}
    }
    with zipfile.ZipFile(path) as archive:
        entries = {info.filename: archive.read(info.filename) for info in archive.infolist()}
        infos = {info.filename: info for info in archive.infolist()}
    names_casefold = {name.casefold(): name for name in entries}
    documents = [name for name in entries if name.lower().endswith((".htm", ".html", ".xhtml"))]
    added = []
    missing = []
    changed_documents = {}

    def source_page(document: str) -> Path | None:
        direct = source_pages.get(document.casefold())
        if direct:
            return direct
        candidates = [item for item in source_pages.values() if item.name.casefold() == Path(document).name.casefold()]
        return candidates[0] if len(candidates) == 1 else None

    def source_image(document: str, raw: str) -> Path | None:
        page = source_page(document)
        parsed = urllib.parse.urlsplit(urllib.parse.unquote(raw))
        relative = parsed.path.replace("\\", "/").lstrip("/")
        candidates = []
        if page:
            candidates.append(posixpath.normpath(posixpath.join(page.parent.relative_to(extracted).as_posix(), relative)))
        candidates.append(posixpath.normpath(relative))
        for candidate in candidates:
            item = source_files.get(candidate.casefold())
            if item and item.suffix.lower() in {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}:
                return item
        basename = Path(relative).name.casefold()
        matches = [item for item in source_files.values() if item.name.casefold() == basename]
        return matches[0] if len(matches) == 1 else None

    image_pattern = re.compile(
        r"(<(?:[A-Za-z_][\w.-]*:)?img\b[^>]*?\bsrc\s*=\s*)([\"'])([^\"']+)(\2)",
        re.IGNORECASE,
    )
    for document in documents:
        text = entries[document].decode("utf-8", "replace")

        def replace_image(match):
            raw = html.unescape(match.group(3)).strip()
            parsed = urllib.parse.urlsplit(raw)
            if not raw or parsed.scheme or parsed.netloc or parsed.fragment or raw.lower().startswith("data:"):
                return match.group(0)
            target = posixpath.normpath(posixpath.join(posixpath.dirname(document), parsed.path.replace("\\", "/")))
            if target.startswith("../"):
                return match.group(0)
            existing = names_casefold.get(target.casefold())
            if existing:
                if existing != target:
                    relative = posixpath.relpath(existing, posixpath.dirname(document) or ".")
                    return match.group(1) + match.group(2) + relative + match.group(4)
                return match.group(0)
            image = source_image(document, raw)
            if image is None:
                missing.append({"document": document, "src": raw})
                return match.group(0)
            entries[target] = image.read_bytes()
            names_casefold[target.casefold()] = target
            added.append({"document": document, "src": raw, "path": target})
            return match.group(0)

        rewritten = image_pattern.sub(replace_image, text)
        if rewritten != text:
            changed_documents[document] = rewritten.encode("utf-8")

    if not added and not changed_documents:
        return {"added": [], "missing": missing}

    container_name = "META-INF/container.xml"
    container = ET.fromstring(entries[container_name])
    rootfile = container.find(".//{*}rootfile")
    if rootfile is None or not rootfile.attrib.get("full-path"):
        raise RuntimeError("CHM EPUB has no package document")
    opf_name = rootfile.attrib["full-path"]
    package = ET.fromstring(entries[opf_name])
    manifest = package.find(".//{*}manifest")
    if manifest is None:
        raise RuntimeError("CHM EPUB has no manifest")
    existing_ids = {item.attrib.get("id") for item in manifest.findall("{*}item")}
    existing_hrefs = {item.attrib.get("href") for item in manifest.findall("{*}item")}
    manifest_namespace = manifest.tag.rsplit("}", 1)[0] + "}" if "}" in manifest.tag else ""
    opf_dir = posixpath.dirname(opf_name)
    for index, item in enumerate(added, start=1):
        target = item["path"]
        href = posixpath.relpath(target, opf_dir or ".")
        if href in existing_hrefs:
            continue
        item_id = f"chm-image-{index}"
        while item_id in existing_ids:
            index += 1
            item_id = f"chm-image-{index}"
        media_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
        manifest.append(ET.Element(f"{manifest_namespace}item", {"id": item_id, "href": href, "media-type": media_type}))
        existing_ids.add(item_id)
        existing_hrefs.add(href)
    entries.update(changed_documents)
    entries[opf_name] = ET.tostring(package, encoding="utf-8", xml_declaration=False)
    rewritten = work / "repaired-chm.epub"
    with zipfile.ZipFile(rewritten, "w") as target_archive:
        for name, data in entries.items():
            info = infos.get(name)
            if info is None:
                target_archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
            else:
                if name == "mimetype":
                    info.compress_type = zipfile.ZIP_STORED
                target_archive.writestr(info, data)
    shutil.move(rewritten, path)
    return {"added": added, "missing": missing}


def convert_chm(source: Path, target: Path, work: Path) -> None:
    try:
        from .chm_navigation import repair_conversion
    except ImportError:
        from chm_navigation import repair_conversion
    try:
        run_checked(
            ["ebook-convert", str(source), str(target), "--flow-size", "0"],
            timeout_seconds=CHM_COMMAND_TIMEOUT_SECONDS,
        )
        sanitize_chm_epub(target, work)
        if target.exists():
            repair_conversion(source, target, work)
        validate_output(target, "epub")
        validate_chm_epub(target)
        return
    except (RuntimeError, ValueError, FileNotFoundError, zipfile.BadZipFile, KeyError) as exc:
        initial_error = exc
    source_html = work / "chm-fallback.html"
    try:
        convert_chm_to_html(source, source_html, work)
    except Exception:
        raise initial_error
    target.unlink(missing_ok=True)
    run_checked(
        ["ebook-convert", str(source_html), str(target), "--flow-size", "0"],
        timeout_seconds=CHM_COMMAND_TIMEOUT_SECONDS,
    )
    sanitize_chm_epub(target, work)
    if target.exists():
        try:
            repair_conversion(source, target, work)
        except ValueError:
            # Dynamic menu templates can hide chapters from Calibre entirely.
            # The recovery path parses static data and verifies every page's text.
            try:
                from .recover_chm import recover
            except ImportError:
                from recover_chm import recover
            recover(source, target, source.stem)
    validate_output(target, "epub")
    validate_chm_epub(target)


def convert_chm_to_html(source: Path, target: Path, work: Path) -> None:
    """Extract CHM pages and merge their sanitized text into one HTML document."""
    try:
        listing = command_output(["7z", "l", "-slt", str(source)], timeout_seconds=CHM_COMMAND_TIMEOUT_SECONDS)
    except ReaderConversionCommandError:
        # 7z can list the readable portion of a damaged CHM only with a
        # non-zero status; extraction below still recovers its HTML pages.
        listing = ""
    expanded_size = sum(int(value) for value in re.findall(r"^Size = (\d+)\s*$", listing, re.MULTILINE))
    member_count = len(re.findall(r"^Path = ", listing, re.MULTILINE))
    if expanded_size > MAX_CHM_EXPANDED_BYTES or member_count > MAX_EPUB_MEMBERS:
        raise RuntimeError("CHM expanded content exceeds limits")
    extracted = work / "chm-html"
    extracted.mkdir()
    try:
        subprocess.run(
            ["7z", "x", "-y", f"-o{extracted}", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=CHM_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass
    pages = sorted(path for path in extracted.rglob("*") if path.is_file() and path.suffix.lower() in {".htm", ".html"})
    text_pages = sorted(path for path in extracted.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")
    mhtml = sorted(path for path in extracted.rglob("*") if path.is_file() and path.suffix.lower() in {".mht", ".mhtml"})
    hhc = sorted(path for path in extracted.rglob("*") if path.is_file() and path.suffix.lower() == ".hhc")
    documents = []
    for path in pages:
        document = decode_html_source(path)
        documents.append((path.relative_to(extracted).as_posix(), inline_local_html_resources(
            document, extracted, path.parent, allow_relative=True)))
    for path in text_pages:
        document = decode_html_source(path)
        expanded, was_script = expand_document_writes(document)
        if was_script:
            documents.append((path.relative_to(extracted).as_posix(), inline_local_html_resources(
                expanded, extracted, path.parent, allow_relative=True)))
        else:
            documents.append((path.relative_to(extracted).as_posix(), f"<pre>{html.escape(document)}</pre>"))
    for index, path in enumerate(mhtml):
        if path.stat().st_size > MAX_MHTML_SOURCE_BYTES:
            raise RuntimeError("CHM MHTML source exceeds size limit")
        converted = work / f"chm-mhtml-{index:04d}.html"
        mhtml_to_html(path, converted)
        documents.append((converted.name, decode_html_source(converted)))
    if not documents:
        raise RuntimeError("CHM contains no HTML pages")

    toc_nodes = []
    if hhc:
        class TocParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.entries = []
                self.stack = [self.entries]
                self.current = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                tag = tag.lower()
                if tag == "object":
                    self.current = {}
                elif tag == "param" and self.current is not None:
                    name = attrs.get("name", "").lower()
                    if name in {"name", "local"}:
                        self.current[name] = attrs.get("value", "")
                elif tag == "ul" and self.stack[-1]:
                    last = self.stack[-1][-1]
                    if isinstance(last, dict) and "children" in last:
                        self.stack.append(last["children"])

            def handle_endtag(self, tag):
                tag = tag.lower()
                if tag == "object" and self.current:
                    node = {**self.current, "children": []}
                    self.stack[-1].append(node)
                    self.current = None
                elif tag == "ul" and len(self.stack) > 1:
                    self.stack.pop()

        parser = TocParser()
        parser.feed(decode_html_source(hhc[0]))
        toc_nodes = parser.entries

    page_map = {path.lower(): index for index, (path, _) in enumerate(documents)}
    basename_map = {}
    for path, index in page_map.items():
        basename_map.setdefault(posixpath.basename(path), []).append(index)

    def page_target(raw: str, current: str):
        value = urllib.parse.unquote(html.unescape(raw).replace("\\", "/"))
        value, fragment = value.split("#", 1) if "#" in value else (value, "")
        if not value:
            return None, fragment
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(current), value)).lower()
        index = page_map.get(resolved)
        if index is None and len(basename_map.get(posixpath.basename(resolved), [])) == 1:
            index = basename_map[posixpath.basename(resolved)][0]
        return index, fragment

    ordered_indices = []

    directory_candidates = []
    for index, (_, document) in enumerate(documents):
        visible = html.unescape(re.sub(r"<[^>]+>", " ", document))
        internal_links = len(re.findall(
            r"\bhref\s*=\s*[\"'][^\"']+\.(?:htm|html)(?:#[^\"']*)?[\"']",
            document, re.IGNORECASE,
        ))
        directory_candidates.append(((2 if "目录" in visible else 0) + min(internal_links, 100) / 100, index))
    directory_candidates.sort(reverse=True)

    def collect_toc_order(nodes):
        for node in nodes:
            page_index, _ = page_target(node.get("local", ""), "")
            if page_index is not None and page_index not in ordered_indices:
                ordered_indices.append(page_index)
            collect_toc_order(node.get("children", []))

    collect_toc_order(toc_nodes)
    if ordered_indices:
        # A partial .hhc describes order, not the complete inventory of pages.
        listed = set(ordered_indices)
        ordered_indices.extend(index for index in range(len(documents)) if index not in listed)
        ordered = [documents[index] for index in ordered_indices]
        documents = ordered
        page_map = {path.lower(): index for index, (path, _) in enumerate(documents)}
        basename_map = {}
        for path, index in page_map.items():
            basename_map.setdefault(posixpath.basename(path), []).append(index)

    def rewrite_document(document: str, current: str, index: int) -> str:
        prefix = f"chm-page-{index}--"
        document = re.sub(r"(\b(?:id|name)\s*=\s*[\"'])([^\"']+)([\"'])",
                          lambda match: match.group(1) + prefix + match.group(2) + match.group(3), document,
                          flags=re.IGNORECASE)

        def link(match):
            before, raw, quote = match.groups()
            if raw.startswith("#"):
                value = "#" + prefix + raw[1:]
            else:
                target, fragment = page_target(raw, current)
                if target is None:
                    return match.group(0)
                value = f"#chm-page-{target}" + (f"--{fragment}" if fragment else "")
            return before + value + quote

        return re.sub(r"(\bhref\s*=\s*[\"'])([^\"']+)([\"'])", link, document, flags=re.IGNORECASE)

    body_parts = []
    for index, (path, document) in enumerate(documents):
        body_parts.append(f'<section id="chm-page-{index}">{rewrite_document(document, path, index)}</section>')
    body = "".join(body_parts)
    target.write_text(f"<!doctype html><meta charset=\"utf-8\"><main>{body}</main>", encoding="utf-8")
    validate_html_content(target)


def detect_caj_family(source: Path) -> str:
    header = source.read_bytes()[:16]
    if header.startswith(b"%PDF-"):
        return "pdf"
    if header.startswith(b"KDH "):
        return "kdh"
    if header.startswith(b"CAJ"):
        return "caj"
    if header.startswith(b"HN"):
        return "hn"
    if header.startswith(b"\xc8"):
        return "c8"
    if header[:8] == b"\0" * 8:
        return "hn"
    raise RuntimeError("unsupported CAJ-family container")


def extract_kdh_pdf(source: Path, target: Path, work: Path) -> None:
    data = source.read_bytes()
    if len(data) <= 254:
        raise RuntimeError("KDH container is truncated")
    key = b"FZHMEI"
    decrypted = bytes(value ^ key[index % len(key)] for index, value in enumerate(data[254:]))
    end = decrypted.rfind(b"%%EOF")
    if end < 0:
        raise RuntimeError("KDH container has no embedded PDF terminator")
    raw = work / "kdh-raw.pdf"
    raw.write_bytes(decrypted[:end + 5])
    run_checked(["mutool", "clean", str(raw), str(target)])


def convert_caj_family(source: Path, target: Path, work: Path) -> None:
    kind = detect_caj_family(source)
    if kind == "pdf":
        shutil.copyfile(source, target)
        return
    if kind == "kdh":
        extract_kdh_pdf(source, target, work)
        return
    converter = CAJ2PDF_DIR / "caj2pdf"
    if not converter.is_file():
        raise RuntimeError("pinned caj2pdf converter is unavailable")
    for library in ("libjbigdec.so", "libjbig2codec.so"):
        source_library = CAJ2PDF_DIR / library
        if source_library.is_file():
            shutil.copyfile(source_library, work / library)
    run_checked(["python3", str(converter), "convert", str(source), "--output", str(target)], cwd=work)


def validate_djvu_pdf(path: Path, work: Path) -> None:
    info = command_output(["pdfinfo", str(path)], timeout_seconds=DJVU_COMMAND_TIMEOUT_SECONDS)
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    if not match or int(match.group(1)) < 1:
        raise RuntimeError("converted DjVu PDF has no pages")
    page_count = int(match.group(1))
    for page in sorted({1, page_count}):
        prefix = work / f"djvu-page-{page}"
        run_checked([
            "pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-png", "-r", "72",
            str(path), str(prefix),
        ], timeout_seconds=DJVU_COMMAND_TIMEOUT_SECONDS)
        rendered = prefix.with_suffix(".png")
        if not rendered.is_file() or rendered.stat().st_size == 0:
            raise RuntimeError(f"converted DjVu PDF page {page} is not renderable")


def validate_pdf_content(path: Path, work: Path) -> None:
    info = command_output(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    if not match or int(match.group(1)) < 1:
        raise RuntimeError("converted PDF has no pages")
    page_count = int(match.group(1))
    if page_count <= 12:
        samples = list(range(1, page_count + 1))
    else:
        samples = sorted({1, 2, page_count // 4, page_count // 2, page_count * 3 // 4, page_count - 1, page_count})
    content_pages = []
    sample_dir = work / "pdf-content-samples"
    sample_dir.mkdir(exist_ok=True)
    for page in samples:
        prefix = sample_dir / f"page-{page}"
        run_checked([
            "pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-png", "-r", "72",
            str(path), str(prefix),
        ])
        rendered = prefix.with_suffix(".png")
        if not rendered.is_file() or rendered.stat().st_size == 0:
            raise RuntimeError(f"converted PDF page {page} is not renderable")
        if page_content_ratio(rendered) >= MIN_PAGE_CONTENT_RATIO:
            content_pages.append(page)
    if not content_pages:
        raise RuntimeError("converted PDF has no visible page content")
    if page_count >= 4 and not any(page > page_count // 2 for page in content_pages):
        raise RuntimeError("converted PDF has no visible content after its midpoint")


def rasterize_pdf(path: Path, work: Path) -> None:
    pages = work / "raster-pages"
    pages.mkdir()
    run_checked(["pdftoppm", "-png", "-r", "150", str(path), str(pages / "page")])
    images = sorted(pages.glob("page-*.png"), key=lambda item: int(item.stem.rsplit("-", 1)[-1]))
    if not images:
        raise RuntimeError("PDF rasterization produced no pages")
    sample = "".join(command_output(["tesseract", str(image), "stdout", "-l", "chi_sim"]) for image in images[:3])
    if not CJK_RE.search(sample):
        raise RuntimeError("rasterized office PDF has no recognizable CJK text")
    content = [page_content_ratio(image) >= MIN_PAGE_CONTENT_RATIO for image in images]
    for index in range(1, len(images) - 1):
        if content[index] or not any(content[:index]) or not any(content[index + 1:]):
            continue
        replacement = pages / f"ghostscript-{index + 1}.png"
        run_checked([
            "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r150",
            f"-dFirstPage={index + 1}", f"-dLastPage={index + 1}",
            f"-sOutputFile={replacement}", str(path),
        ])
        if not replacement.is_file() or page_content_ratio(replacement) < MIN_PAGE_CONTENT_RATIO:
            raise RuntimeError(f"rasterized office PDF has blank interior page {index + 1}")
        shutil.move(replacement, images[index])
    frames = [Image.open(image).convert("RGB") for image in images]
    rewritten = work / "rasterized.pdf"
    frames[0].save(rewritten, "PDF", save_all=True, append_images=frames[1:], resolution=150.0)
    for frame in frames:
        frame.close()
    shutil.move(rewritten, path)


def page_content_ratio(path: Path) -> float:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        histogram = grayscale.histogram()
        pixels = grayscale.width * grayscale.height
    return sum(histogram[:245]) / pixels if pixels else 0.0


def epub_package(path: Path):
    archive = zipfile.ZipFile(path)
    try:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        opf_path = rootfile.attrib["full-path"]
        package = ET.fromstring(archive.read(opf_path))
    except (KeyError, ET.ParseError, AttributeError) as exc:
        archive.close()
        raise RuntimeError("converted EPUB has no package document") from exc
    return archive, package, opf_path


def validate_epub_content(path: Path) -> None:
    archive, package, opf_path = epub_package(path)
    with archive:
        names = set(archive.namelist())
        manifest = {item.attrib.get("id"): item for item in package.findall(".//{*}manifest/{*}item") if item.attrib.get("id")}
        spine = [item.attrib.get("idref") for item in package.findall(".//{*}spine/{*}itemref")]
        if not spine:
            raise RuntimeError("converted EPUB has no spine")
        base = posixpath.dirname(opf_path)
        meaningful = []
        for idref in spine:
            item = manifest.get(idref)
            if item is None or not item.attrib.get("href"):
                raise RuntimeError("converted EPUB has an invalid spine reference")
            href = urllib.parse.unquote(item.attrib["href"].split("#", 1)[0])
            document_path = posixpath.normpath(posixpath.join(base, href))
            if document_path.startswith("../"):
                raise RuntimeError("converted EPUB spine escapes its package directory")
            try:
                document = archive.read(document_path).decode("utf-8", "replace")
                root = ET.fromstring(document)
            except KeyError as exc:
                raise RuntimeError("converted EPUB spine document is missing") from exc
            except ET.ParseError as exc:
                raise RuntimeError("converted EPUB has malformed spine content") from exc
            if re.search(r"<(?:script|object|embed|iframe)\b", document, re.IGNORECASE):
                raise RuntimeError("converted EPUB contains active content")
            if re.search(r"\b(?:src|poster)\s*=\s*[\"']\s*(?:https?:)?//", document, re.IGNORECASE):
                raise RuntimeError("converted EPUB contains an external embedded resource")
            body = root.find(".//{*}body")
            text = html.unescape("".join(body.itertext())) if body is not None else ""
            has_text = len(re.sub(r"\s+", "", text)) >= 20
            has_image = False
            for image in root.findall(".//{*}img"):
                source = urllib.parse.unquote((image.attrib.get("src") or "").split("#", 1)[0])
                if not source:
                    raise RuntimeError("converted EPUB contains an image without a source")
                if source.lower().startswith("data:image/"):
                    has_image = True
                    continue
                parsed = urllib.parse.urlsplit(source)
                if parsed.scheme or parsed.netloc:
                    raise RuntimeError("converted EPUB contains an external image")
                image_path = posixpath.normpath(posixpath.join(posixpath.dirname(document_path), parsed.path))
                if image_path.startswith("../") or image_path not in names:
                    raise RuntimeError(
                        f"converted EPUB image resource is missing: {document_path} <- {source}"
                    )
                has_image = True
            meaningful.append(has_text or has_image)
        if not any(meaningful):
            raise RuntimeError("converted EPUB has no readable content")
        if len(meaningful) >= 4 and not any(meaningful[len(meaningful) // 2:]):
            raise RuntimeError("converted EPUB has no readable content after its midpoint")


def validate_docx_content(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        try:
            document = ET.fromstring(archive.read("word/document.xml"))
        except (KeyError, ET.ParseError) as exc:
            raise RuntimeError("converted DOCX has malformed document content") from exc
        text = "".join(node.text or "" for node in document.findall(".//{*}t"))
        visible_objects = document.findall(".//{*}drawing") + document.findall(".//{*}pict") + document.findall(".//{*}object")
        if not re.sub(r"\s+", "", text) and not visible_objects:
            raise RuntimeError("converted DOCX has no readable content")


def validate_html_content(path: Path) -> None:
    document = path.read_text(encoding="utf-8", errors="replace")
    visible = re.sub(r"<(script|style|template|object|iframe)\b[^>]*>.*?</\1\s*>", "", document, flags=re.IGNORECASE | re.DOTALL)
    visible = html.unescape(re.sub(r"<[^>]+>", " ", visible))
    has_image = bool(re.search(r"<img\b[^>]*\bsrc\s*=\s*[\"']data:image/", document, re.IGNORECASE))
    if not re.sub(r"\s+", "", visible) and not has_image:
        raise RuntimeError("converted HTML has no readable content")


def validate_page_manifest(path: Path) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("converted page manifest is invalid") from exc
    page_count = manifest.get("page_count") if isinstance(manifest, dict) else None
    if (manifest.get("version") != 2 or manifest.get("kind") != "pdf-pages"
            or type(page_count) is not int or page_count < 1 or "pages" in manifest):
        raise RuntimeError("converted page manifest is invalid")


def convert_spreadsheet_to_pages(source: Path, target: Path, work: Path, item: dict,
                                 source_sha256: str, object_path: str) -> None:
    output = work / "spreadsheet-html"
    output.mkdir()
    helper = Path(__file__).with_name("render_spreadsheet_html.py")
    run_checked([
        "/usr/bin/python3", str(helper), str(source), str(output),
    ], timeout_seconds=SPREADSHEET_RENDER_COMMAND_TIMEOUT_SECONDS)
    html_pages = sorted(output.glob("sheet-*/sheet.html"))
    if not html_pages:
        raise RuntimeError("LibreOffice produced no spreadsheet worksheet HTML")

    expected_sheets, expected_values, expected_charts, expected_images = spreadsheet_source_inventory(
        source, item.get("extension", source.suffix.lower().lstrip(".")),
    )

    if len(html_pages) < len(expected_sheets):
        raise RuntimeError("spreadsheet HTML export omitted worksheet pages")
    if len(expected_values) >= 50_000 and expected_charts == 0 and expected_images == 0:
        html_pages = split_spreadsheet_html_rows(html_pages, work / "spreadsheet-html-chunks")
    exported_html = "\n".join(page.read_text(encoding="utf-8", errors="replace") for page in html_pages)
    visible_html = spreadsheet_text_key(extract_html_text(exported_html))
    missing_values = [value for value in expected_values
                      if not spreadsheet_text_present(value, visible_html)]
    if missing_values:
        diagnostics = [len(spreadsheet_text_key(value)) for value in missing_values[:8]]
        raise RuntimeError(
            f"spreadsheet HTML export omitted {len(missing_values)} non-empty cell value(s); "
            f"normalized cell lengths: {diagnostics!r}"
        )
    exported_images = sum(len(image_sources(page.read_text(encoding="utf-8", errors="replace")))
                          for page in html_pages)
    if exported_images < expected_charts + expected_images:
        raise RuntimeError("spreadsheet HTML export omitted chart or image objects")

    rendered = work / "spreadsheet-rendered"
    rendered.mkdir()
    screenshots = render_spreadsheet_html(html_pages, rendered)
    page_root = target.parent / "pages"
    page_root.mkdir(parents=True, exist_ok=True)
    page_count = 0
    for number, image_path in enumerate(screenshots, 1):
        destination = page_root / f"page-{number:06d}.webp"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.save(destination, "WEBP", method=6, quality=85)
        page_count = number
    target.write_bytes(canonical_json({
        "version": 2, "kind": "pdf-pages", "source_sha256": source_sha256,
        "profile": item["profile"], "page_count": page_count,
    }, pretty=True))


def render_spreadsheet_html(html_pages: list[Path], output: Path) -> list[Path]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("spreadsheet image rendering requires Playwright") from exc
    screenshots = []
    with sync_playwright() as playwright:
        chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
        browser = playwright.chromium.launch(
            headless=True, args=["--no-sandbox"], **({"channel": "chrome"} if chrome else {})
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 1600}, device_scale_factor=1)
            for sheet_number, source in enumerate(html_pages, 1):
                page.goto(source.resolve().as_uri(), wait_until="load", timeout=120000)
                page.wait_for_function(
                    "() => [...document.images].every(image => image.complete && image.naturalWidth > 0)",
                    timeout=60000,
                )
                page.evaluate("""() => document.fonts?.ready.then(() => true)""")
                dimensions = page.evaluate("""() => ({
                  width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                  height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
                })""")
                if dimensions["width"] < 2 or dimensions["height"] < 2:
                    raise RuntimeError(f"spreadsheet worksheet {sheet_number} rendered blank")
                width, height = dimensions["width"], dimensions["height"]
                if (width <= SPREADSHEET_FULL_PAGE_MAX_EDGE
                        and height <= SPREADSHEET_FULL_PAGE_MAX_EDGE
                        and width * height <= SPREADSHEET_FULL_PAGE_MAX_PIXELS):
                    page.set_viewport_size({"width": width, "height": min(height, 1600)})
                    page.wait_for_timeout(80)
                    target = output / f"sheet-{sheet_number:04d}.png"
                    page.screenshot(path=str(target), full_page=True)
                    screenshots.append(target)
                    continue

                tile_width, tile_height = min(width, 1800), min(height, 1600)
                page.set_viewport_size({"width": tile_width, "height": tile_height})
                page.wait_for_timeout(80)
                dimensions = page.evaluate("""() => ({
                  width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                  height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
                })""")
                columns = max(1, (max(0, dimensions["width"] - tile_width) + 1759) // 1760 + 1)
                rows = max(1, (max(0, dimensions["height"] - tile_height) + 1559) // 1560 + 1)
                for row in range(rows):
                    for column in range(columns):
                        page.evaluate("""({x, y}) => window.scrollTo(x, y)""", {
                            "x": column * 1760, "y": row * 1560,
                        })
                        page.wait_for_timeout(80)
                        target = output / f"sheet-{sheet_number:04d}-tile-{row:04d}-{column:04d}.png"
                        page.screenshot(path=str(target))
                        screenshots.append(target)
            return screenshots
        finally:
            browser.close()


def split_spreadsheet_html_rows(html_pages: list[Path], output: Path,
                                rows_per_chunk: int = 50) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for sheet_number, source in enumerate(html_pages, 1):
        document = lxml_html.parse(str(source))
        tables = document.xpath("//table")
        if not tables:
            result.append(source)
            continue
        table = max(tables, key=lambda candidate: len(candidate.xpath(".//tr")))
        rows = table.xpath(".//tr")
        if len(rows) <= rows_per_chunk:
            result.append(source)
            continue
        table_path = document.getpath(table)
        for chunk_number, start in enumerate(range(0, len(rows), rows_per_chunk), 1):
            end = min(len(rows), start + rows_per_chunk)
            chunk = copy.deepcopy(document)
            chunk_table = chunk.xpath(table_path)[0]
            chunk_rows = chunk_table.xpath(".//tr")
            for index in range(len(chunk_rows) - 1, -1, -1):
                if not start <= index < end:
                    parent = chunk_rows[index].getparent()
                    if parent is not None:
                        parent.remove(chunk_rows[index])
            target = output / f"sheet-{sheet_number:04d}-chunk-{chunk_number:04d}.html"
            target.write_bytes(lxml_html.tostring(
                chunk, encoding="utf-8", method="html", doctype="<!DOCTYPE html>"))
            result.append(target)
    return result


def spreadsheet_source_inventory(source: Path, extension: str = "") -> tuple[list[str], list[str], int, int]:
    extension = (extension or source.suffix.lower().lstrip(".")).lower().lstrip(".")
    if extension == "csv":
        raw = source.read_bytes()
        text = None
        for encoding in ("utf-8-sig", "gb18030", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise RuntimeError("spreadsheet CSV text encoding is unreadable")
        sample = text[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        values = [cell.strip() for row in csv.reader(io.StringIO(text, newline=""), dialect)
                  for cell in row if cell.strip()]
        return ["Sheet1"], list(dict.fromkeys(values)), 0, 0
    try:
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
            if "content.xml" in names and extension == "ods":
                content = ET.fromstring(archive.read("content.xml"))
                tables = content.findall(".//{*}table")
                sheets = [next((value for key, value in table.attrib.items()
                                if key.endswith("}name")), "") for table in tables]
                values = []
                for table in tables:
                    for cell in table.findall(".//{*}table-cell"):
                        text = "".join(node.text or "" for node in cell.findall(".//{*}p"))
                        if text.strip():
                            values.append(text.strip())
                charts = sum(name.startswith("Object ") and name.endswith("/content.xml")
                             for name in names)
                images = sum(name.startswith("Pictures/") and not name.endswith("/")
                             for name in names)
                return sheets, list(dict.fromkeys(values)), charts, images
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            targets = {node.attrib.get("Id"): node.attrib.get("Target", "")
                       for node in relationships.findall(".//{*}Relationship")}
            sheet_nodes = workbook.findall(".//{*}sheet")
            sheets = [node.attrib.get("name", "") for node in sheet_nodes]
            shared_strings = []
            if "xl/sharedStrings.xml" in names:
                strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared_strings = ["".join(node.itertext()) for node in strings.findall("{*}si")]
            values = []
            relationship_id = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            for sheet_node in sheet_nodes:
                target = targets.get(sheet_node.get(relationship_id), "")
                worksheet = posixpath.normpath(target.lstrip("/") if target.startswith("/")
                                               else posixpath.join("xl", target))
                if worksheet not in names:
                    raise RuntimeError(f"spreadsheet worksheet XML is missing: {sheet_node.get('name', '')}")
                root = ET.fromstring(archive.read(worksheet))
                hidden_rows = {int(row.get("r")) for row in root.findall(".//{*}row")
                               if row.get("r", "").isdigit() and row.get("hidden") in {"1", "true"}}
                hidden_columns = set()
                for column in root.findall(".//{*}cols/{*}col"):
                    if column.get("hidden") not in {"1", "true"}:
                        continue
                    first, last = int(column.get("min", "1")), int(column.get("max", "1"))
                    hidden_columns.update(range(first, last + 1))
                for cell in root.findall(".//{*}c"):
                    coordinate = cell.get("r", "")
                    row_match = re.search(r"([0-9]+)$", coordinate)
                    column_match = re.match(r"([A-Z]+)", coordinate, re.IGNORECASE)
                    if not row_match or not column_match or int(row_match.group(1)) in hidden_rows:
                        continue
                    column_number = 0
                    for letter in column_match.group(1).upper():
                        column_number = column_number * 26 + ord(letter) - ord("A") + 1
                    if column_number in hidden_columns:
                        continue
                    kind = cell.attrib.get("t")
                    if kind == "s":
                        value = cell.findtext("{*}v", "")
                        if value.isdigit() and int(value) < len(shared_strings):
                            text = shared_strings[int(value)]
                            if text.strip():
                                values.append(text.strip())
                    elif kind in {"inlineStr", "str"}:
                        node = cell.find("{*}is") if kind == "inlineStr" else cell.find("{*}v")
                        text = "".join(node.itertext()) if node is not None else ""
                        if text.strip():
                            values.append(text.strip())
            charts = sum(name.startswith("xl/charts/chart") and name.endswith(".xml") for name in names)
            images = sum(name.startswith("xl/media/") and not name.endswith("/") for name in names)
            return sheets, list(dict.fromkeys(values)), charts, images
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        try:
            import xlrd
            legacy = xlrd.open_workbook(str(source), on_demand=True)
        except Exception as exc:
            raise RuntimeError("spreadsheet workbook structure is unreadable") from exc
        sheets, values = [], []
        try:
            for sheet in legacy.sheets():
                sheets.append(sheet.name)
                for row in range(sheet.nrows):
                    row_info = sheet.rowinfo_map.get(row)
                    if row_info and row_info.hidden:
                        continue
                    for column in range(sheet.ncols):
                        column_info = sheet.colinfo_map.get(column)
                        if column_info and column_info.hidden:
                            continue
                        value = sheet.cell_value(row, column)
                        if isinstance(value, str) and value.strip():
                            values.append(value.strip())
        finally:
            legacy.release_resources()
        return sheets, list(dict.fromkeys(values)), 0, 0


def spreadsheet_text_variants(value: str) -> set[str]:
    variants = {value}
    for encoding in ("latin-1", "cp1252"):
        try:
            variants.add(value.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return variants


def spreadsheet_text_key(value: str) -> str:
    """Ignore layout whitespace inserted or normalized by HTML export."""
    return re.sub(r"\s+", "", value)


def spreadsheet_text_present(value: str, rendered_text: str) -> bool:
    for candidate in spreadsheet_text_variants(value):
        key = spreadsheet_text_key(candidate)
        if not key or key in rendered_text:
            return True
        position = 0
        matched = True
        for offset in range(0, len(key), 24):
            chunk = key[offset:offset + 24]
            found = rendered_text.find(chunk, position)
            if found >= 0:
                position = found + len(chunk)
                continue
            window_end = min(len(rendered_text), position + len(chunk) * 8 + 512)
            window = rendered_text[position:window_end]
            for char in chunk:
                char_position = window.find(char)
                if char_position < 0:
                    matched = False
                    break
                window = window[char_position + 1:]
            if not matched:
                break
            position = window_end - len(window)
        if matched:
            return True
        chunks = [key[offset:offset + 16] for offset in range(0, len(key), 16)]
        if all(chunk in rendered_text for chunk in chunks):
            return True
    return False


class _ImageSources(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "img":
            self.sources.append(dict(attrs).get("src", "").strip())


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"td", "th", "tr", "br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def extract_html_text(document: str) -> str:
    parser = _VisibleText()
    parser.feed(document)
    return " ".join(parser.parts)


def image_sources(document: str) -> list[str]:
    parser = _ImageSources()
    parser.feed(document)
    return parser.sources


def validate_mhtml_content(path: Path) -> None:
    validate_html_content(path)
    parser = _ImageSources()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    for source in parser.sources:
        if not source:
            raise RuntimeError("converted MHTML contains an image without a source")
        match = re.fullmatch(r"data:image/(?:gif|jpeg|png|webp);base64,([A-Za-z0-9+/]*={0,2})", source, re.IGNORECASE)
        if not match:
            raise RuntimeError("converted MHTML contains an unembedded image")
        try:
            payload = base64.b64decode(match.group(1), validate=True)
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
        except Exception as exc:
            raise RuntimeError("converted MHTML contains an invalid embedded image") from exc


def validate_reader_content(path: Path, item: dict, work: Path) -> None:
    mode = item["reader_mode"]
    if mode == "pdf":
        if item.get("output_name") == "page-manifest.json":
            validate_page_manifest(path)
        else:
            validate_pdf_content(path, work)
    elif mode == "epub":
        validate_epub_content(path)
    elif mode == "docx":
        validate_docx_content(path)
    elif mode == "html":
        if item.get("extension") in {"mht", "mhtml"}:
            validate_mhtml_content(path)
        else:
            validate_html_content(path)
    elif mode == "foliate" and path.stat().st_size == 0:
        raise RuntimeError("original Foliate asset is empty")


def convert_file(item: dict, source: Path, target: Path, work: Path,
                 source_sha256: str = "", object_path: str = "") -> None:
    ext = item["extension"]
    office_profile = (work / "libreoffice-profile").resolve().as_uri()
    explicit_password = item.get("source_password") or source_password(item.get("repo", ""), item.get("path", ""))
    password = explicit_password
    if not password and ext == "xlsx":
        password = os.environ.get("READER_CONVERSION_PASSWORD", "")
    ole_xlsx = ext == "xlsx" and source.read_bytes()[:8] == OLE_SIGNATURE
    if (password or ole_xlsx) and ext in {"doc", "docx", "ppt", "pptx", "pps", "xls", "xlsx"}:
        import msoffcrypto
        with source.open("rb") as encrypted:
            document = None
            try:
                document = msoffcrypto.OfficeFile(encrypted)
            except Exception:
                if explicit_password:
                    raise
            if document is not None and document.is_encrypted():
                if not password:
                    raise RuntimeError(
                        "encrypted spreadsheet requires READER_CONVERSION_PASSWORD or a source password"
                    )
                decrypted_source = work / f"decrypted.{ext}"
                document.load_key(password=password)
                with decrypted_source.open("wb") as decrypted:
                    document.decrypt(decrypted)
                source = decrypted_source
    if ext == "pdf":
        if item.get("profile") == GBK_PDF_CONTRACT[0]:
            try:
                from .repair_gbk_pdf import repair_pdf
            except ImportError:
                from repair_gbk_pdf import repair_pdf
            repair_pdf(source, target)
            return
        if not password:
            shutil.copyfile(source, target)
            return
        run_checked(["qpdf", f"--password={password}", "--decrypt", str(source), str(target)])
    elif ext in {"htm", "html"}:
        source_url = item.get("source_url")
        if source_url:
            prepared = inline_html_resources(source, source_url, work)
        else:
            prepared = work / "sanitized.html"
            prepared.write_text(sanitize_html(decode_html_source(source)), encoding="utf-8")
        shutil.copyfile(prepared, target)
    elif ext == "doc":
        out = work / "office"
        out.mkdir()
        with source.open("rb") as handle:
            prefix = handle.read(256).lstrip(b"\xef\xbb\xbf\x00\t\r\n ")
        office_source = source
        if prefix.lower().startswith((b"<html", b"<!doctype html")):
            office_source = work / "source.html"
            shutil.copyfile(source, office_source)
            intermediate = work / "html-office"
            intermediate.mkdir()
            run_checked(["libreoffice", "--headless", f"-env:UserInstallation={office_profile}", "--convert-to", "odt", "--outdir", str(intermediate), str(office_source)])
            office_source = intermediate / "source.odt"
            if not office_source.is_file():
                raise RuntimeError("LibreOffice produced no ODT from HTML source")
        run_checked(["libreoffice", "--headless", f"-env:UserInstallation={office_profile}", "--convert-to", "docx", "--outdir", str(out), str(office_source)])
        produced = out / f"{office_source.stem}.docx"
        if not produced.exists():
            raise RuntimeError("LibreOffice produced no DOCX")
        shutil.move(produced, target)
    elif ext == "docx":
        try:
            with zipfile.ZipFile(source) as archive:
                archive.getinfo("word/document.xml")
            shutil.copyfile(source, target)
        except (zipfile.BadZipFile, KeyError):
            with source.open("rb") as handle:
                is_ole = handle.read(8) == OLE_SIGNATURE
            if is_ole:
                out = work / "mislabeled-office"
                out.mkdir()
                mislabeled = work / "mislabeled.doc"
                shutil.copyfile(source, mislabeled)
                run_checked(["libreoffice", "--headless", f"-env:UserInstallation={office_profile}", "--convert-to", "docx", "--outdir", str(out), str(mislabeled)])
                produced = out / "mislabeled.docx"
                if not produced.is_file():
                    raise RuntimeError("LibreOffice produced no DOCX from mislabeled source")
                shutil.move(produced, target)
            else:
                raise
    elif ext in {"txt", "md", "markdown"}:
        shutil.copyfile(source, target)
    elif ext in {"jpg", "jpeg", "png", "gif", "bmp", "webp"}:
        # Keep the source repository untouched, but serve one CDN-friendly
        # image format from the shared Reader bucket.
        if item.get("reader_mode") == "image" and item.get("output_name", "").endswith(".webp"):
            image = Image.open(source)
            image.seek(0)
            image.convert("RGB").save(target, "WEBP", method=6, quality=88)
        else:
            shutil.copyfile(source, target)
    elif ext == "pdf" and item.get("profile") == "native-pdf-v1":
        shutil.copyfile(source, target)
    elif ext in {"epub", "mobi", "azw3", "fb2"} and item.get("reader_mode") == "foliate":
        shutil.copyfile(source, target)
    elif ext == "odt":
        output = work / "odt-html"
        output.mkdir()
        run_checked(["libreoffice", "--headless", f"-env:UserInstallation={office_profile}", "--convert-to", "html", "--outdir", str(output), str(source)])
        generated = output / f"{source.stem}.html"
        if not generated.is_file():
            raise RuntimeError("LibreOffice produced no HTML from ODT")
        target.write_text(inline_local_html_resources(generated.read_text(encoding="utf-8", errors="replace"), output, output), encoding="utf-8")
    elif ext in {"mobi", "azw3", "fb2"}:
        run_checked(["ebook-convert", str(source), str(target), "--flow-size", "0"])
    elif ext == "rtf":
        intermediate = work / "rtf-html"
        intermediate.mkdir()
        run_checked([
            "libreoffice", "--headless", f"-env:UserInstallation={office_profile}",
            "--convert-to", "html", "--outdir", str(intermediate), str(source),
        ])
        html_source = intermediate / f"{source.stem}.html"
        if not html_source.is_file():
            raise RuntimeError("LibreOffice produced no HTML from RTF")
        target.write_text(inline_local_html_resources(html_source.read_text(encoding="utf-8", errors="replace"), intermediate, intermediate), encoding="utf-8")
    elif ext == "chm":
        convert_chm(source, target, work)
    elif ext in {"tif", "tiff"}:
        convert_tiff(source, target, work)
    elif ext == "djvu":
        if source.read_bytes()[:5] == b"%PDF-":
            run_checked(["qpdf", "--linearize", str(source), str(target)], timeout_seconds=DJVU_COMMAND_TIMEOUT_SECONDS)
            return
        last_error = None
        for attempt in range(2):
            try:
                run_checked(["ddjvu", "-format=pdf", str(source), str(target)], timeout_seconds=DJVU_COMMAND_TIMEOUT_SECONDS)
                break
            except (ReaderConversionCommandError, ReaderConversionTimeout) as exc:
                last_error = exc
                target.unlink(missing_ok=True)
                if attempt == 1:
                    raise
                time.sleep(2)
    elif ext in {"ppt", "pptx", "pps", "odp", "xls", "xlsx", "csv", "ods", "wps"}:
        if item.get("output_name") == "page-manifest.json":
            spreadsheet_source = source
            if ext == "xlsx" and source.read_bytes()[:8] == OLE_SIGNATURE:
                spreadsheet_source = work / "spreadsheet-source.xls"
                shutil.copyfile(source, spreadsheet_source)
            elif ext == "xls" and zipfile.is_zipfile(source):
                with zipfile.ZipFile(source) as workbook:
                    is_ooxml_workbook = "xl/workbook.xml" in workbook.namelist()
                if is_ooxml_workbook:
                    spreadsheet_source = work / "spreadsheet-source.xlsx"
                    shutil.copyfile(source, spreadsheet_source)
            convert_spreadsheet_to_pages(
                spreadsheet_source, target, work, item, source_sha256, object_path,
            )
            return
        out = work / "office-pdf"
        out.mkdir()
        office_source = source
        if ext == "xlsx" and source.read_bytes()[:8] == OLE_SIGNATURE:
            office_source = work / "source.xls"
            shutil.copyfile(source, office_source)
        run_checked([
            "libreoffice", "--headless", f"-env:UserInstallation={office_profile}",
            "--convert-to", "pdf", "--outdir", str(out), str(office_source),
        ])
        produced = out / f"{office_source.stem}.pdf"
        if not produced.is_file():
            raise RuntimeError("LibreOffice produced no PDF")
        shutil.move(produced, target)
    elif ext in {"mht", "mhtml"}:
        mhtml_to_html(source, target)
    elif ext == "ps":
        run_checked([
            "gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.7",
            f"-sOutputFile={target}", str(source),
        ], timeout_seconds=POSTSCRIPT_COMMAND_TIMEOUT_SECONDS)
    elif ext in {"caj", "kdh"}:
        convert_caj_family(source, target, work)
    elif ext in {"ape", "wma", "amr"}:
        run_checked([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-map_metadata", "-1", "-c:a", "libmp3lame", "-q:a", "3", str(target),
        ], timeout_seconds=MEDIA_COMMAND_TIMEOUT_SECONDS)
    elif ext in {"flv", "f4v", "rm", "rmvb", "mkv", "avi", "mpg", "mpeg", "mts", "ts", "wmv"}:
        if ext in {"rm", "rmvb"} and item.get("source_media_mode") == "audio":
            run_checked([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=640x360:r=1",
                "-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                "-map_metadata", "-1", "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(target),
            ], timeout_seconds=MEDIA_COMMAND_TIMEOUT_SECONDS)
            return
        run_checked([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
            "-map_metadata", "-1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-vf", "scale=w='min(1920,iw)':h='min(1080,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(target),
        ], timeout_seconds=MEDIA_COMMAND_TIMEOUT_SECONDS)
    else:
        raise ValueError(f"unsupported conversion extension: {ext}")


def normalized_office_pdf(source: Path, work: Path) -> Path:
    normalized = work / "normalized-office"
    normalized.mkdir()
    run_checked(["libreoffice", "--headless", "--convert-to", "docx", "--outdir", str(normalized), str(source)])
    docx = normalized / f"{source.stem}.docx"
    if not docx.is_file():
        raise RuntimeError("LibreOffice produced no normalized DOCX")
    pdf_dir = work / "normalized-pdf"
    pdf_dir.mkdir()
    run_checked(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(pdf_dir), str(docx)])
    pdf = pdf_dir / f"{source.stem}.pdf"
    if not pdf.is_file():
        raise RuntimeError("LibreOffice produced no normalized PDF")
    return pdf


def validate_output(path: Path, reader_mode: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("conversion output is empty")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise RuntimeError("conversion output exceeds size limit")
    if reader_mode == "pdf" and not path.read_bytes()[:5] == b"%PDF-":
        raise RuntimeError("conversion output is not a PDF")
    if reader_mode == "html" and not path.read_bytes():
        raise RuntimeError("conversion output is empty HTML")
    if reader_mode in {"text", "markdown", "image"} and not path.read_bytes():
        raise RuntimeError("conversion output is empty")
    if reader_mode == "epub":
        with zipfile.ZipFile(path) as archive:
            if archive.read("mimetype") != b"application/epub+zip":
                raise RuntimeError("conversion output is not an EPUB")
    if reader_mode == "foliate" and path.stat().st_size == 0:
        raise RuntimeError("conversion output is empty")
    if reader_mode == "docx":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise RuntimeError("conversion output is not a DOCX")
            if not archive.read("word/document.xml").strip():
                raise RuntimeError("DOCX document body is empty")
    if reader_mode in {"audio", "video"}:
        validate_media_output(path, reader_mode)


def validate_chm_epub(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            opf_path = rootfile.attrib["full-path"]
            package = ET.fromstring(archive.read(opf_path))
        except (KeyError, ET.ParseError, AttributeError) as exc:
            raise RuntimeError("converted CHM EPUB has no package document") from exc

        manifest = {
            item.attrib.get("id"): item
            for item in package.findall(".//{*}manifest/{*}item")
            if item.attrib.get("id")
        }
        spine = [item.attrib.get("idref") for item in package.findall(".//{*}spine/{*}itemref")]
        if not spine:
            raise RuntimeError("converted CHM EPUB has no spine")

        readable_characters = 0
        base = posixpath.dirname(opf_path)
        for idref in spine:
            item = manifest.get(idref)
            if item is None or not item.attrib.get("href"):
                raise RuntimeError("converted CHM EPUB has an invalid spine reference")
            href = urllib.parse.unquote(item.attrib["href"].split("#", 1)[0])
            document_path = posixpath.normpath(posixpath.join(base, href))
            try:
                document = archive.read(document_path).decode("utf-8", "replace")
            except KeyError as exc:
                raise RuntimeError("converted CHM EPUB spine document is missing") from exc
            if re.search(r"<(?:script|object|embed|iframe)\b", document, re.IGNORECASE):
                raise RuntimeError("converted CHM EPUB contains active content")
            if re.search(r"\b(?:src|poster)\s*=\s*[\"']\s*(?:https?:)?//", document, re.IGNORECASE):
                raise RuntimeError("converted CHM EPUB contains an external embedded resource")
            try:
                root = ET.fromstring(document)
            except ET.ParseError as exc:
                raise RuntimeError("converted CHM EPUB has malformed spine content") from exc
            body = root.find(".//{*}body")
            if body is not None:
                text = html.unescape("".join(body.itertext()))
                readable_characters += len(re.sub(r"\s+", "", text))
        if readable_characters < 20:
            raise RuntimeError("converted CHM EPUB has no readable content")


def validate_office_pdf(path: Path, item: dict, work: Path, source: Path | None = None) -> None:
    text = command_output(["pdftotext", str(path), "-"])
    has_cjk_path = bool(CJK_RE.search(item.get("path", "")))
    if has_cjk_path and not CJK_RE.search(text):
        try:
            rasterize_pdf(path, work)
        except RuntimeError as exc:
            if (source is None or item.get("profile") not in {
                    "libreoffice-pdf-office-v2", "libreoffice-pdf-office-xlsx-v3"
                    } or "blank interior page" not in str(exc)):
                raise
            candidate = normalized_office_pdf(source, work)
            shutil.move(candidate, path)
            rasterize_pdf(path, work)
        validate_output(path, "pdf")
        return
    if not embedded_pdf_fonts(path):
        outline_pdf_fonts(path, work)
        validate_output(path, "pdf")


def convert_item(item: dict, bundle: Path, reusable: dict | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="reader-convert-") as root:
        work = Path(root)
        source = work / f"source.{item['extension']}"
        digest, source_bytes = download_source(item["source_url"], source)
        if item["extension"] in {"rm", "rmvb"}:
            item = dict(item)
            item["source_media_mode"] = source_media_mode(source)
        if item["extension"] in {"mht", "mhtml"} and source_bytes > MAX_MHTML_SOURCE_BYTES:
            raise RuntimeError("MHTML source exceeds size limit")
        identity = reusable_object_key(
            digest, item["profile"], extension=item["extension"],
            source_revision=item["source_revision"], key=item["key"],
        )
        existing = (reusable or {}).get(identity)
        profile_path = object_profile_path(
            item["profile"], extension=item["extension"],
            source_revision=item["source_revision"], key=item["key"],
        )
        if item.get("output_name") == "page-manifest.json":
            profile_path = hashlib.sha256(
                f"{item['key']}\0{item['profile']}".encode("utf-8")
            ).hexdigest()[:16]
        reusable_path = existing.get("path") if existing else None
        object_path = reusable_path or (
            f"objects/{digest[:2]}/{digest}/{profile_path}/{item['output_name']}"
        )
        validate_storage_path(object_path)
        target = bundle / object_path
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = existing is not None
        with artifact_lock(target):
            if not target.exists():
                if existing:
                    asset_url = f"https://huggingface.co/datasets/{READER_ASSETS_REPO}/resolve/main/{existing['path']}"
                    download_existing(asset_url, target, existing["sha256"])
                    if target.stat().st_size != existing["bytes"]:
                        raise RuntimeError("reusable reader artifact size mismatch")
                    if item.get("output_name") == "page-manifest.json":
                        validate_page_manifest(target)
                    else:
                        validate_output(target, item["reader_mode"])
                    validate_reader_content(target, item, work)
                else:
                    temporary = work / item["output_name"]
                    if item.get("output_name") == "page-manifest.json":
                        convert_file(item, source, temporary, work, digest, object_path)
                    else:
                        convert_file(item, source, temporary, work)
                    if item["reader_mode"] == "epub" and item["extension"] != "chm":
                        sanitize_chm_epub(temporary, work)
                    if item.get("output_name") == "page-manifest.json":
                        validate_page_manifest(temporary)
                    else:
                        validate_output(temporary, item["reader_mode"])
                    if item["extension"] in {"odt", "rtf", "chm"}:
                        validate_html_content(temporary)
                    if item["extension"] == "djvu":
                        validate_djvu_pdf(temporary, work)
                    if (item["extension"] in {"doc", "docx", "htm", "html", "ppt", "pptx", "pps", "odp", "xls", "xlsx", "csv", "ods", "wps"}
                            and item["reader_mode"] == "pdf"
                            and item.get("output_name") != "page-manifest.json"):
                        validate_office_pdf(temporary, item, work, source)
                    validate_reader_content(temporary, item, work)
                    if item.get("output_name") == "page-manifest.json":
                        staged_pages = temporary.parent / "pages"
                        final_pages = target.parent / "pages"
                        if not staged_pages.is_dir():
                            raise RuntimeError("spreadsheet page stream is missing")
                        if final_pages.exists():
                            shutil.rmtree(final_pages)
                        shutil.move(staged_pages, final_pages)
                    shutil.move(temporary, target)
            else:
                if item.get("output_name") == "page-manifest.json":
                    validate_page_manifest(target)
                else:
                    validate_output(target, item["reader_mode"])
                if item["extension"] == "chm" and item["reader_mode"] == "epub":
                    validate_chm_epub(target)
                elif item["extension"] == "chm":
                    validate_html_content(target)
                validate_reader_content(target, item, work)
                if existing and file_sha256(target) != existing["sha256"]:
                    raise RuntimeError("reusable reader artifact digest mismatch")
        chapter_manifest_path = None
        chapter_bundle_error = None
        if needs_epub_chapters(item["extension"], item["reader_mode"], source_bytes):
            try:
                try:
                    from . import epub_chapters
                except ImportError:
                    import epub_chapters
                chapter_source = target
                if item["extension"] == "epub":
                    # Preserve the original spine, text and links. Normalizing
                    # an EPUB again can split chapters and alter source text.
                    chapter_source = source
                elif item["extension"] in {"mobi", "azw3", "fb2"}:
                    chapter_source = work / "chapter-source.epub"
                    run_checked(["ebook-convert", str(source), str(chapter_source), "--flow-size", "0"],
                                timeout_seconds=EPUB_COMMAND_TIMEOUT_SECONDS)
                    validate_output(chapter_source, "epub")
                # Calibre versions can produce different bytes under one
                # profile. Hash all resources so those builds never overwrite
                # each other's immutable URLs.
                staged_chapters = work / "chapter-bundle"
                epub_chapters.build_bundle(
                    chapter_source, staged_chapters,
                    include_all_documents=item["extension"] == "chm",
                )
                chapter_parent = (Path(*Path(object_path).parts[:3])
                                  / epub_chapters.bundle_version(staged_chapters)
                                  / f"{Path(object_path).parent.name}-{EPUB_CHAPTER_PROFILE}")
                chapter_dir = bundle / chapter_parent / EPUB_CHAPTER_BUNDLE_DIR
                with artifact_lock(chapter_dir):
                    if not (chapter_dir / "chapter-manifest.json").is_file():
                        chapter_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(staged_chapters, chapter_dir)
                chapter_manifest_path = (chapter_parent / EPUB_CHAPTER_BUNDLE_DIR
                                         / "chapter-manifest.json").as_posix()
            except Exception as exc:
                chapter_bundle_error = f"{type(exc).__name__}: {exc}"
                print(f"warning: {item['repo']}/{item['path']}: EPUB chapter bundle skipped: "
                      f"{chapter_bundle_error}")
        result = {
            "key": item["key"], "status": "ready", "source_revision": item["source_revision"],
            "source_sha256": digest, "source_bytes": source_bytes,
            "source_extension": item["extension"], "profile": item["profile"],
            "reader_mode": item["reader_mode"], "path": object_path, "bytes": target.stat().st_size,
        "sha256": file_sha256(target), "reused": reused,
        }
        if item["extension"] == "epub" and item["reader_mode"] == "pdf":
            result["fallback_path"] = object_path
        if item.get("output_name") == "page-manifest.json":
            result["page_stream"] = True
            result["page_count"] = json.loads(target.read_text(encoding="utf-8"))["page_count"]
        if chapter_manifest_path:
            result["chapter_manifest"] = chapter_manifest_path
            result["chapter_bundle_profile"] = EPUB_CHAPTER_PROFILE
        elif needs_epub_chapters(item["extension"], item["reader_mode"], source_bytes):
            result["chapter_bundle_profile"] = EPUB_CHAPTER_PROFILE
            result["chapter_bundle_error"] = chapter_bundle_error or "chapter bundle was not produced"
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("output/reader-assets/queue.json"))
    parser.add_argument("--bundle", type=Path, default=Path("output/reader-assets/bundle"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue_data = load_json(args.queue)
    queue = queue_data.get("items", [])
    reusable = {} if queue_data.get("force_rebuild") else queue_data.get("objects", {})
    if args.bundle.exists():
        shutil.rmtree(args.bundle)
    args.bundle.mkdir(parents=True)
    results = []
    if args.dry_run:
        print(f"dry run: would convert {len(queue)} reader asset(s)")
    else:
        def convert(item):
            try:
                return convert_item(item, args.bundle, reusable)
            except Exception as exc:
                print(f"failed: {item['repo']}/{item['path']}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return {
                    "key": item["key"], "status": "failed", "source_revision": item["source_revision"],
                    "source_extension": item["extension"], "profile": item["profile"],
                    "error": conversion_error_class(exc),
                }
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONVERSION_WORKERS) as executor:
            results.extend(executor.map(convert, queue))
    bundle_data = {
        "version": 1,
        "results": results,
        "force_rebuild": bool(queue_data.get("force_rebuild")),
        "bucket_migration": bool(queue_data.get("bucket_migration")),
    }
    if queue_data.get("authoritative_snapshot") is True:
        bundle_data["active_keys"] = queue_data.get("active_keys", [])
        bundle_data["authoritative_snapshot"] = True
    (args.bundle / "bundle.json").write_bytes(canonical_json(bundle_data, pretty=True))
    failed = sum(item["status"] == "failed" for item in results)
    print(f"converted {len(results) - failed}; failed {failed}")
    return 1 if results and failed == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
