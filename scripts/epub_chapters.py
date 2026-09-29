#!/usr/bin/env python3
"""Build a sanitized, independently fetchable EPUB chapter bundle."""

import hashlib
import gzip
import html
import json
import posixpath
import re
import zipfile
import copy
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

MAX_CHAPTER_RESOURCES = 2000
MAX_CHAPTER_RESOURCE_BYTES = 512 * 1024 * 1024
TARGET_CHAPTER_BYTES = 1024 * 1024
MAX_CHAPTER_BYTES = 8 * 1024 * 1024
MAX_CHAPTERS = 10000
MAX_SEARCH_INDEX_TEXT_BYTES = 256 * 1024 * 1024
MAX_SEARCH_INDEX_PACKED_BYTES = 64 * 1024 * 1024

try:
    from .convert_reader_assets import sanitize_css, sanitize_html, sanitize_xml_document
    from .reader_assets import canonical_json, validate_chapter_manifest
except ImportError:
    from convert_reader_assets import sanitize_css, sanitize_html, sanitize_xml_document
    from reader_assets import canonical_json, validate_chapter_manifest


def _zip_path(base: str, href: str) -> str:
    value = unquote(str(href or "").split("#", 1)[0])
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        raise ValueError("external EPUB resource")
    result = posixpath.normpath(posixpath.join(base, parsed.path))
    if result.startswith("../") or result == ".." or "\\" in result:
        raise ValueError("unsafe EPUB resource path")
    return result


def bundle_version(output: Path) -> str:
    """Include chapters and every resource in an immutable bundle identity."""
    digest = hashlib.sha256()
    for path in sorted(file for file in output.rglob('*') if file.is_file()):
        digest.update(path.relative_to(output).as_posix().encode('utf-8') + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def _safe_resource_path(path: str) -> str:
    def replace(match):
        value = match.group()
        return "_x00_" if value.lower() == "x00" else f"_x{ord(value):02x}_"

    return re.sub(r"x00|[\x00-\x1f\x7f]", replace, path, flags=re.IGNORECASE)


def _local_name(node) -> str:
    return node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""


def _node_text(node) -> str:
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def _toc_entries(archive: zipfile.ZipFile, opf_path: str, manifest: dict, names: set[str]) -> list[dict]:
    """Read the navigation tree, independently of physical spine splits."""
    base = posixpath.dirname(opf_path)
    candidates = sorted(manifest.values(), key=lambda item: "nav" not in item.get("properties", "").split())
    for item in candidates:
        media_type = item.get("media-type", "").lower()
        if media_type != "application/x-dtbncx+xml" and "nav" not in item.get("properties", "").split():
            continue
        try:
            toc_path = _zip_path(base, item.get("href", ""))
        except ValueError:
            continue
        if toc_path not in names:
            continue
        try:
            root = ET.fromstring(archive.read(toc_path))
        except ET.ParseError:
            continue
        entries = []
        def add(label, href, depth):
            if not href:
                return
            target = _zip_path(posixpath.dirname(toc_path), href)
            entries.append({"title": label, "source_path": target,
                            "fragment": unquote(urlsplit(href).fragment), "depth": depth})
        def ncx(parent, depth=0):
            for node in parent:
                if _local_name(node) != "navpoint":
                    continue
                content = next((child for child in node if _local_name(child) == "content"), None)
                label = next((child for child in node if _local_name(child) == "navlabel"), None)
                if content is not None:
                    add(_node_text(label) if label is not None else "", content.get("src"), depth)
                ncx(node, depth + 1)
        def nav_list(parent, depth=0):
            for li in parent:
                if _local_name(li) != "li":
                    continue
                anchor = next((child for child in li if _local_name(child) in {"a", "span"}), None)
                nested = [child for child in li if _local_name(child) == "ol"]
                if anchor is not None:
                    href = anchor.get("href")
                    # An unlinked group shares its first child's destination.
                    if not href:
                        first = next((child for ol in nested for child in ol.iter()
                                      if _local_name(child) == "a" and child.get("href")), None)
                        href = first.get("href") if first is not None else None
                    add(_node_text(anchor), href, depth)
                for ol in nested:
                    nav_list(ol, depth + 1)
        if media_type == "application/x-dtbncx+xml":
            for node in root.iter():
                if _local_name(node) == "navmap":
                    ncx(node)
        else:
            for nav in root.iter():
                kinds = nav.get("{http://www.idpf.org/2007/ops}type", "").split()
                if _local_name(nav) == "nav" and ("toc" in kinds or nav.get("role") == "doc-toc"):
                    for ol in nav:
                        if _local_name(ol) == "ol":
                            nav_list(ol)
        if entries:
            return entries
    return []


def _placeholder_title(title: str) -> bool:
    return not title.strip() or title.strip().lower() in {"unknown text", "untitled", "unknown", "无标题"}


def _target_title(document: str, fragment: str = "", *, root=None) -> str:
    if root is None:
        try:
            root = ET.fromstring(document)
        except ET.ParseError:
            return _document_title(document)
    body = next((node for node in root.iter() if _local_name(node) == "body"), root)
    nodes = list(body.iter())
    start = 0
    if fragment:
        start = next((i for i, node in enumerate(nodes)
                      if node.get("id") == fragment or node.get("name") == fragment), -1)
        if start < 0:
            return ""
        text = _node_text(nodes[start])
        if text and len(text) <= 500 and not _placeholder_title(text):
            return text
    for node in nodes[start:]:
        if _local_name(node) in {"h1", "h2", "h3", "h4", "h5", "h6", "p"}:
            text = _node_text(node)
            if text and len(text) <= 500 and not _placeholder_title(text):
                return text
    if not fragment:
        title = next((_node_text(node) for node in root.iter() if _local_name(node) == "title"), "")
        if not _placeholder_title(title):
            return title
    return ""


def bundle_toc(entries: list[dict], records: list[dict]) -> list[dict]:
    by_source = {record["source_path"]: record for record in reversed(records)}
    toc = []
    documents = {}
    for entry in entries:
        record = by_source.get(entry["source_path"])
        if record is None:
            raise ValueError(f'EPUB TOC target is outside readable spine: {entry["source_path"]}')
        if entry["source_path"] not in documents:
            try:
                documents[entry["source_path"]] = ET.fromstring(record["clean"])
            except ET.ParseError:
                documents[entry["source_path"]] = None
        title = entry["title"]
        if _placeholder_title(title):
            title = _target_title(record["clean"], entry["fragment"], root=documents[entry["source_path"]])
        if _placeholder_title(title):
            raise ValueError(f'EPUB TOC title cannot be recovered: {entry["source_path"]}#{entry["fragment"]}')
        fragment = entry["fragment"]
        root = documents.get(entry["source_path"])
        if fragment and root is not None and not any(
                node.get("id") == fragment or node.get("name") == fragment
                for node in root.iter()):
            # Keep a usable chapter destination when the source TOC points at
            # an anchor omitted by the source document or its sanitizer.
            fragment = ""
        toc.append({"title": title, "chapter": record["index"],
                    "fragment": fragment, "depth": entry["depth"]})
    return toc


def _can_share_resource(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp",
        ".eot", ".otf", ".ttf", ".woff", ".woff2",
    }


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


class _TitleExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.capture = False

    def handle_starttag(self, tag, attrs):
        if tag.rsplit(":", 1)[-1].lower() in {"title", "h1", "h2", "h3"} and not self.parts:
            self.capture = True

    def handle_endtag(self, tag):
        if tag.rsplit(":", 1)[-1].lower() in {"title", "h1", "h2", "h3"}:
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)


def _document_title(document: str) -> str:
    parser = _TitleExtractor()
    parser.feed(document)
    parser.close()
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def _chapter_text(document: str) -> str:
    parser = _TextExtractor()
    parser.feed(document)
    parser.close()
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def _split_spine_record(record: dict) -> list[dict]:
    """Split oversized spine documents at top-level XHTML elements."""
    root = ET.fromstring(record["clean"])
    body = next((node for node in root.iter() if _local_name(node) == "body"), None)
    if body is None:
        raise ValueError("EPUB chapter has no body")
    children = list(body)
    if not children:
        if len(record["clean"].encode("utf-8")) > MAX_CHAPTER_BYTES:
            raise ValueError("EPUB chapter exceeds the Reader chapter limit")
        return [record]
    chunks = []
    current = []
    wrapper = copy.deepcopy(root)
    wrapper_body = next(node for node in wrapper.iter() if _local_name(node) == "body")
    for node in list(wrapper_body):
        wrapper_body.remove(node)
    wrapper_bytes = len(ET.tostring(wrapper, encoding="utf-8"))
    current_bytes = 0
    for child in children:
        child_bytes = len(ET.tostring(child, encoding="utf-8"))
        if current and wrapper_bytes + current_bytes + child_bytes > TARGET_CHAPTER_BYTES:
            chunks.append(current)
            current = [child]
            current_bytes = child_bytes
        else:
            current.append(child)
            current_bytes += child_bytes
    if current:
        chunks.append(current)
    output = []
    for chunk in chunks:
        part = copy.deepcopy(root)
        part_body = next(node for node in part.iter() if _local_name(node) == "body")
        for node in list(part_body):
            part_body.remove(node)
        for node in chunk:
            part_body.append(copy.deepcopy(node))
        clean = ET.tostring(part, encoding="unicode", method="xml")
        size = len(clean.encode("utf-8"))
        if size > MAX_CHAPTER_BYTES:
            raise ValueError("single EPUB spine element exceeds the Reader chapter limit")
        output.append({**record, "clean": clean})
    return output


def build_bundle(epub: Path, output: Path, *, fallback: str | None = None,
                 include_resources: bool = True) -> dict:
    """Write chapter files and return the validated manifest.

    The output directory contains only files intended for a dataset commit.
    """
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(epub) as archive:
        names = set(archive.namelist())
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next((node for node in container.iter() if _local_name(node) == "rootfile"), None)
        if rootfile is None:
            raise ValueError("EPUB package is missing")
        opf_path = rootfile.attrib.get("full-path", "")
        opf = ET.fromstring(archive.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {}
        for node in opf.iter():
            if _local_name(node) == "item":
                manifest[node.attrib.get("id", "")] = node.attrib
        chapters = []
        search_chapters = []
        search_text_bytes = 0
        chapter_records = []
        resource_usage = Counter()
        toc_entries = _toc_entries(archive, opf_path, manifest, names)
        toc_titles = {}
        for entry in toc_entries:
            if not _placeholder_title(entry["title"]):
                toc_titles.setdefault(entry["source_path"], entry["title"])
        for number, ref in enumerate((n for n in opf.iter() if _local_name(n) == "itemref"), 1):
            item = manifest.get(ref.attrib.get("idref"))
            if (not item or "nav" in item.get("properties", "").split()
                    or item.get("media-type", "").lower() not in {"application/xhtml+xml", "text/html"}):
                continue
            source_path = _zip_path(base, item.get("href", ""))
            if source_path not in names:
                # Keep readable chapters when a broken package has one stale spine entry.
                continue
            document = archive.read(source_path).decode("utf-8", "replace")
            try:
                clean = sanitize_xml_document(document)
            except RuntimeError as exc:
                if str(exc) != "EPUB XML content is malformed":
                    raise
                # Some EPUBs label HTML as XHTML but contain recoverable HTML.
                clean = sanitize_html(document, allow_relative=True)
            chapter_index = len(chapter_records) + 1
            resources = set()
            def rewrite(match):
                value = match.group(2)
                try:
                    resource = _zip_path(posixpath.dirname(source_path), value)
                except ValueError:
                    return match.group(0)
                if resource not in names or resource.lower().endswith((".xhtml", ".html", ".htm")):
                    return match.group(0)
                resources.add(resource)
                safe_resource = _safe_resource_path(resource)
                return f'{match.group(1)}="../resources/__CHAPTER_RESOURCE__/{safe_resource}"'
            clean = re.sub(r'((?:src|href))=["\']([^"\'#]+)["\']', rewrite, clean, flags=re.I)
            title = toc_titles.get(source_path) or _target_title(clean) or _document_title(clean) or f"章节 {number}"
            chapter_records.append({"index": chapter_index, "source_path": source_path,
                                    "title": title, "clean": clean, "resources": resources})
            resource_usage.update(resources)
        if not chapter_records:
            raise ValueError("EPUB spine has no readable chapters")
        split_records = []
        for record in chapter_records:
            safe_resources = {_safe_resource_path(resource): resource for resource in record["resources"]}
            for split in _split_spine_record(record):
                split["resources"] = {
                    resource for safe, resource in safe_resources.items()
                    if f"../resources/__CHAPTER_RESOURCE__/{safe}" in split["clean"]
                }
                split_records.append(split)
        if len(split_records) > MAX_CHAPTERS:
            raise ValueError("EPUB chapter count exceeds Reader limit")
        chapter_records = split_records
        resource_usage = Counter(resource for record in chapter_records for resource in record["resources"])
        for index, record in enumerate(chapter_records, 1):
            record["index"] = index
        toc = bundle_toc(toc_entries, chapter_records)
        # Resolve links only after every readable spine item has its final name.
        # Original filenames cannot be used after chapters move into the bundle.
        chapter_paths = {}
        for record in chapter_records:
            chapter_paths.setdefault(record["source_path"], f'chapter-{record["index"]:04d}.xhtml')
        for record in chapter_records:
            chapter_index = record["index"]
            clean = record["clean"]
            def rewrite_chapter_link(match):
                value = html.unescape(match.group(2))
                if value.startswith("#"):
                    return match.group(0)
                try:
                    original = _zip_path(posixpath.dirname(record["source_path"]), value)
                    fragment = urlsplit(value).fragment
                except ValueError:
                    return match.group(0)
                target = chapter_paths.get(original)
                if target is None:
                    return match.group(0)
                if fragment:
                    target += "#" + fragment
                return f'{match.group(1)}="{html.escape(target, quote=True)}"'
            clean = re.sub(r'(?<![\w:-])(href)\s*=\s*["\']([^"\']*)["\']',
                           rewrite_chapter_link, clean, flags=re.I)
            resources = record["resources"]
            if include_resources:
                resource_bytes = sum(archive.getinfo(resource).file_size for resource in resources)
                if (len(resources) > MAX_CHAPTER_RESOURCES
                        or resource_bytes > MAX_CHAPTER_RESOURCE_BYTES):
                    raise ValueError("EPUB chapter resource budget exceeded")
                for resource in sorted(resources):
                    prefix = "shared" if resource_usage[resource] > 1 and _can_share_resource(resource) else f"chapter-{chapter_index:04d}"
                    target = output / "resources" / prefix / _safe_resource_path(resource)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.read(resource)
                    if resource.lower().endswith(".css"):
                        data = sanitize_css(data.decode("utf-8", "replace")).encode("utf-8")
                    if not target.exists():
                        target.write_bytes(data)
                    clean = clean.replace(
                        f"../resources/__CHAPTER_RESOURCE__/{_safe_resource_path(resource)}",
                        f"../resources/{prefix}/{_safe_resource_path(resource)}",
                    )
            target = output / "chapters" / f"chapter-{chapter_index:04d}.xhtml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(clean, encoding="utf-8")
            data = target.read_bytes()
            chapters.append({"index": chapter_index, "title": record["title"], "source_path": record["source_path"], "path": target.relative_to(output).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            chapter_text = _chapter_text(clean)
            search_text_bytes += len(chapter_text.encode("utf-8"))
            if search_text_bytes > MAX_SEARCH_INDEX_TEXT_BYTES:
                raise ValueError("EPUB full-text index exceeds Reader limit")
            search_chapters.append({"index": chapter_index, "title": record["title"], "path": target.relative_to(output).as_posix(), "text": chapter_text})
    search_data = canonical_json({"version": 1, "kind": "epub-search-index", "chapters": search_chapters})
    if len(search_data) > MAX_SEARCH_INDEX_TEXT_BYTES:
        raise ValueError("EPUB full-text index exceeds Reader limit")
    search_bytes = gzip.compress(search_data, mtime=0)
    if len(search_bytes) > MAX_SEARCH_INDEX_PACKED_BYTES:
        raise ValueError("EPUB compressed search index exceeds Reader limit")
    search_target = output / "epub-search-index.json.gz"
    search_target.write_bytes(search_bytes)
    result = {"version": 1, "kind": "epub-chapters", "chapters": chapters, "search_index": {"path": search_target.relative_to(output).as_posix(), "bytes": len(search_bytes), "sha256": hashlib.sha256(search_bytes).hexdigest()}}
    result["toc"] = toc
    if fallback:
        result["fallback"] = fallback
    validate_chapter_manifest(result)
    (output / "chapter-manifest.json").write_bytes(canonical_json(result, pretty=True))
    return result
