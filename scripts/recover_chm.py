#!/usr/bin/env python3
"""Recover a CHM from its complete page inventory, with a per-page text oracle.

Explicit recovery tool; does not publish. Never executes source JavaScript.
Requires 7z plus the pinned Python dependencies in requirements.txt.
"""

import argparse
import base64
import hashlib
import html
import io
import json
import mimetypes
import posixpath
import re
import subprocess
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

from ebooklib import epub
from lxml import etree, html as HTML
from PIL import Image

try:
    from . import convert_reader_assets as converter
except ImportError:
    import convert_reader_assets as converter

PROFILE = "manual-chm-complete-v2"


def normalized(text):
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return re.sub(r"\s+", "", text)


def extract(source, root):
    listing = subprocess.run(["7z", "l", "-slt", str(source)], capture_output=True, timeout=120)
    info = listing.stdout.decode("utf-8", "replace")
    paths = re.findall(r"^Path = (.+)$", info, re.MULTILINE)[1:]
    expanded = sum(map(int, re.findall(r"^Size = (\d+)\s*$", info, re.MULTILINE)))
    listing_failed = bool(listing.returncode)
    if not listing_failed and (expanded > converter.MAX_CHM_EXPANDED_BYTES or len(paths) > converter.MAX_EPUB_MEMBERS):
        raise ValueError("CHM exceeds extraction bounds")
    if not listing_failed and any(p.startswith(("/", "\\")) or ".." in p.replace("\\", "/").split("/") for p in paths):
        raise ValueError("CHM contains unsafe member names")
    result = subprocess.run(["7z", "x", "-y", f"-o{root}", str(source)], capture_output=True, timeout=120)
    if result.returncode and not listing_failed:
        raise ValueError("CHM extraction incomplete")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("CHM contains symbolic links")
    extracted_files = [p for p in root.rglob("*") if p.is_file()]
    extracted_bytes = sum(p.stat().st_size for p in extracted_files)
    if len(extracted_files) > converter.MAX_EPUB_MEMBERS or extracted_bytes > converter.MAX_CHM_EXPANDED_BYTES:
        raise ValueError("CHM extracted content exceeds limits")
    if listing_failed and not extracted_files:
        raise ValueError("CHM listing and extraction both failed")


def source_pages(root, work):
    pages = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        script_content = None
        if suffix == '.js':
            try:
                from .chm_static_navigation import script_page
            except ImportError:
                from chm_static_navigation import script_page
            script_content = script_page(converter.decode_html_source(path), allow_writes=path.parent.name.lower()=='txt')
        if suffix not in {".htm", ".html", ".xhtml", ".txt", ".mht", ".mhtml"} and script_content is None:
            continue
        if path.stat().st_size > converter.MAX_EPUB_MEMBER_BYTES:
            raise ValueError("CHM page exceeds size limit")
        if suffix in {".mht", ".mhtml"}:
            temporary = work / "mhtml.html"
            converter.mhtml_to_html(path, temporary)
            text = temporary.read_text()
        else:
            text = converter.decode_html_source(path)
        if script_content is not None:
            text = '<html><head><title>'+html.escape(script_content[0])+'</title></head><body>'+script_content[1]+'</body></html>'
        # Old Windows HTML commonly declares Latin-1 while using Windows-1252
        # punctuation. Preserve the visible punctuation rather than C1 controls.
        def windows_punctuation(match):
            try:
                return match.group().encode("latin-1").decode("cp1252")
            except UnicodeDecodeError:
                return match.group()
        text = re.sub(r"[\x80-\x9f]", windows_punctuation, text)
        # XML 1.0 cannot represent these control characters. Record the removal
        # explicitly rather than letting the sanitizer turn them into '?'.
        text, invalid_controls = re.subn(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        static_writes = False
        if suffix == ".txt":
            expanded, static_writes = converter.expand_document_writes(text)
            if static_writes:
                text = expanded
            elif re.search(r"\bdocument\s*\.\s*write", text):
                raise ValueError(f"unsupported script-backed page: {path.relative_to(root)}")
            else:
                text = "<pre>" + html.escape(text) + "</pre>"
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        doc = HTML.document_fromstring(text or "<html><body></body></html>")
        etree.strip_elements(doc, etree.Comment, with_tail=False)
        scripts = 0
        for script in list(doc.xpath("//script")):
            scripts += 1
            expanded, found = converter.expand_document_writes(script.text or "")
            if found:
                wrapper = HTML.fragment_fromstring(expanded or "<span></span>", create_parent="div")
                wrapper.tail = script.tail
                script.getparent().replace(script, wrapper)
                static_writes = True
            else:
                script.drop_tree()
        for node in list(doc.xpath("//style|//iframe|//object|//embed")):
            node.drop_tree()
        title = doc.findtext(".//title") or path.stem
        body = doc.find("body")
        if body is None:
            body = doc
        for node in body.iter():
            for field in ("text", "tail"):
                value = getattr(node, field)
                if value:
                    setattr(node, field, re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value))
        expected = normalized(body.text_content())
        # Neutralize obsolete UI containers without discarding their labels.
        for node in list(body.iterdescendants()):
            if isinstance(node.tag, str) and node.tag.lower() not in converter.HTML_TAGS:
                node.drop_tag()
        for node in body.iter():
            for attribute in list(node.attrib):
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", attribute):
                    del node.attrib[attribute]
        safe = converter.sanitize_xml_document(etree.tostring(body, encoding="unicode", method="xml", with_tail=False))
        clean = etree.fromstring(safe.encode())
        if normalized("".join(clean.itertext())) != expected:
            actual = normalized("".join(clean.itertext()))
            difference = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
            raise ValueError(f"sanitization changes text: {path.relative_to(root)} at {difference}: "
                             f"{expected[difference:difference+80]!r} != {actual[difference:difference+80]!r}")
        pages.append({"path": path.relative_to(root).as_posix(), "title": title.strip(),
                      "body": clean, "expected": expected, "static_writes": static_writes, "scripts": scripts,
                      "removed_xml_controls": invalid_controls})
    return pages


def ordered_pages(pages, root):
    by_name = {p["path"].casefold(): p for p in pages}
    ordered, seen = [], set()
    for hhc in sorted(root.rglob("*")):
        if hhc.suffix.lower() != ".hhc":
            continue
        doc = HTML.document_fromstring(converter.decode_html_source(hhc))
        for node in doc.xpath("//param"):
            if node.get("name", "").lower() != "local":
                continue
            value = urllib.parse.unquote(node.get("value", "").split("#")[0]).replace("\\", "/")
            name = posixpath.normpath(posixpath.join(hhc.parent.relative_to(root).as_posix(), value)).casefold()
            if name in by_name and name not in seen:
                ordered.append(by_name[name])
                seen.add(name)
    ordered.extend(p for p in pages if p["path"].casefold() not in seen)
    return ordered


def recover(source, target, title):
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="chm-recovery-") as temporary:
        work = Path(temporary)
        root = work / "source"
        root.mkdir()
        extract(source, root)
        source_paths = {p.relative_to(root).as_posix().casefold(): p for p in root.rglob('*') if p.is_file()}
        pages = ordered_pages(source_pages(root, work), root)
        book = epub.EpubBook()
        book.set_identifier(source_sha)
        book.set_title(title)
        book.set_language("zh")
        names = {p["path"].casefold(): f"chapters/{i:05d}.xhtml" for i, p in enumerate(pages)}
        resource_names = {}
        missing_images = []
        checks = []
        for i, page in enumerate(pages):
            chapter = epub.EpubHtml(title=page["title"], file_name=names[page["path"].casefold()])
            for node in page["body"].xpath("//*[@src or @href]"):
                for attribute in ("src", "href"):
                    raw = node.get(attribute)
                    if not raw or raw.startswith("#"):
                        continue
                    parsed = urllib.parse.urlsplit(raw)
                    if parsed.scheme == "data" and node.tag == "img":
                        header, encoded = raw.split(",", 1)
                        data = base64.b64decode(encoded, validate=True)
                        mime = header[5:].split(";")[0]
                        key = hashlib.sha256(data).hexdigest()
                    elif parsed.scheme or parsed.netloc:
                        continue
                    else:
                        key = posixpath.normpath(posixpath.join(posixpath.dirname(page["path"]), urllib.parse.unquote(parsed.path).replace("\\", "/")))
                        if key.casefold() in names:
                            node.set(attribute, posixpath.relpath(names[key.casefold()], "chapters") + ("#" + parsed.fragment if parsed.fragment else ""))
                            continue
                        resource = root / key
                        if not resource.is_file():
                            resource = source_paths.get(key.casefold(), resource)
                        if not resource.resolve().is_relative_to(root.resolve()):
                            node.attrib.pop(attribute, None)
                            continue
                        if node.tag != "img":
                            continue
                        if not resource.is_file():
                            missing_images.append({"page": page["path"], "src": raw})
                            continue
                        data = resource.read_bytes()
                    try:
                        with Image.open(io.BytesIO(data)) as image:
                            mime = Image.MIME.get(image.format)
                            if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
                                converted = io.BytesIO()
                                image.convert("RGBA").save(converted, format="PNG")
                                data, mime = converted.getvalue(), "image/png"
                    except Exception as exc:
                        raise ValueError(f"unreadable source image: {page['path']} {raw}") from exc
                    if key not in resource_names:
                        ext = mimetypes.guess_extension(mime) or ".bin"
                        name = f"resources/{len(resource_names):05d}{ext}"
                        resource_names[key] = name
                        book.add_item(epub.EpubItem(uid=f"resource-{len(resource_names)}", file_name=name, media_type=mime, content=data))
                    node.set(attribute, "../" + resource_names[key])
            page["body"].tag = "div"
            chapter.content = HTML.tostring(page["body"], encoding="unicode")
            book.add_item(chapter)
            book.spine.append(chapter)
            book.toc.append(chapter)
            checks.append({"source": page["path"], "chapter": chapter.file_name, "chars": len(page["expected"]),
                           "text_sha256": hashlib.sha256(page["expected"].encode()).hexdigest(),
                           "static_writes": page["static_writes"], "scripts": page["scripts"],
                           "removed_xml_controls": page["removed_xml_controls"]})
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        target.parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(str(target), book, {})
        with zipfile.ZipFile(target) as archive:
            for check in checks:
                doc = etree.fromstring(archive.read("EPUB/" + check["chapter"]))
                body = doc.find("{http://www.w3.org/1999/xhtml}body")
                text = normalized("".join(body.itertext()))
                if hashlib.sha256(text.encode()).hexdigest() != check["text_sha256"]:
                    raise ValueError(f"EPUB changes page text: {check['source']}")
        try:
            from . import chm_navigation
        except ImportError:
            import chm_navigation
        navigation = chm_navigation.repair_epub(target, root,
            {check['source']: ['EPUB/' + check['chapter']] for check in checks})
        return {"source_sha256": source_sha, "profile": PROFILE, "chapters": checks, "navigation": navigation,
                "images": len(resource_names), "missing_images": missing_images,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    report = recover(args.source, args.target, args.title)
    args.target.with_suffix(".audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"chapters": len(report["chapters"]), "images": report["images"], "missing_images": len(report["missing_images"])}))
