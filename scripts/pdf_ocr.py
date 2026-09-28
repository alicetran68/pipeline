#!/usr/bin/env python3
"""Stable page-level PDF OCR assets for the VoiceOfML Reader.

The module is intentionally usable without PaddleOCR installed.  Planning,
manifest validation and native PDF text extraction are kept in the small
pipeline environment; only ``ocr_page`` imports PaddleOCR.  This lets queue
planning and publication tests run without a model download.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path

try:
    from . import pdf_assets, reader_assets, shared
except ImportError:
    import pdf_assets
    import reader_assets
    import shared


OCR_MANIFEST_NAME = "ocr-manifest.json"
OCR_MANIFEST_VERSION = 1
OCR_PAGE_VERSION = 1
OCR_PROFILE = "pdf-ocr-v1-pp-ocrv6-medium"
OCR_LANG = os.environ.get("PDF_OCR_LANG", "auto").strip().lower() or "auto"
OCR_VERSION_OVERRIDE = os.environ.get("PDF_OCR_VERSION", "").strip() or None
OCR_BACKEND = os.environ.get("PDF_OCR_BACKEND", "rapidocr_onnxruntime").strip().lower() or "rapidocr_onnxruntime"
OCR_BACKENDS = frozenset({"paddle_static", "paddle_onnxruntime", "rapidocr_onnxruntime"})
PP_OCRV6_LANGS = frozenset({
    "ch", "chinese_cht", "en", "japan", "af", "az", "bs", "ca", "cs", "cy", "da", "de",
    "es", "et", "eu", "fi", "fr", "ga", "gl", "hr", "hu", "id", "is", "it", "ku", "la",
    "lb", "lt", "lv", "mi", "ms", "mt", "nl", "no", "oc", "pl", "pt", "qu", "rm", "ro",
    "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz", "vi", "french", "german",
})
PP_OCRV5_LANGS = frozenset({
    "ch", "en", "fr", "de", "japan", "korean", "chinese_cht", "af", "it", "es", "bs", "pt",
    "cs", "cy", "da", "et", "ga", "hr", "hu", "rs_latin", "id", "oc", "is", "lt", "mi",
    "ms", "nl", "no", "pl", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz", "la", "ru",
    "be", "uk", "th", "el", "az", "ku", "lv", "mt", "pi", "ro", "vi", "fi", "eu", "gl",
    "lb", "rm", "ca", "qu", "te", "sr", "bg", "mn", "ab", "ady", "kbd", "av", "dar", "inh",
    "ce", "lki", "lez", "tab", "kk", "ky", "tg", "mk", "tt", "cv", "ba", "mhr", "mo", "udm",
    "kv", "os", "bua", "xal", "tyv", "sah", "kaa", "ar", "fa", "ug", "ur", "ps", "sd", "bal",
    "hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom", "sa", "bgc", "ta",
})


def ocr_version() -> str:
    version = OCR_VERSION_OVERRIDE or ("PP-OCRv6" if OCR_LANG in PP_OCRV6_LANGS else "PP-OCRv5")
    if version not in {"PP-OCRv5", "PP-OCRv6"}:
        raise ValueError(f"unsupported PDF_OCR_VERSION: {version}")
    supported = PP_OCRV6_LANGS if version == "PP-OCRv6" else PP_OCRV5_LANGS
    if OCR_LANG not in supported:
        raise ValueError(f"language {OCR_LANG!r} is not supported by {version}")
    return version


if OCR_BACKEND not in OCR_BACKENDS:
    raise ValueError(f"unsupported PDF_OCR_BACKEND: {OCR_BACKEND}")
OCR_VERSION = ocr_version() if OCR_LANG != "auto" else "auto"
OCR_ENGINE = f"{OCR_BACKEND} / {OCR_VERSION} / CPU / {OCR_LANG}"
OCR_DPI = int(os.environ.get("PDF_OCR_DPI", "300"))
WEBP_QUALITY = int(os.environ.get("PDF_WEBP_QUALITY", "85"))
WEBP_MAX_DIMENSION = int(os.environ.get("PDF_WEBP_MAX_DIMENSION", "1800"))
JXL_ENABLED = os.environ.get("PDF_JXL_ENABLED", "0").lower() in {"1", "true", "yes"}
JXL_DISTANCE = float(os.environ.get("PDF_JXL_DISTANCE", "1.5"))
JXL_EFFORT = int(os.environ.get("PDF_JXL_EFFORT", "7"))
MIN_NATIVE_PAGE_CHARS = int(os.environ.get("PDF_OCR_MIN_NATIVE_PAGE_CHARS", "48"))
NATIVE_PAGE_RATIO = float(os.environ.get("PDF_OCR_NATIVE_PAGE_RATIO", "0.90"))
MAX_PAGE_PIXELS = int(os.environ.get("PDF_OCR_MAX_PAGE_PIXELS", "50000000"))
COMMAND_TIMEOUT = int(os.environ.get("PDF_OCR_COMMAND_TIMEOUT", "600"))
OCR_TIMEOUT = int(os.environ.get("PDF_OCR_PAGE_TIMEOUT", "300"))
MI = 1024 * 1024
OCR_OBJECT_PATH_RE = re.compile(
    r"^objects/[0-9a-f]{2}/[0-9a-f]{64}/[0-9a-f]{16}/"
    r"(?:ocr-manifest\.json|page-manifest\.json|render-manifest\.json|"
    r"render-range-[0-9]{6}-[0-9]{6}\.json|"
    r"pages/page-[0-9]{6}\.(?:webp|jxl)|"
    r"ocr-input/page-[0-9]{6}\.png|"
    r"ocr/page-[0-9]{6}\.json\.gz|ocr/book-text\.json\.gz)$"
)


def resolve_ocr_config(language=None, backend=None):
    language = (language or OCR_LANG).strip().lower()
    backend = (backend or OCR_BACKEND).strip().lower()
    if language == "auto":
        language = "ch"
    if backend == "auto":
        backend = "rapidocr_onnxruntime" if language in {"ch", "en"} else "paddle_onnxruntime"
    if backend not in OCR_BACKENDS:
        raise ValueError(f"unsupported PDF_OCR_BACKEND: {backend}")
    if backend == "rapidocr_onnxruntime" and language not in {"ch", "en"}:
        backend = "paddle_onnxruntime"
    version = OCR_VERSION_OVERRIDE or ("PP-OCRv6" if language in PP_OCRV6_LANGS else "PP-OCRv5")
    supported = PP_OCRV6_LANGS if version == "PP-OCRv6" else PP_OCRV5_LANGS
    if version not in {"PP-OCRv5", "PP-OCRv6"} or language not in supported:
        raise ValueError(f"language {language!r} is not supported by {version}")
    return language, version, backend


def detect_language(value: str) -> str:
    text = str(value or "")
    counts = {
        "ch": sum("\u3400" <= c <= "\u9fff" for c in text),
        "japan": sum(("\u3040" <= c <= "\u30ff") for c in text),
        "korean": sum("\uac00" <= c <= "\ud7af" for c in text),
        "ru": sum(("\u0400" <= c <= "\u052f") for c in text),
        "ar": sum("\u0600" <= c <= "\u06ff" for c in text),
        "en": sum(("A" <= c <= "Z") or ("a" <= c <= "z") for c in text),
    }
    non_latin = {key: value for key, value in counts.items() if key != "en" and value}
    strongest = max(non_latin, key=non_latin.get) if non_latin else "en"
    if counts[strongest] < 2 and not non_latin:
        return "ch"
    if strongest == "ar":
        lowered = text.casefold()
        return "fa" if any(word in lowered for word in ("iran", "persian", "farsi", "ایران", "فارسی")) else "ar"
    return strongest


def book_ocr_config(book: dict):
    language = book.get("ocr_language") or detect_language(book.get("key", ""))
    backend = book.get("ocr_backend") or ("rapidocr_onnxruntime" if language in {"ch", "en"} else "paddle_onnxruntime")
    language, _, backend = resolve_ocr_config(language, backend)
    return language, backend


def asset_profile(language=None, backend=None) -> str:
    language, version, backend = resolve_ocr_config(language, backend)
    model_profile = OCR_PROFILE if language == "ch" and version == "PP-OCRv6" and backend == "paddle_static" else (
        f"pdf-ocr-v1-{version.lower()}-medium-lang-{language}")
    if backend != "paddle_static":
        model_profile += f"-backend-{backend.replace('_', '-')}"
    return (f"{model_profile}-dpi-{OCR_DPI}-webp-{WEBP_QUALITY}-{WEBP_MAX_DIMENSION}"
            f"-native-{MIN_NATIVE_PAGE_CHARS}-{NATIVE_PAGE_RATIO:g}-maxpix-{MAX_PAGE_PIXELS}"
            f"-jxl-{int(JXL_ENABLED)}-{JXL_DISTANCE:g}-{JXL_EFFORT}")


def valid_published_profile(profile: str) -> bool:
    candidate = str(profile or "")
    base, separator, layout = candidate.partition("-layout-")
    if separator and not re.fullmatch(r"v1-[0-9a-f]{16}", layout):
        return False
    return bool(re.fullmatch(
        r"pdf-ocr-v1-pp-ocrv[56]-medium"
        r"(?:-lang-[a-z0-9_]+)?"
        r"(?:-backend-(?:paddle-onnxruntime|rapidocr-onnxruntime))?"
        r"-dpi-[0-9]+-webp-[0-9]+-[0-9]+"
        r"-native-[0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?"
        r"-maxpix-[0-9]+-jxl-[01]-[0-9]+(?:\.[0-9]+)?-[0-9]+",
        base,
    ))


def ocr_profile_without_jxl(profile: str) -> str:
    return str(profile or "").split("-jxl-", 1)[0]


def object_root(source_sha: str, key: str, language=None, backend=None) -> Path:
    key_sha = hashlib.sha256(f"{key}\0{asset_profile(language, backend)}".encode("utf-8")).hexdigest()[:16]
    return Path("objects") / source_sha[:2] / source_sha / key_sha


def _run(command: list[str], *, timeout: int = COMMAND_TIMEOUT) -> str:
    try:
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timeout: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-1000:]
        raise RuntimeError(f"command failed: {command[0]}: {detail}") from exc
    return result.stdout


def pdf_page_count(path: Path) -> int:
    for line in _run(["pdfinfo", str(path)]).splitlines():
        if line.startswith("Pages:"):
            value = int(line.split(":", 1)[1].strip())
            if value < 1:
                break
            return value
    raise RuntimeError("PDF page count unavailable")


def clean_text(value: str) -> str:
    value = html.unescape(str(value or "")).replace("\x00", "")
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


class _BBoxParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.words: list[dict] = []
        self.page_width = 0.0
        self.page_height = 0.0
        self._word: dict | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag == "page":
            self.page_width = float(attrs.get("width") or 0)
            self.page_height = float(attrs.get("height") or 0)
        elif tag == "word":
            self._word = {
                "x0": float(attrs.get("xMin") or 0), "y0": float(attrs.get("yMin") or 0),
                "x1": float(attrs.get("xMax") or 0), "y1": float(attrs.get("yMax") or 0),
            }
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._word is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "word" and self._word is not None:
            text = clean_text("".join(self._parts))
            if text:
                self.words.append({**self._word, "text": text, "confidence": 1.0, "source": "native"})
            self._word = None
            self._parts = []


def native_page(path: Path, page: int) -> dict:
    """Extract words and coordinates from the PDF text layer."""
    raw = _run(["pdftotext", "-bbox-layout", "-f", str(page), "-l", str(page), str(path), "-"])
    parser = _BBoxParser()
    parser.feed(raw)
    words = parser.words
    # pdftotext's y origin is top-left, matching the browser overlay.
    lines: list[str] = []
    previous_y = None
    current: list[str] = []
    for word in words:
        y = word["y0"]
        if previous_y is not None and abs(y - previous_y) > 4 and current:
            lines.append(" ".join(current))
            current = []
        current.append(word["text"])
        previous_y = y
    if current:
        lines.append(" ".join(current))
    text = "\n".join(lines).strip()
    return {
        "page": page, "status": "ready", "source": "native", "text": text,
        "blocks": normalize_blocks(words, parser.page_width, parser.page_height),
        "width": parser.page_width, "height": parser.page_height,
    }


def page_text_probe(path: Path, page: int) -> int:
    for mode in ([], ["-raw"], ["-layout"]):
        raw = _run(["pdftotext", *mode, "-f", str(page), "-l", str(page),
                    "-enc", "UTF-8", str(path), "-"])
        count = len(re.sub(r"\s+", "", clean_text(raw)))
        if count:
            return count
    return 0


def document_text_probe(path: Path, page_count: int) -> list[int]:
    """Extract a document's text once, then split it into PDF pages.

    Starting Poppler once per page makes planning multi-thousand-page books
    needlessly slow.  Keep the alternate extraction modes because repaired
    CJK fonts may only decode through ``-raw`` or ``-layout`` on some Poppler
    builds.
    """
    for mode in ([], ["-raw"], ["-layout"]):
        try:
            raw = _run(["pdftotext", *mode, "-enc", "UTF-8", str(path), "-"])
        except RuntimeError:
            continue
        pages = raw.split("\f")[:page_count]
        counts = [len(re.sub(r"\s+", "", clean_text(text))) for text in pages]
        if any(counts) or not raw:
            return counts + [0] * (page_count - len(counts))
    return [0] * page_count


def probe_pdf(path: Path) -> dict:
    page_count = pdf_page_count(path)
    page_chars = document_text_probe(path, page_count)
    native_pages = sum(chars >= MIN_NATIVE_PAGE_CHARS for chars in page_chars)
    ratio = native_pages / page_count
    if native_pages == page_count:
        classification = "native-text"
    elif native_pages == 0:
        classification = "scan"
    else:
        classification = "mixed"
    return {
        "page_count": page_count, "page_chars": page_chars,
        "native_pages": native_pages, "native_page_ratio": ratio,
        "classification": classification,
    }


def normalize_blocks(blocks, width: float, height: float) -> list[dict]:
    width = max(1.0, float(width or 1))
    height = max(1.0, float(height or 1))
    output = []
    for block in blocks or []:
        text = clean_text(block.get("text", ""))
        if not text:
            continue
        x0 = max(0.0, min(1.0, float(block.get("x0", 0)) / width))
        y0 = max(0.0, min(1.0, float(block.get("y0", 0)) / height))
        x1 = max(x0, min(1.0, float(block.get("x1", 0)) / width))
        y1 = max(y0, min(1.0, float(block.get("y1", 0)) / height))
        output.append({"t": text, "b": [x0, y0, x1, y1],
                       "c": round(max(0.0, min(1.0, float(block.get("confidence", 1)))), 4),
                       "s": block.get("source", "ocr"),
                       **({"q": [[max(0, min(1, x / width)), max(0, min(1, y / height))]
                                 for x, y in block["polygon"]]} if block.get("polygon") else {})})
    output.sort(key=lambda item: (item["b"][1], item["b"][0]))
    return output


def validate_object_path(path: str) -> str:
    if (not isinstance(path, str) or not path.startswith("objects/") or "\\" in path
            or path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise ValueError("invalid PDF OCR object path")
    return path


def validate_ocr_object_path(path: str, suffix: str | None = None) -> str:
    """Validate an OCR object path before it is written to a manifest."""
    validate_object_path(path)
    if not OCR_OBJECT_PATH_RE.fullmatch(path) or (suffix and not path.endswith(suffix)):
        raise ValueError("invalid PDF OCR object path")
    return path


def normalize_ocr_result(result, width: int, height: int) -> list[dict]:
    """Normalize PaddleOCR 3.x result objects and JSON fixtures alike."""
    raw = getattr(result, "json", result)
    if callable(raw):
        raw = raw()
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and isinstance(raw.get("res"), dict):
        raw = raw["res"]
    if not isinstance(raw, dict):
        raise RuntimeError("PP-OCRv6 returned an invalid result")
    def sequence(*names):
        for name in names:
            value = raw.get(name)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if value is not None and len(value):
                return [value] if isinstance(value, str) else value
        return []
    texts = sequence("rec_texts", "rec_text")
    scores = sequence("rec_scores")
    polys = sequence("rec_polys", "dt_polys", "rec_boxes")
    blocks = []
    for index, text in enumerate(texts):
        text = clean_text(text)
        if not text:
            continue
        polygon = polys[index] if index < len(polys) else []
        if len(polygon) == 4 and all(isinstance(v, (float, int)) for v in polygon):
            x0, y0, x1, y1 = polygon
            polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        points = []
        for point in polygon or []:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if points:
            x0, y0 = min(point[0] for point in points), min(point[1] for point in points)
            x1, y1 = max(point[0] for point in points), max(point[1] for point in points)
        else:
            x0 = y0 = x1 = y1 = 0
        confidence = scores[index] if index < len(scores) else 0.0
        blocks.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                       "confidence": confidence, "source": "ocr", "polygon": points})
    return normalize_blocks(blocks, width, height)


def normalize_rapid_result(result, width: int, height: int) -> list[dict]:
    """Adapt RapidOCR's boxes/txts/scores output to the shared OCR contract."""
    if isinstance(result, tuple):
        result = result[0]
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if isinstance(result, dict):
        boxes = result.get("boxes", boxes)
        texts = result.get("txts", result.get("texts", texts))
        scores = result.get("scores", scores)
    if boxes is None or texts is None:
        # RapidOCR reports a normal blank page as an empty detection result.
        # Keep the page in the v2 index instead of turning it into a failed task.
        return []
    blocks = []
    score_values = scores if scores is not None else [1.0] * len(texts)
    for polygon, text, score in zip(boxes, texts, score_values):
        points = [[float(point[0]), float(point[1])] for point in polygon]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        blocks.append({"text": text, "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys),
                       "confidence": score, "source": "ocr", "polygon": points})
    return normalize_blocks(blocks, width, height)


_OCR_ENGINE_INSTANCES = {}


def get_ocr_engine(language=None, backend=None):
    language, version, backend = resolve_ocr_config(language, backend)
    cache_key = (language, version, backend)
    if cache_key not in _OCR_ENGINE_INSTANCES:
        if backend == "rapidocr_onnxruntime":
            try:
                from rapidocr import EngineType, LangDet, LangRec, OCRVersion, RapidOCR
            except ImportError as exc:
                raise RuntimeError("RapidOCR ONNX dependencies are not installed") from exc
            rapid_version = getattr(OCRVersion, version.replace("-", "").upper())
            rapid_det_lang = getattr(LangDet, language.upper())
            rapid_rec_lang = getattr(LangRec, language.upper())
            _OCR_ENGINE_INSTANCES[cache_key] = RapidOCR(params={
                "Det.engine_type": EngineType.ONNXRUNTIME, "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Rec.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": rapid_det_lang,
                "Rec.lang_type": rapid_rec_lang, "Det.ocr_version": rapid_version,
                "Rec.ocr_version": rapid_version,
            })
        else:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("PaddleOCR dependencies are not installed") from exc
            _OCR_ENGINE_INSTANCES[cache_key] = PaddleOCR(
                lang=language, ocr_version=version, device="cpu",
                engine="onnxruntime" if backend == "paddle_onnxruntime" else "paddle_static",
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False,
            )
    return _OCR_ENGINE_INSTANCES[cache_key]


def _page_render_dpi(path: Path, page: int) -> int:
    """Choose a DPI that keeps unusually large PDF pages within the OCR budget."""
    try:
        info = _run(["pdfinfo", "-f", str(page), "-l", str(page), "-box", str(path)])
        match = re.search(r"Page(?:\s+\d+)?\s+size:\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info)
        if not match:
            return OCR_DPI
        width_points, height_points = (float(value) for value in match.groups())
        area = width_points * height_points
        if area <= 0:
            return OCR_DPI
        max_dpi = int(72 * (MAX_PAGE_PIXELS / area) ** 0.5)
        return max(1, min(OCR_DPI, max_dpi))
    except Exception:
        return OCR_DPI


def scan_reader_images(path: Path, start: int, end: int) -> dict[int, tuple[int, int]]:
    """Find unambiguous, nearly full-page raster scans and their source pixels."""
    try:
        listing = _run(["pdfimages", "-f", str(start), "-l", str(end), "-list", str(path)])
    except RuntimeError:
        return {}
    images: dict[int, list[tuple[int, int, int, int]]] = {}
    for line in listing.splitlines()[2:]:
        parts = line.split()
        if len(parts) < 14 or not parts[0].isdigit():
            continue
        page = int(parts[0])
        if not start <= page <= end:
            continue
        if parts[2] != "image":
            images.setdefault(page, []).append((0, 0, 0, 0))
            continue
        try:
            image = tuple(int(parts[index]) for index in (3, 4, 12, 13))
        except ValueError:
            continue
        images.setdefault(page, []).append(image)
    result = {}
    for page, candidates in images.items():
        if len(candidates) != 1:
            continue
        width, height, x_ppi, y_ppi = candidates[0]
        if min(width, height, x_ppi, y_ppi) < 1:
            continue
        try:
            info = _run(["pdfinfo", "-f", str(page), "-l", str(page), "-box", str(path)])
            match = re.search(r"Page(?:\s+\d+)? size:\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info)
            if not match:
                continue
            page_width, page_height = (float(value) / 72 for value in match.groups())
            coverage = (width / x_ppi / page_width, height / y_ppi / page_height)
            if all(0.85 <= value <= 1.1 for value in coverage):
                result[page] = (width, height)
        except (RuntimeError, ZeroDivisionError):
            continue
    return result


def reader_webp_quality(image, full_page_scan: bool) -> int:
    if not full_page_scan:
        return WEBP_QUALITY
    sample = image.resize((128, 128))
    pixels = sample.tobytes()
    bright = dark = colorful = 0
    for offset in range(0, len(pixels), 3):
        r, g, b = pixels[offset:offset + 3]
        bright += min(r, g, b) > 235
        dark += max(r, g, b) < 96
        colorful += max(r, g, b) - min(r, g, b) > 22
    count = sample.width * sample.height
    return min(WEBP_QUALITY, 80) if (bright / count >= .7 and dark / count <= .12
                                    and colorful / count <= .03) else WEBP_QUALITY


def render_page(path: Path, page: int, directory: Path,
                reader_pixels: tuple[int, int] | None = None, reader_jxl: bool = False) -> tuple[Path, int, int]:
    prefix = directory / f"page-{page:06d}"
    _run([
        "pdftocairo", "-png", "-singlefile", "-r", str(_page_render_dpi(path, page)),
        "-f", str(page), "-l", str(page), str(path), str(prefix),
    ], timeout=COMMAND_TIMEOUT)
    png = prefix.with_suffix(".png")
    if not png.is_file():
        raise RuntimeError(f"page {page} render missing")
    from PIL import Image
    with Image.open(png) as image:
        width, height = image.size
        rgb = image.convert("RGB")
        if width * height > MAX_PAGE_PIXELS:
            scale = (MAX_PAGE_PIXELS / (width * height)) ** 0.5
            rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))))
            width, height = rgb.size
            rgb.save(png, "PNG")
        if reader_pixels is None and not reader_jxl:
            # Preserve the older single-stage profile's delivery bytes.
            if WEBP_MAX_DIMENSION and max(width, height) > WEBP_MAX_DIMENSION:
                scale = WEBP_MAX_DIMENSION / max(width, height)
                resized = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))))
                rgb.close()
                rgb = resized
        else:
            scale = min(1.0, WEBP_MAX_DIMENSION / max(width, height)) if WEBP_MAX_DIMENSION else 1.0
            if reader_pixels:
                scale = min(scale, reader_pixels[0] / width, reader_pixels[1] / height)
            if scale < 1:
                reader_width, reader_height = max(1, int(width * scale)), max(1, int(height * scale))
                if reader_pixels:
                    reader_width = min(reader_width, reader_pixels[0])
                    reader_height = min(reader_height, reader_pixels[1])
                resized = rgb.resize((reader_width, reader_height), Image.Resampling.LANCZOS)
                rgb.close()
                rgb = resized
        webp = prefix.with_suffix(".webp")
        rgb.save(webp, "WEBP", quality=reader_webp_quality(rgb, reader_pixels is not None), method=6)
        if reader_jxl and rgb.size != (width, height):
            rgb.save(prefix.with_name(prefix.name + "-reader").with_suffix(".png"), "PNG")
        rgb.close()
    return png, width, height


def encode_jxl(png: Path, destination: Path) -> tuple[str, int]:
    """Encode an optional reader stream from a lossless PNG input."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(["cjxl", str(png), str(destination), "-d", str(JXL_DISTANCE), "-e", str(JXL_EFFORT)],
         timeout=COMMAND_TIMEOUT)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("cjxl produced no output")
    return shared.hash_file(destination)


def ocr_page(image: Path, width: int, height: int, language=None, backend=None) -> list[dict]:
    started = time.monotonic()
    _, _, backend = resolve_ocr_config(language, backend)
    engine = get_ocr_engine(language, backend)
    if backend == "rapidocr_onnxruntime":
        result = normalize_rapid_result(engine(str(image), use_det=True, use_cls=False, use_rec=True), width, height)
        if time.monotonic() - started > OCR_TIMEOUT:
            raise RuntimeError("RapidOCR page timeout")
        return result
    results = list(engine.predict(str(image)))
    if time.monotonic() - started > OCR_TIMEOUT:
        raise RuntimeError("PP-OCRv6 page timeout")
    if len(results) != 1:
        raise RuntimeError("PP-OCRv6 returned an unexpected page count")
    return normalize_ocr_result(results[0], width, height)


def page_payload(page: int, width: float, height: float, blocks: list[dict], source: str) -> dict:
    text = "\n".join(block["t"] for block in blocks).strip()
    return {
        "version": OCR_PAGE_VERSION, "kind": "pdf-ocr-page", "page": page,
        "width": round(float(width), 3), "height": round(float(height), 3),
        "source": source, "text": text, "blocks": blocks,
    }


def write_gzip_json(path: Path, payload: dict) -> tuple[str, int]:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
    return shared.hash_file(path)


def write_json(path: Path, payload: dict) -> tuple[str, int]:
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return shared.hash_file(path)


def read_bucket_json(path: str) -> dict:
    from huggingface_hub import HfFileSystem
    uri = f"hf://buckets/vomebook/pdf-pages/{path}"
    with HfFileSystem(token=os.environ.get("HF_TOKEN")).open(uri, "rb") as stream:
        return json.loads(stream.read())


def read_bucket_gzip_json(path: str) -> dict:
    from huggingface_hub import HfFileSystem
    uri = f"hf://buckets/vomebook/pdf-pages/{path}"
    with HfFileSystem(token=os.environ.get("HF_TOKEN")).open(uri, "rb") as stream:
        return json.loads(gzip.decompress(stream.read()))


def build_item(item: dict, source: Path, bundle: Path) -> dict:
    source_sha, source_bytes = shared.hash_file(source)
    if item.get("source_sha256") and item["source_sha256"] != source_sha:
        raise ValueError("PDF source changed after planning")
    bundle.mkdir(parents=True, exist_ok=True)
    probe = item.get("probe") or probe_pdf(source)
    pages = int(probe["page_count"])
    root = object_root(source_sha, item["key"])
    public_item = {key: value for key, value in item.items() if not key.startswith("_")}
    if probe["classification"] == "native-text":
        # A native PDF already has a browser-readable text layer.  Do not spend
        # time producing duplicate page images or OCR JSON for it.
        return {
            **public_item, "source_sha256": source_sha, "source_bytes": source_bytes,
            "status": "skipped", "reason": "native-text-pdf", "profile": asset_profile(),
            "classification": probe["classification"], "page_count": pages, "stream": False,
        }
    previous = item.get("_previous_ocr") if isinstance(item.get("_previous_ocr"), dict) else None
    previous_manifest = None
    previous_book = None
    previous_profile = str(previous.get("profile", "")) if previous else ""
    reuse_previous = bool(
        previous and previous.get("source_sha256") == source_sha
        and ocr_profile_without_jxl(previous_profile) == ocr_profile_without_jxl(asset_profile())
    )
    reencode_jxl = reuse_previous and JXL_ENABLED and previous_profile != asset_profile()
    if reuse_previous:
        try:
            previous_manifest = read_bucket_json(previous["ocr_manifest"])
            old_pages = previous_manifest.get("pages")
            if (previous_manifest.get("source_sha256") != source_sha
                    or previous_manifest.get("page_count") != pages
                    or not isinstance(old_pages, list) or len(old_pages) != pages
                    or not previous_manifest.get("page_manifest")):
                raise ValueError("incomplete previous OCR manifest")
            for number, old in enumerate(old_pages, 1):
                if old.get("p") != number:
                    raise ValueError("invalid previous OCR page order")
                for field, suffix in (("o", ".json.gz"), ("w", ".webp"), ("i", ".png")):
                    if field == "i" and field not in old:
                        continue
                    validate_ocr_object_path(old.get(field))
                    if not old[field].endswith(f"page-{number:06d}{suffix}"):
                        raise ValueError("invalid previous OCR page path")
            previous_book = read_bucket_gzip_json(previous_manifest["book_text"]["path"])
            if (previous_book.get("kind") != "pdf-ocr-book-text"
                    or len(previous_book.get("pages", [])) != pages):
                raise ValueError("incomplete previous OCR book text")
        except Exception:
            # A missing or stale old manifest is recoverable: fall back to a full OCR build.
            reuse_previous = False
            previous_manifest = previous_book = None
    previous_pages = previous_manifest.get("pages", []) if reuse_previous else []
    page_results = []
    book_pages = []
    with tempfile.TemporaryDirectory(dir=bundle) as temp_name:
        temp = Path(temp_name)
        for page in range(1, pages + 1):
            old_page = previous_pages[page - 1] if reuse_previous else None
            if old_page:
                page_entry = dict(old_page)
                if probe["classification"] != "native-text":
                    if not JXL_ENABLED:
                        for field in ("j", "js", "jb"):
                            page_entry.pop(field, None)
                    elif reencode_jxl:
                        with tempfile.TemporaryDirectory(dir=temp) as page_temp:
                            rendered, width, height = render_page(source, page, Path(page_temp))
                            jxl_path = bundle / root / "pages" / f"page-{page:06d}.jxl"
                            jxl_sha, jxl_bytes = encode_jxl(rendered, jxl_path)
                            page_entry.update({"j": (root / "pages" / jxl_path.name).as_posix(),
                                               "js": jxl_sha, "jb": jxl_bytes})
                page_results.append(page_entry)
                continue
            native_chars = int(probe.get("page_chars", [0] * pages)[page - 1])
            source_kind = "native" if native_chars >= MIN_NATIVE_PAGE_CHARS else "ocr"
            webp_path = None
            input_png_path = None
            if probe["classification"] != "native-text":
                rendered, width, height = render_page(source, page, temp)
                if source_kind == "native":
                    native = native_page(source, page)
                    blocks = native["blocks"]
                    page_text = native["text"]
                    width, height = native["width"] or width, native["height"] or height
                    source_kind = "native"
                else:
                    blocks = ocr_page(rendered, width, height)
                    page_text = "\n".join(block["t"] for block in blocks).strip()
                # Empty recognition is a valid page result (blank/image-only pages).
                if probe["classification"] != "native-text":
                    webp_path = bundle / root / "pages" / f"page-{page:06d}.webp"
                    webp_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(temp / f"page-{page:06d}.webp", webp_path)
                if source_kind == "ocr":
                    # Keep the high-quality PNG used by PaddleOCR so a later
                    # OCR-only workflow can consume it without downloading or
                    # rendering the source PDF again.  WebP remains the
                    # Reader delivery image; PNG is an internal OCR input.
                    input_png_path = bundle / root / "ocr-input" / f"page-{page:06d}.png"
                    input_png_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(rendered, input_png_path)
            else:
                native = native_page(source, page)
                width, height, blocks = native["width"], native["height"], native["blocks"]
                page_text = native["text"]
            payload = page_payload(page, width, height, blocks, source_kind)
            payload["text"] = page_text
            ocr_path = bundle / root / "ocr" / f"page-{page:06d}.json.gz"
            ocr_sha, ocr_bytes = write_gzip_json(ocr_path, payload)
            book_pages.append({"page": page, "text": page_text})
            page_entry = {
                "p": page, "o": (root / "ocr" / ocr_path.name).as_posix(),
                "os": ocr_sha, "ob": ocr_bytes, "source": source_kind,
                "chars": len(payload["text"]),
            }
            if webp_path:
                webp_sha, webp_bytes = shared.hash_file(webp_path)
                page_entry.update({"w": (root / "pages" / webp_path.name).as_posix(),
                                   "ws": webp_sha, "wb": webp_bytes})
                if JXL_ENABLED:
                    jxl_path = bundle / root / "pages" / f"page-{page:06d}.jxl"
                    jxl_sha, jxl_bytes = encode_jxl(rendered, jxl_path)
                    page_entry.update({"j": (root / "pages" / jxl_path.name).as_posix(),
                                       "js": jxl_sha, "jb": jxl_bytes})
            if input_png_path:
                input_sha, input_bytes = shared.hash_file(input_png_path)
                page_entry.update({"i": (root / "ocr-input" / input_png_path.name).as_posix(),
                                   "is": input_sha, "ib": input_bytes})
            page_results.append(page_entry)
            for suffix in (".png", ".webp"):
                (temp / f"page-{page:06d}{suffix}").unlink(missing_ok=True)
    # Keep the searchable representation compact and derive it from page files.
    if reuse_previous and previous_book:
        book_text = previous_book
        previous_book_meta = previous_manifest["book_text"]
        book_sha, book_bytes = previous_book_meta["sha256"], previous_book_meta["bytes"]
        book_path = Path(previous_book_meta["path"])
    else:
        book_text = {"version": 1, "kind": "pdf-ocr-book-text", "profile": OCR_PROFILE,
                     "pages": book_pages}
        book_path = bundle / root / "ocr" / "book-text.json.gz"
        book_sha, book_bytes = write_gzip_json(book_path, book_text)
    page_manifest_meta = previous_manifest["page_manifest"] if reuse_previous else None
    image_pages = [
        {"page": entry["p"], "path": entry["w"], "sha256": entry["ws"], "bytes": entry["wb"]}
        for entry in page_results if entry.get("w")
    ]
    if image_pages and not reuse_previous:
        page_manifest = pdf_assets.compact_page_manifest(
            source_sha, asset_profile(), image_pages, manifest_dir=root,
        )
        page_manifest_path = bundle / root / "page-manifest.json"
        page_manifest_sha, page_manifest_bytes = write_json(page_manifest_path, page_manifest)
        page_manifest_meta = {
            "path": page_manifest_path.relative_to(bundle).as_posix(),
            "sha256": page_manifest_sha, "bytes": page_manifest_bytes,
            "version": pdf_assets.PAGE_MANIFEST_VERSION,
        }
    manifest = {
        "version": OCR_MANIFEST_VERSION, "kind": "pdf-ocr",
        "source_sha256": source_sha, "source_bytes": source_bytes,
        "source_revision": item.get("source_revision", ""), "profile": asset_profile(),
        "engine": OCR_ENGINE, "dpi": OCR_DPI, "classification": probe["classification"],
        "page_count": pages, "complete": True, "pages": page_results,
        "book_text": {"path": previous_manifest["book_text"]["path"] if reuse_previous else (root / "ocr" / book_path.name).as_posix(),
                       "sha256": book_sha, "bytes": book_bytes},
    }
    if page_manifest_meta:
        manifest["page_manifest"] = page_manifest_meta
    manifest_path = bundle / root / OCR_MANIFEST_NAME
    # The browser manifest is deliberately plain JSON; page text is compressed.
    manifest_sha, manifest_bytes = write_json(manifest_path, manifest)
    return {
        **public_item, "source_sha256": source_sha, "source_bytes": source_bytes,
        "status": "ready", "profile": asset_profile(), "ocr_manifest": (root / OCR_MANIFEST_NAME).as_posix(),
        "ocr_manifest_sha256": manifest_sha, "ocr_manifest_bytes": manifest_bytes,
        "classification": probe["classification"], "page_count": pages,
        "stream": True, **({"page_manifest": page_manifest_meta} if page_manifest_meta else {}),
    }


def empty_manifest() -> dict:
    return {"version": 1, "profile": asset_profile(), "files": {}}


def load_manifest(path: Path | None) -> dict:
    if not path or not path.is_file():
        return empty_manifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError("invalid PDF OCR manifest")
    return data


def is_current(entry: dict | None, item: dict) -> bool:
    return bool(
        isinstance(entry, dict) and entry.get("status") == "ready"
        and entry.get("profile") == asset_profile()
        and entry.get("source_revision") == item.get("source_revision")
        and entry.get("source_sha256") == item.get("source_sha256")
    )


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid PDF OCR manifest")
    for key, entry in manifest["files"].items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ValueError("invalid PDF OCR entry")
        if entry.get("status") not in {"ready", "failed", "skipped", "rendered"}:
            raise ValueError("invalid PDF OCR entry status")
        if entry.get("status") == "rendered":
            validate_ocr_object_path(entry["render_manifest"]["path"], "/render-manifest.json")
            if entry.get("page_manifest"):
                validate_ocr_object_path(entry["page_manifest"]["path"], "/page-manifest.json")
        if entry.get("status") == "ready":
            if (not valid_published_profile(entry.get("profile"))
                    or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("source_sha256", "")))):
                raise ValueError("invalid PDF OCR ready entry")
            if not isinstance(entry.get("page_count"), int) or entry["page_count"] < 1:
                raise ValueError("invalid PDF OCR page count")
            validate_ocr_object_path(entry["ocr_manifest"], "/ocr-manifest.json")
            page_manifest = entry.get("page_manifest")
            if page_manifest is not None:
                if not isinstance(page_manifest, dict):
                    raise ValueError("invalid PDF OCR page manifest metadata")
                validate_ocr_object_path(page_manifest.get("path"), "/page-manifest.json")
                if not isinstance(page_manifest.get("bytes"), int) or page_manifest["bytes"] <= 0:
                    raise ValueError("invalid PDF OCR page manifest size")
    return manifest


def source_records(search_data: Path, revisions: Path, assets_manifest: dict | None = None,
                   repo: str = "", range_manifest: dict | None = None) -> list[dict]:
    records = pdf_assets.load_records(search_data, revisions, repo, "pdf")
    if assets_manifest:
        generated = pdf_assets.load_generated_records(
            assets_manifest, repo=repo, assets_revision=str(assets_manifest.get("revision", "main")),
            min_bytes=0,
        )
        for item in generated:
            # Structure-optimized PDFs are delivery artifacts in a separate
            # Bucket. OCR the ordinary Reader-Assets PDF instead of downloading
            # or reprocessing the range artifact.
            records.append(item)
    # A repaired/generated Reader PDF is authoritative for OCR. In particular,
    # never extract text from the original GBK-encoded 林一章版 PDF when its
    # repaired asset is available. If the repair is not published yet, wait
    # for it instead of indexing the known-bad source text.
    records = [item for item in records if not (
        item.get("source_kind") == "upstream"
        and reader_assets.known_gbk_pdf(item.get("repo"), item.get("path"))
    )]
    by_key = {}
    for item in records:
        key = item["key"]
        if key not in by_key or item.get("source_kind") == "generated":
            by_key[key] = item
    records = list(by_key.values())
    records.sort(key=lambda item: (item.get("repo", ""), item.get("path", ""), item.get("source_kind", "")))
    range_files = (range_manifest or {}).get("files", {})
    for item in records:
        range_entry = range_files.get(item["key"], {})
        if isinstance(range_entry, dict) and range_entry.get("status") == "failed":
            item["range_status"] = "failed"
            item["range_reason"] = str(range_entry.get("reason") or "structure optimization failed")[:1000]
            item["force_image_render"] = True
    return records


def queue(records: list[dict], limit: int, checkpoint: int) -> list[dict]:
    if limit < 1 or checkpoint < 0:
        raise ValueError("limit must be positive and checkpoint must be non-negative")
    return records[checkpoint * limit:(checkpoint + 1) * limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_item({"key": args.key}, args.source, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
