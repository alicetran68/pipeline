"""Positioned Unicode extraction for the specifically repaired Lin Yizhang PDFs.

MuPDF understands the source GBK glyphs on runners where Poppler reports zero
characters. Keep this restricted to the inspected Reader conversion contract.
"""

from __future__ import annotations

import re

import pymupdf

try:
    from . import pdf_ocr, reader_assets
except ImportError:
    import pdf_ocr, reader_assets


def applies(item: dict) -> bool:
    return reader_assets.known_gbk_pdf(item.get("repo"), item.get("path"))


def extract(document, number: int) -> dict:
    page = document[number - 1]
    flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_INHIBIT_SPACES
    fragments = []
    for group in page.get_text("dict", flags=flags)["blocks"]:
        for line in group.get("lines", []):
            text = pdf_ocr.clean_text("".join(span["text"] for span in line["spans"]))
            if text:
                x0, y0, x1, y1 = line["bbox"]
                fragments.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    # MuPDF sometimes emits each GBK glyph as a separate line, including
    # synthetic newlines inside an ordinary horizontal sentence. Regroup only
    # fragments on the same baseline and near each other in page coordinates.
    blocks = []
    for fragment in fragments:
        if blocks:
            prior = blocks[-1]
            height = max(1, min(prior["y1"] - prior["y0"], fragment["y1"] - fragment["y0"]))
            baseline_gap = abs((prior["y0"] + prior["y1"] - fragment["y0"] - fragment["y1"]) / 2)
            if baseline_gap <= height * .35 and fragment["x0"] >= prior["x0"] and fragment["x0"] <= prior["x1"] + height * 1.5:
                prior["text"] += fragment["text"]
                prior["x1"] = max(prior["x1"], fragment["x1"])
                prior["y0"] = min(prior["y0"], fragment["y0"])
                prior["y1"] = max(prior["y1"], fragment["y1"])
                continue
        blocks.append({**fragment, "confidence": 1.0, "source": "native"})
    width, height = page.rect.width, page.rect.height
    normalized = pdf_ocr.normalize_blocks(blocks, width, height)
    return {"page": number, "source": "native", "width": width, "height": height,
            "blocks": normalized, "text": "\n".join(block["t"] for block in normalized)}


def probe(path) -> dict:
    with pymupdf.open(path) as document:
        chars = [len(re.sub(r"\s+", "", extract(document, number)["text"]))
                 for number in range(1, len(document) + 1)]
    native = sum(count >= pdf_ocr.MIN_NATIVE_PAGE_CHARS for count in chars)
    return {"page_count": len(chars), "page_chars": chars, "native_pages": native,
            "native_page_ratio": native / len(chars),
            "classification": "native-text" if native == len(chars) else "scan" if not native else "mixed"}
