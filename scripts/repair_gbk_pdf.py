#!/usr/bin/env python3
"""Repair the known non-embedded GBK fonts in the Lin Yizhang PDF edition.

These files declare simple WinAnsi TrueType fonts while their content streams
contain two-byte GBK text. Their one-byte ToUnicode map is also incompatible.
Create proper Type0/Adobe-GB1 fonts and a GBK-to-Unicode map; retain content,
images, page geometry, document metadata and navigation through document cloning.
The planner opts in only the explicitly listed source files. This is not a
general encoding guesser for arbitrary PDFs.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, NameObject,
    NumberObject, TextStringObject,
)

FONT_NAMES = {"宋体": "STSong-Light", "黑体": "SimHei",
              "仿宋_GB2312": "FangSong", "仿宋": "FangSong", "仿宋体": "FangSong",
              "楷体_GB2312": "KaiTi", "楷体": "KaiTi"}


def font_name(value) -> str:
    name = str(value or "").lstrip("/")
    if name in FONT_NAMES:
        return name
    try:
        return name.encode("latin1").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def repairable_font(font) -> str | None:
    if not isinstance(font, DictionaryObject) or font.get("/Type") != "/Font":
        return None
    if font.get("/Subtype") != "/TrueType" or font.get("/Encoding") != "/WinAnsiEncoding":
        return None
    name = font_name(font.get("/BaseFont"))
    descriptor = font.get("/FontDescriptor")
    if descriptor is None or name not in FONT_NAMES:
        return None
    descriptor = descriptor.get_object()
    descriptor_name = font_name(descriptor.get("/FontName"))
    # The combined edition lost descriptor names and ToUnicode entirely.
    combined_placeholder = (font.get("/ToUnicode") is None
                            and re.fullmatch(r"\?{4,6}", descriptor_name) is not None)
    # Some other Lin Yizhang volumes have no ToUnicode at all; their heading
    # font also advertises a legacy small-Song descriptor under a Hei base font.
    small_song_heading = (name == "黑体" and descriptor_name == "小标宋"
                          and font.get("/ToUnicode") is None)
    if FONT_NAMES.get(descriptor_name) != FONT_NAMES[name] and not combined_placeholder and not small_song_heading:
        return None
    if any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
        return None
    widths = font.get("/Widths")
    if (font.get("/FirstChar") != 0 or font.get("/LastChar") != 255
            or widths is None or len(widths.get_object()) != 256
            or any(width != 500 for width in widths.get_object())):
        return None
    unicode_map = font.get("/ToUnicode")
    if combined_placeholder or (unicode_map is None and
                                (FONT_NAMES.get(descriptor_name) == FONT_NAMES[name] or small_song_heading)):
        return FONT_NAMES[name]
    if unicode_map is None or not hasattr(unicode_map.get_object(), "get_data"):
        return None
    data = unicode_map.get_object().get_data()
    spaces = re.findall(rb"begincodespacerange\s*(.*?)\s*endcodespacerange", data, re.S)
    if len(spaces) != 1 or re.sub(rb"\s+", b"", spaces[0]).lower() != b"<00><ff>":
        return None
    if not re.search(rb"<a0>\s*<ff>\s*<00a0>", data, re.I):
        return None
    return FONT_NAMES[name]


@lru_cache(maxsize=1)
def gbk_unicode_cmap() -> bytes:
    entries = []
    for code in range(65536):
        raw = bytes([code]) if code < 128 else code.to_bytes(2, "big")
        if code >= 128 and not (0x81 <= raw[0] <= 0xfe
                                and 0x40 <= raw[1] <= 0xfe and raw[1] != 0x7f):
            continue
        try:
            text = raw.decode("gbk", errors="strict")
        except UnicodeDecodeError:
            continue
        entries.append(f"<{raw.hex()}> <{text.encode('utf-16-be').hex()}>")
    lines = [
        "/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Reader-GBK-UCS def", "/CMapType 2 def",
        "2 begincodespacerange", "<00> <7f>", "<8140> <fefe>", "endcodespacerange",
    ]
    for start in range(0, len(entries), 100):
        group = entries[start:start + 100]
        lines.extend([f"{len(group)} beginbfchar", *group, "endbfchar"])
    lines.extend(["endcmap", "CMapName currentdict /CMap defineresource pop", "end end"])
    return ("\n".join(lines) + "\n").encode("ascii")


def repair_pdf(source: Path, target: Path) -> dict:
    if source.resolve() == target.resolve():
        raise ValueError("GBK repair requires a separate output file")
    reader = PdfReader(source)
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError("GBK repair requires a password for encrypted PDFs")
    writer = PdfWriter(clone_from=reader)
    # Clone all document objects, including fonts in Form XObjects and inherited
    # resources, rather than visiting only fonts directly attached to pages.
    candidates = [(obj, repairable_font(obj)) for obj in list(writer._objects)]
    candidates = [(font, name) for font, name in candidates if name is not None]
    if not candidates:
        raise ValueError("PDF does not contain the expected malformed GBK fonts")
    cmap = DecodedStreamObject()
    cmap.set_data(gbk_unicode_cmap())
    cmap_ref = writer._add_object(cmap.flate_encode())
    for font, normalized in candidates:
        name = NameObject("/" + normalized)
        descriptor = DictionaryObject(dict(font["/FontDescriptor"]))
        descriptor[NameObject("/FontName")] = name
        descriptor[NameObject("/Flags")] = NumberObject(4 if normalized == "SimHei" else 6)
        descendant = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/CIDFontType0"),
            NameObject("/BaseFont"): name,
            NameObject("/CIDSystemInfo"): DictionaryObject({
                NameObject("/Registry"): TextStringObject("Adobe"),
                NameObject("/Ordering"): TextStringObject("GB1"),
                NameObject("/Supplement"): NumberObject(0),
            }),
            NameObject("/FontDescriptor"): writer._add_object(descriptor),
            NameObject("/DW"): NumberObject(1000),
            # GBK-EUC-H maps printable ASCII to Adobe-GB1 CIDs 1..95.
            NameObject("/W"): ArrayObject([NumberObject(1), NumberObject(95), NumberObject(500)]),
        })
        font.clear()
        font.update({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type0"),
            NameObject("/BaseFont"): name,
            NameObject("/Encoding"): NameObject("/GBK-EUC-H"),
            NameObject("/DescendantFonts"): ArrayObject([writer._add_object(descendant)]),
            NameObject("/ToUnicode"): cmap_ref,
        })
    writer.write(target)
    return {"page_count": len(writer.pages), "fonts_repaired": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair_pdf(args.source, args.target), sort_keys=True))


if __name__ == "__main__":
    main()
