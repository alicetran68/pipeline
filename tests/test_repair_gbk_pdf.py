import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, NameObject,
    NumberObject, TextStringObject,
)

from scripts import convert_reader_assets, reader_assets, repair_gbk_pdf


def fixture() -> bytes:
    writer = PdfWriter()
    descriptor = DictionaryObject({
        NameObject("/Type"): NameObject("/FontDescriptor"),
        NameObject("/FontName"): NameObject("/宋体"),
        NameObject("/Flags"): NumberObject(4),
        NameObject("/FontBBox"): ArrayObject([NumberObject(n) for n in [0, -200, 1000, 1000]]),
        NameObject("/Ascent"): NumberObject(1000), NameObject("/Descent"): NumberObject(-200),
        NameObject("/CapHeight"): NumberObject(1000), NameObject("/StemV"): NumberObject(100),
        NameObject("/ItalicAngle"): NumberObject(0),
    })
    unicode_map = DecodedStreamObject()
    unicode_map.set_data(b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
                        b"/CMapName /BadMap def /CMapType 2 def\n"
                        b"1 begincodespacerange <00> <ff> endcodespacerange\n"
                        b"2 beginbfrange <20> <7e> <0020> <a0> <ff> <00a0> endbfrange\n"
                        b"endcmap CMapName currentdict /CMap defineresource pop end end")
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/TrueType"),
        NameObject("/BaseFont"): NameObject("/宋体"),
        NameObject("/FontDescriptor"): writer._add_object(descriptor),
        NameObject("/FirstChar"): NumberObject(0), NameObject("/LastChar"): NumberObject(255),
        NameObject("/Widths"): ArrayObject([NumberObject(500)] * 256),
        NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        NameObject("/ToUnicode"): writer._add_object(unicode_map),
    })
    ref = writer._add_object(font)
    for index in range(2):
        page = writer.add_blank_page(width=400, height=600)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): ref})})
        content = DecodedStreamObject()
        text = "建国以来毛泽东文稿 ABC 123".encode("gbk").hex().encode("ascii")
        content.set_data(b"BT /F1 14 Tf 30 500 Td <" + text + b"> Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
    writer.add_metadata({"/Title": "测试原始标题"})
    writer.add_outline_item("第二页", 1)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class GbkPdfRepairTests(unittest.TestCase):
    def test_only_known_sources_enter_repair_queue(self):
        for repo, path in reader_assets.KNOWN_GBK_PDFS:
            self.assertEqual(reader_assets.source_conversion_contract(repo, path, "pdf"),
                             reader_assets.GBK_PDF_CONTRACT)
        known_path = next(iter(reader_assets.KNOWN_GBK_PDFS))[1]
        self.assertIsNone(reader_assets.source_conversion_contract("other", known_path, "pdf"))
        self.assertIsNone(reader_assets.source_conversion_contract(
            "VoiceOfML/Teachers", reader_assets.GBK_PDF_FOLDER + "/new.pdf", "pdf"))
        for folder in reader_assets.GBK_PDF_REPAIR_FOLDERS:
            self.assertEqual(reader_assets.source_conversion_contract(
                "VoiceOfML/Teachers", folder + "/volume.pdf", "pdf"), reader_assets.GBK_PDF_CONTRACT)
            self.assertFalse(reader_assets.known_gbk_pdf("Other/Teachers", folder + "/volume.pdf"))
        self.assertIsNone(reader_assets.source_conversion_contract(
            "VoiceOfML/Teachers", "A1 马克思&恩格斯/01-03 马克思恩格斯全集 林一章旧版（博览网版）/1上.pdf", "pdf"))

    def test_repair_keeps_pages_content_geometry_metadata_and_outline(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "original.pdf", Path(root) / "repaired.pdf"
            original = fixture()
            source.write_bytes(original)
            result = repair_gbk_pdf.repair_pdf(source, target)
            self.assertEqual(result, {"page_count": 2, "fonts_repaired": 1})
            self.assertEqual(source.read_bytes(), original)
            before, after = PdfReader(source), PdfReader(target)
            self.assertEqual(len(before.pages), len(after.pages))
            for a, b in zip(before.pages, after.pages):
                self.assertEqual(a.get_contents().get_data(), b.get_contents().get_data())
                self.assertEqual(a.mediabox, b.mediabox)
            self.assertEqual(after.metadata.title, "测试原始标题")
            self.assertEqual(after.outline[0].title, "第二页")
            self.assertEqual(after.get_destination_page_number(after.outline[0]), 1)
            self.assertEqual(after.pages[0]["/Resources"]["/Font"]["/F1"]["/Encoding"], "/GBK-EUC-H")
            self.assertEqual(after.pages[0]["/Resources"]["/Font"].raw_get("/F1"),
                             after.pages[1]["/Resources"]["/Font"].raw_get("/F1"))

    @unittest.skipUnless(shutil.which("pdftotext"), "Poppler is required for independent text decoding")
    def test_converter_produces_readable_chinese_and_ascii(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "source.pdf", Path(root) / "document.pdf"
            source.write_bytes(fixture())
            convert_reader_assets.convert_file({"extension": "pdf", "profile": reader_assets.GBK_PDF_CONTRACT[0],
                                                "repo": "VoiceOfML/Teachers", "path": "test.pdf"}, source, target, Path(root))
            text = subprocess.check_output(["pdftotext", str(target), "-"], text=True)
            self.assertEqual("".join(text.split()), "建国以来毛泽东文稿ABC123" * 2)

    def test_embedded_valid_multibyte_and_other_fonts_are_not_repaired(self):
        for variant in ["embedded", "multibyte", "latin", "widths", "different-name"]:
            with self.subTest(variant=variant):
                font = PdfReader(io.BytesIO(fixture())).pages[0]["/Resources"]["/Font"]["/F1"]
                if variant == "embedded":
                    font["/FontDescriptor"][NameObject("/FontFile2")] = DecodedStreamObject()
                elif variant == "multibyte":
                    font["/ToUnicode"].set_data(b"1 begincodespacerange <0000> <ffff> endcodespacerange")
                elif variant == "latin":
                    font[NameObject("/BaseFont")] = NameObject("/Arial")
                elif variant == "widths":
                    font["/Widths"][0] = NumberObject(600)
                else:
                    font["/FontDescriptor"][NameObject("/FontName")] = NameObject("/黑体")
                self.assertIsNone(repair_gbk_pdf.repairable_font(font))

    def test_raw_gbk_font_names_are_recognized(self):
        self.assertEqual(repair_gbk_pdf.font_name("/ËÎÌå"), "宋体")
        self.assertEqual(repair_gbk_pdf.font_name("/ºÚÌå"), "黑体")

    def test_gb2312_base_fonts_accept_legacy_descriptor_aliases(self):
        for base, descriptor, expected in [("仿宋_GB2312", "·ÂËÎÌå", "FangSong"),
                                            ("楷体_GB2312", "¿¬Ìå", "KaiTi")]:
            font = PdfReader(io.BytesIO(fixture())).pages[0]["/Resources"]["/Font"]["/F1"]
            font[NameObject("/BaseFont")] = NameObject("/" + base)
            font["/FontDescriptor"][NameObject("/FontName")] = TextStringObject(descriptor)
            self.assertEqual(repair_gbk_pdf.repairable_font(font), expected)

    def test_combined_edition_placeholder_names_without_unicode(self):
        for name, expected in repair_gbk_pdf.FONT_NAMES.items():
            font = PdfReader(io.BytesIO(fixture())).pages[0]["/Resources"]["/Font"]["/F1"]
            font[NameObject("/BaseFont")] = NameObject("/" + name)
            font["/FontDescriptor"][NameObject("/FontName")] = TextStringObject("????")
            del font["/ToUnicode"]
            self.assertEqual(repair_gbk_pdf.repairable_font(font), expected)

    def test_missing_unicode_map_with_matching_legacy_descriptor(self):
        for base, descriptor in [("宋体", "ËÎÌå"), ("仿宋_GB2312", "·ÂËÎÌå"),
                                 ("黑体", "Ð¡±êËÎ")]:
            font = PdfReader(io.BytesIO(fixture())).pages[0]["/Resources"]["/Font"]["/F1"]
            font[NameObject("/BaseFont")] = NameObject("/" + base)
            font["/FontDescriptor"][NameObject("/FontName")] = TextStringObject(descriptor)
            del font["/ToUnicode"]
            self.assertIsNotNone(repair_gbk_pdf.repairable_font(font))

    def test_refuses_overwrite_and_unmatched_source(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "source.pdf", Path(root) / "target.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(source)
            with self.assertRaisesRegex(ValueError, "separate output"):
                repair_gbk_pdf.repair_pdf(source, source)
            with self.assertRaisesRegex(ValueError, "expected malformed"):
                repair_gbk_pdf.repair_pdf(source, target)
            self.assertFalse(target.exists())

    def test_empty_password_encrypted_fixture_is_repaired(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "source.pdf", Path(root) / "target.pdf"
            source.write_bytes(fixture())
            writer = PdfWriter(clone_from=PdfReader(source))
            writer.encrypt("")
            with source.open("wb") as handle:
                writer.write(handle)
            result = repair_gbk_pdf.repair_pdf(source, target)
            self.assertEqual(result["fonts_repaired"], 1)
            self.assertFalse(PdfReader(target).is_encrypted)


if __name__ == "__main__":
    unittest.main()
