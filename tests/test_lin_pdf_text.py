import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf
from PIL import Image

from scripts import lin_pdf_text, pdf_ocr, pdf_ocr_stages as stages, plan_pdf_ocr
from scripts import publish_lin_native_text, reader_assets, repair_gbk_pdf
from tests.test_repair_gbk_pdf import fixture


class LinPdfTextTests(unittest.TestCase):
    def test_planner_marks_only_supported_native_text(self):
        item = {"key": reader_assets.asset_key("VoiceOfML/Teachers", reader_assets.GBK_PDF_FOLDER + "/合订本.pdf"),
                "repo": "VoiceOfML/Teachers", "path": reader_assets.GBK_PDF_FOLDER + "/合订本.pdf",
                "source_kind": "generated", "source_revision": "revision"}
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.pdf"
            source.write_bytes(b"pdf")
            probe = {"page_count": 10, "page_chars": [80] * 10, "classification": "native-text",
                     "native_page_ratio": .9}
            with patch.object(plan_pdf_ocr, "download_source", return_value=source), \
                    patch.object(lin_pdf_text, "probe", return_value=probe):
                record = plan_pdf_ocr.plan([item])["shards"][0]["records"][0]
            self.assertEqual(record["native_extractor"], "pymupdf-v1")
            with patch.object(plan_pdf_ocr, "download_source", return_value=source), \
                    patch.object(lin_pdf_text, "probe", return_value={**probe, "native_page_ratio": .2}):
                record = plan_pdf_ocr.plan([item])["shards"][0]["records"][0]
            self.assertNotIn("native_extractor", record)

    def test_separate_gbk_glyphs_form_searchable_lines(self):
        fragments = [((10, 10, 24, 26), "吊"), ((22, 10, 36, 26), "罗"),
                     ((34, 10, 48, 26), "荣"), ((10, 35, 24, 51), "同"),
                     ((22, 35, 36, 51), "志")]
        page = Mock(rect=Mock(width=100, height=100))
        page.get_text.return_value = {"blocks": [{"lines": [
            {"bbox": box, "spans": [{"text": glyph}]} for box, glyph in fragments]}]}
        text = lin_pdf_text.extract([page], 1)
        self.assertEqual(text["text"], "吊罗荣\n同志")
        self.assertEqual([block["t"] for block in text["blocks"]], ["吊罗荣", "同志"])

    def test_native_extraction_and_existing_page_stream_backfill(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, repaired = root / "original.pdf", root / "repaired.pdf"
            source.write_bytes(fixture())
            repair_gbk_pdf.repair_pdf(source, repaired)
            path = reader_assets.GBK_PDF_FOLDER + "/第1册 (1949.9-1950.12).pdf"
            item = {"key": reader_assets.asset_key("VoiceOfML/Teachers", path),
                    "repo": "VoiceOfML/Teachers", "path": path,
                    "source_kind": "generated", "source_revision": "source-revision",
                    "reader_assets_repo": "vomebook/Reader-Assets",
                    "reader_assets_path": "objects/a/gbk-font-repair-v1/document.pdf",
                    "reader_assets_revision": "assets-revision", "force_image_render": True}

            with pymupdf.open(repaired) as document:
                text = lin_pdf_text.extract(document, 1)
            self.assertIn("建国以来毛泽东文稿", text["text"])
            self.assertIn("ABC 123", text["text"])
            self.assertTrue(text["blocks"])
            self.assertTrue(all(0 <= value <= 1 for block in text["blocks"] for value in block["b"]))
            with patch.object(pdf_ocr, "MIN_NATIVE_PAGE_CHARS", 5):
                self.assertEqual(lin_pdf_text.probe(repaired)["classification"], "native-text")

            def render(_source, number, directory, reader_pixels=None, reader_jxl=False):
                image = directory / f"page-{number:06d}.png"
                with Image.new("RGB", (200, 300), "white") as bitmap:
                    bitmap.save(image)
                    bitmap.save(image.with_suffix(".webp"))
                return image, 200, 300

            with patch.object(pdf_ocr, "render_page", side_effect=render), \
                    patch.object(pdf_ocr, "scan_reader_images", return_value={}), \
                    patch.object(pdf_ocr, "MIN_NATIVE_PAGE_CHARS", 5), \
                    patch.object(pdf_ocr, "JXL_ENABLED", False):
                rendered = stages.render_book(item, repaired, root / "render")
            manifest_path = root / "render" / rendered["render_manifest"]["path"]
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(rendered["ocr_pages"], 0)
            self.assertEqual([page["source"] for page in manifest["pages"]], ["native", "native"])
            page_path = root / "render" / manifest["pages"][0]["o"]
            page = json.loads(gzip.decompress(page_path.read_bytes()))
            self.assertEqual(page["text_spans"][0]["end"], len(page["blocks"][0]["t"]))

            def read_render_object(meta, suffix=None):
                pdf_ocr.validate_ocr_object_path(meta["path"], suffix)
                return (root / "render" / meta["path"]).read_bytes()

            with patch.object(stages, "read_object", side_effect=read_render_object):
                queue = stages.plan_images({rendered["key"]: {**rendered, "native_extractor": "pymupdf-v1"}}, {}, {})
                self.assertEqual(queue["total_ocr_pages"], 0)
                generated = stages.assemble_book(queue["books"][0], {}, root / "indexed")
            generated_manifest = json.loads((root / "indexed" / generated["ocr_manifest"]).read_text())
            generated_index = json.loads(gzip.decompress(
                (root / "indexed" / generated_manifest["book_text"]["path"]).read_bytes()))
            self.assertEqual([entry["page"] for entry in generated_index["pages"]], [1, 2])
            self.assertTrue(all("建国以来毛泽东文稿" in entry["text"] for entry in generated_index["pages"]))

            old = {**rendered, "ocr_pages": 2, "classification": "scan"}
            self.assertTrue(stages.skip_ocr_for_generated_text_pdf(old))
            self.assertFalse(stages.skip_ocr_for_generated_text_pdf(
                {**rendered, "native_extractor": "pymupdf-v1"}))
            self.assertFalse(stages.skip_ocr_for_generated_text_pdf(
                {**old, "classification": "mixed", "native_extractor": "pymupdf-v1"}))
            old_manifest = {**manifest, "classification": "scan",
                            "pages": [{k: v for k, v in page.items() if k not in
                                      {"o", "os", "ob", "text", "text_spans", "layout", "chars"}}
                                      | {"source": "ocr"} for page in manifest["pages"]]}
            stages.validate_render(old, old_manifest)
            with patch.object(stages, "read_object", side_effect=AssertionError("no OCR planner download")):
                self.assertEqual(stages.plan_images({old["key"]: old}, {}, {})["books"], [])
            recovered = publish_lin_native_text.build(old, old_manifest, repaired, root / "backfill")
            self.assertEqual(recovered["status"], "ready")
            self.assertEqual(recovered["page_manifest"], old["page_manifest"])
            ocr_manifest = json.loads((root / "backfill" / recovered["ocr_manifest"]).read_text())
            self.assertEqual([page["source"] for page in ocr_manifest["pages"]], ["native", "native"])
            text_index = json.loads(gzip.decompress((root / "backfill" / ocr_manifest["book_text"]["path"]).read_bytes()))
            self.assertEqual(text_index["version"], 2)
            self.assertEqual([page["page"] for page in text_index["pages"]], [1, 2])
            self.assertTrue(all("建国以来毛泽东文稿" in page["text"] for page in text_index["pages"]))
            self.assertEqual([page["text_spans"][0]["start"] for page in text_index["pages"]], [0, 0])


if __name__ == "__main__":
    unittest.main()
