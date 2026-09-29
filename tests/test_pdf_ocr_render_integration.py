"""Explicit local Poppler acceptance: no model download and no network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from scripts import pdf_ocr, pdf_ocr_stages as stages, plan_pdf_ocr


class RealPdfRenderingTests(unittest.TestCase):
    def test_time_estimator_samples_real_pages_locally_and_builds_a_compact_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "sample.pdf"
            with Image.new("RGB", (100, 150), "white") as image:
                image.save(source, "PDF", resolution=100, save_all=True, append_images=[image, image])
            record = {"key": "test\0sample.pdf", "source_revision": "test", "source_kind": "upstream"}
            with patch.object(plan_pdf_ocr, "download_source", return_value=source):
                queue = plan_pdf_ocr.plan([record], workers=1,
                    render_estimator=lambda item, path: stages.estimate_render_cost(item, path, {}))
            self.assertEqual(queue["failed"], [])
            cost = queue["shards"][0]["records"][0]["_render_cost"]
            self.assertEqual((cost["source"], cost["samples"]), ("sample", 3))
            self.assertGreater(cost["seconds_per_page"], 0)
            timed = stages.plan_render_ranges(queue, {}, 1500)
            tasks = [task for shard in timed["shards"] for task in shard["records"]]
            self.assertNotIn("probe", tasks[0])
            self.assertEqual(stages.expand_render_tasks(timed, tasks)[0]["probe"]["page_count"], 3)

    def test_small_scanned_pdf_renders_lossless_ocr_input_before_recognition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "small.pdf"
            with Image.new("RGB", (300, 400), "white") as image:
                ImageDraw.Draw(image).text((30, 50), "OCR input acceptance", fill="black")
                image.save(source, "PDF", resolution=150)
            self.assertLess(source.stat().st_size, 100 * 1024 ** 2)
            bundle = root / "bundle"
            with patch.object(pdf_ocr, "ocr_page", side_effect=AssertionError("rendering must not OCR")):
                result = stages.render_book({"key": "test\0small.pdf", "source_revision": "test"}, source, bundle)
            manifest = json.loads((bundle / result["render_manifest"]["path"]).read_text())
            stages.validate_render(result, manifest)
            self.assertEqual(len(manifest["pages"]), 1)
            page = manifest["pages"][0]
            self.assertEqual(page["source"], "ocr")
            with Image.open(bundle / page["i"]) as png:
                self.assertEqual(png.format, "PNG")
                self.assertEqual(png.size, (600, 800))
            with Image.open(bundle / page["w"]) as webp:
                self.assertEqual(webp.format, "WEBP")
            self.assertNotIn("o", page)

    def test_low_resolution_full_page_scan_preserves_ocr_pixels_but_not_reader_upscale(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "low-res.pdf"
            with Image.new("RGB", (300, 400), "white") as image:
                ImageDraw.Draw(image).text((30, 50), "Source pixels", fill="black")
                image.save(source, "PDF", resolution=100)
            self.assertEqual(pdf_ocr.scan_reader_images(source, 1, 1), {1: (300, 400)})
            def encode(reader_png, destination):
                with Image.open(reader_png) as image:
                    self.assertLessEqual(image.width, 300)
                    self.assertLessEqual(image.height, 400)
                destination.write_bytes(b"jxl")
            with patch.object(pdf_ocr, "JXL_ENABLED", True), patch.object(pdf_ocr, "encode_jxl", side_effect=encode):
                bundle = root / "bundle"
                result = stages.render_book({"key": "test\0low-res.pdf", "source_revision": "test"}, source, bundle)
            page = json.loads((bundle / result["render_manifest"]["path"]).read_text())["pages"][0]
            with Image.open(bundle / page["i"]) as png, Image.open(bundle / page["w"]) as webp:
                self.assertGreater(png.width, 300)
                self.assertGreater(png.height, 400)
                self.assertLessEqual(webp.width, 300)
                self.assertLessEqual(webp.height, 400)
            self.assertEqual((bundle / page["j"]).read_bytes(), b"jxl")

    def test_small_embedded_image_is_not_treated_as_full_page_scan(self):
        listing = ("page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio\n"
                   "----\n 1 0 image 300 400 rgb 3 8 jpeg no 1 0 100 100 3000B 1%")
        def inspect(command):
            return listing if command[0] == "pdfimages" else "Page 1 size: 612 x 792 pts\n"
        with patch.object(pdf_ocr, "_run", side_effect=inspect):
            self.assertEqual(pdf_ocr.scan_reader_images(Path("sample.pdf"), 1, 1), {})

    def test_reader_jxl_and_webp_share_the_reading_cap_without_changing_ocr_png(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "large-page.pdf"
            with Image.new("RGB", (800, 1100), "white") as image:
                image.save(source, "PDF", resolution=150)
            png, width, height = pdf_ocr.render_page(source, 1, root, reader_jxl=True)
            reader_png = png.with_name(png.stem + "-reader.png")
            with Image.open(png) as full, Image.open(reader_png) as reader, \
                    Image.open(png.with_suffix(".webp")) as webp:
                self.assertEqual(full.size, (width, height))
                self.assertGreater(full.height, 1800)
                self.assertEqual(reader.size, webp.size)
                self.assertEqual(reader.height, 1800)

    def test_reader_quality_uses_clean_scan_only_with_or_without_jxl(self):
        with Image.new("RGB", (128, 128), "white") as clean, \
                Image.new("RGB", (128, 128), (196, 166, 115)) as yellowed, \
                Image.new("RGB", (128, 128), "black") as damaged:
            self.assertEqual(pdf_ocr.reader_webp_quality(clean, True), 80)
            self.assertEqual(pdf_ocr.reader_webp_quality(clean, False), 85)
            self.assertEqual(pdf_ocr.reader_webp_quality(yellowed, True), 85)
            self.assertEqual(pdf_ocr.reader_webp_quality(damaged, True), 85)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "clean.pdf"
                clean.save(source, "PDF", resolution=100)
                for jxl in (False, True):
                    with self.subTest(jxl=jxl), patch.object(pdf_ocr, "reader_webp_quality",
                                                           wraps=pdf_ocr.reader_webp_quality) as select:
                        pdf_ocr.render_page(source, 1, root, (128, 128), reader_jxl=jxl)
                        self.assertEqual(select.call_args.args[1], True)


if __name__ == "__main__":
    unittest.main()
