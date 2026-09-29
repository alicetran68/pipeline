import gzip
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import pdf_ocr
from scripts.plan_pdf_ocr import DEFAULT_OCR_TARGET_PAGES_PER_SHARD, pdf_ocr_shards, recommended_ocr_shard_count


class PdfOcrContractTests(unittest.TestCase):
    def test_normalize_ocr_result_preserves_order_and_normalizes_boxes(self):
        blocks = pdf_ocr.normalize_ocr_result({
            "rec_texts": [" 第二 ", "第一"],
            "rec_scores": [0.91, 0.99],
            "dt_polys": [
                [[100, 200], [300, 200], [300, 250], [100, 250]],
                [[10, 20], [80, 20], [80, 60], [10, 60]],
            ],
        }, 1000, 1000)
        self.assertEqual([block["t"] for block in blocks], ["第一", "第二"])
        self.assertEqual(blocks[0]["b"], [0.01, 0.02, 0.08, 0.06])
        self.assertEqual(blocks[1]["s"], "ocr")

    def test_gzip_page_payload_is_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "page.json.gz"
            first = pdf_ocr.write_gzip_json(path, pdf_ocr.page_payload(
                1, 100, 200, [{"t": "正文", "b": [0, 0, 1, 1], "c": 1, "s": "ocr"}], "ocr"))
            bytes_one = path.read_bytes()
            second = pdf_ocr.write_gzip_json(path, pdf_ocr.page_payload(
                1, 100, 200, [{"t": "正文", "b": [0, 0, 1, 1], "c": 1, "s": "ocr"}], "ocr"))
            self.assertEqual(first, second)
            self.assertEqual(bytes_one, path.read_bytes())
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["page"], 1)

    def test_shards_are_page_weighted_and_deterministic(self):
        records = [{"key": f"book-{index}", "page_count": pages} for index, pages in enumerate((100, 90, 80, 70, 60))]
        first = pdf_ocr_shards(records, 3)
        second = pdf_ocr_shards(records, 3)
        self.assertEqual(first, second)
        self.assertEqual(sorted(item["key"] for shard in first for item in shard),
                         sorted(item["key"] for item in records))
        self.assertLessEqual(max(sum(item["page_count"] for item in shard) for shard in first)
                             - min(sum(item["page_count"] for item in shard) for shard in first), 100)

    def test_recommended_shards_amortize_startup_for_page_weighted_queue(self):
        records = [{"key": f"book-{index}", "page_count": pages}
                   for index, pages in enumerate((557, 313, 302, 295, 280, 163, 107, 101, 89, 80,
                                                  79, 70, 40, 40, 33, 27, 25, 24, 22, 17))]
        self.assertEqual(recommended_ocr_shard_count(records), 2)

    def test_recommended_shards_keep_large_book_as_one_record(self):
        records = [{"key": "large", "page_count": 557}, {"key": "small", "page_count": 10}]
        self.assertEqual(recommended_ocr_shard_count(records), 1)

    def test_default_ocr_target_reduces_model_startup_shards(self):
        self.assertEqual(DEFAULT_OCR_TARGET_PAGES_PER_SHARD, 2000)

    def test_manifest_rejects_non_object_paths(self):
        manifest = {"version": 1, "files": {"x": {
            "status": "ready", "profile": pdf_ocr.asset_profile(),
            "source_sha256": "a" * 64, "page_count": 1, "ocr_manifest": "../bad"
        }}}
        with self.assertRaises(ValueError):
            pdf_ocr.validate_manifest(manifest)

    def test_manifest_accepts_persisted_ocr_input_png(self):
        path = "objects/aa/" + "a" * 64 + "/" + "b" * 16 + "/ocr-input/page-000001.png"
        self.assertEqual(pdf_ocr.validate_ocr_object_path(path, ".png"), path)

    def test_jxl_is_part_of_the_profile_identity(self):
        self.assertIn("-jxl-", pdf_ocr.asset_profile())

    def test_language_selects_supported_model_without_changing_render_profile(self):
        root = Path(__file__).resolve().parents[1]
        script = "from scripts import pdf_ocr; print(pdf_ocr.OCR_VERSION); print(pdf_ocr.asset_profile()); print(pdf_ocr.OCR_ENGINE)"
        for language, version in (("en", "PP-OCRv6"), ("japan", "PP-OCRv6"),
                                  ("ar", "PP-OCRv5"), ("fa", "PP-OCRv5"), ("korean", "PP-OCRv5")):
            with self.subTest(language=language):
                env = {**os.environ, "PDF_OCR_LANG": language}
                output = subprocess.check_output(["python3", "-c", script], cwd=root, env=env, text=True)
                lines = output.splitlines()
                self.assertEqual(lines[0], version)
                self.assertIn(f"-lang-{language}-", lines[1])
                self.assertIn(language, lines[2])

        for backend in ("paddle_onnxruntime", "rapidocr_onnxruntime"):
            env = {**os.environ, "PDF_OCR_BACKEND": backend}
            output = subprocess.check_output(["python3", "-c", script], cwd=root, env=env, text=True)
            self.assertIn(f"-backend-{backend.replace('_', '-')}", output.splitlines()[1])

        invalid = {**os.environ, "PDF_OCR_LANG": "fa", "PDF_OCR_VERSION": "PP-OCRv6"}
        result = subprocess.run(["python3", "-c", "from scripts import pdf_ocr"], cwd=root,
                                env=invalid, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        automatic_fa = {**os.environ, "PDF_OCR_LANG": "fa", "PDF_OCR_BACKEND": "rapidocr_onnxruntime"}
        output = subprocess.check_output(["python3", "-c", script], cwd=root, env=automatic_fa, text=True)
        self.assertIn("paddle-onnxruntime", output)

    def test_rapidocr_result_uses_shared_block_contract(self):
        result = type("RapidResult", (), {
            "boxes": [[[0, 0], [50, 0], [50, 20], [0, 20]]],
            "txts": ("English",), "scores": (0.95,),
        })()
        blocks = pdf_ocr.normalize_rapid_result(result, 100, 100)
        self.assertEqual(blocks[0]["t"], "English")
        self.assertEqual(blocks[0]["b"], [0.0, 0.0, 0.5, 0.2])
        empty = type("RapidEmpty", (), {"boxes": None, "txts": None, "scores": None})()
        self.assertEqual(pdf_ocr.normalize_rapid_result(empty, 100, 100), [])

    def test_auto_language_detection_selects_per_book_backend(self):
        cases = (
            ("repo\0中国历史.pdf", "ch", "rapidocr_onnxruntime"),
            ("repo\0Iran Persian تاریخ ایران.pdf", "fa", "paddle_onnxruntime"),
            ("repo\0Arabic كتاب.pdf", "ar", "paddle_onnxruntime"),
            ("repo\0한국어.pdf", "korean", "paddle_onnxruntime"),
            ("repo\0English book.pdf", "en", "rapidocr_onnxruntime"),
        )
        for key, language, backend in cases:
            with self.subTest(key=key):
                self.assertEqual(pdf_ocr.detect_language(key), language)
                self.assertEqual(pdf_ocr.book_ocr_config({"key": key}), (language, backend))

    def test_failed_structure_assessment_marks_source_for_image_render(self):
        item = {"key": "repo\0book.pdf", "repo": "repo", "path": "book.pdf"}
        with patch.object(pdf_ocr.pdf_assets, "load_records", return_value=[item]):
            records = pdf_ocr.source_records(Path("unused"), Path("unused"), range_manifest={
                "files": {item["key"]: {"status": "failed", "reason": "slow PDF"}}})
        self.assertEqual(records[0]["range_status"], "failed")
        self.assertTrue(records[0]["force_image_render"])

    def test_gbk_repaired_reader_pdf_replaces_original_in_ocr_sources(self):
        repo, path = next(iter(pdf_ocr.reader_assets.KNOWN_GBK_PDFS))
        original = {"key": repo + "\0" + path, "repo": repo, "path": path,
                    "source_kind": "upstream", "source_revision": "source"}
        repaired = {**original, "source_kind": "generated", "reader_assets_path": "objects/repaired/document.pdf",
                    "reader_assets_repo": "vomebook/Reader-Assets", "reader_assets_revision": "assets"}
        with patch.object(pdf_ocr.pdf_assets, "load_records", return_value=[original]), \
                patch.object(pdf_ocr.pdf_assets, "load_generated_records", return_value=[repaired]):
            records = pdf_ocr.source_records(Path("unused"), Path("unused"), {"revision": "assets"})
        self.assertEqual(records, [repaired])

        with patch.object(pdf_ocr.pdf_assets, "load_records", return_value=[original]), \
                patch.object(pdf_ocr.pdf_assets, "load_generated_records", return_value=[]):
            self.assertEqual(pdf_ocr.source_records(Path("unused"), Path("unused"), {"revision": "assets"}), [])

    def test_manifest_accepts_published_jxl_layout_from_separate_ocr_worker(self):
        current = pdf_ocr.asset_profile()
        other = current.replace(f"-jxl-{int(pdf_ocr.JXL_ENABLED)}-",
                                f"-jxl-{int(not pdf_ocr.JXL_ENABLED)}-")
        entry = {"status": "ready", "profile": other + "-layout-v1-" + "a" * 16,
                 "source_sha256": "b" * 64, "page_count": 1,
                 "ocr_manifest": "objects/bb/" + "b" * 64 + "/" + "c" * 16 + "/ocr-manifest.json"}
        manifest = {"version": 1, "files": {"book": entry}}
        self.assertIs(pdf_ocr.validate_manifest(manifest), manifest)
        with self.assertRaises(ValueError):
            pdf_ocr.validate_manifest({"version": 1, "files": {"book": {**entry, "profile": other + "-layout-v1-bad"}}})
        with self.assertRaises(ValueError):
            pdf_ocr.validate_manifest({"version": 1, "files": {"book": {**entry, "profile": other + "-layout-"}}})

    def test_manifest_accepts_other_language_and_backend_profiles(self):
        profiles = (
            "pdf-ocr-v1-pp-ocrv5-medium-lang-fa-backend-paddle-onnxruntime-dpi-300-webp-85-1800-native-48-0.9-maxpix-50000000-jxl-0-1.5-7",
            "pdf-ocr-v1-pp-ocrv6-medium-lang-en-backend-rapidocr-onnxruntime-dpi-300-webp-85-1800-native-48-0.9-maxpix-50000000-jxl-0-1.5-7",
        )
        for profile in profiles:
            entry = {"status": "ready", "profile": profile, "source_sha256": "c" * 64,
                     "page_count": 1, "ocr_manifest": "objects/cc/" + "c" * 64 + "/" + "d" * 16 + "/ocr-manifest.json"}
            self.assertTrue(pdf_ocr.valid_published_profile(profile))
            self.assertIsNotNone(pdf_ocr.validate_manifest({"version": 1, "files": {"book": entry}}))


if __name__ == "__main__":
    unittest.main()
