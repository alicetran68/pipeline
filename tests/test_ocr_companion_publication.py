import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import pdf_ocr, pdf_ocr_stages as stages, pdf_reading_v3 as v3
from scripts import publish_pdf_ocr_assets as publication
from tests.test_pdf_text_layer import raw_page


class OcrCompanionPublicationTests(unittest.TestCase):
    def test_mixed_raw_pages_publish_one_complete_verified_companion(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            source = "a" * 64
            root = Path("objects") / "aa" / source / ("b" * 16)
            objects, pages = {}, []
            for number, kind, content in ((1, "native", "Native"), (2, "ocr", "Recognized")):
                raw = raw_page(content)
                raw.update(page=number, source=kind)
                path = root / "ocr" / f"page-{number:06d}.json.gz"
                digest, size = pdf_ocr.write_gzip_json(bundle / path, raw)
                objects[path.as_posix()] = (bundle / path).read_bytes()
                pages.append({"p": number, "source": kind, "o": path.as_posix(), "os": digest, "ob": size})
            book = {"key": "repo\0book.pdf", "source_sha256": source, "source_bytes": 100,
                    "status": "ready", "page_count": 2, "classification": "mixed", "pages": pages,
                    "ocr_language": "ch", "ocr_backend": "rapidocr_onnxruntime",
                    "profile": pdf_ocr.asset_profile("ch", "rapidocr_onnxruntime"),
                    "render_manifest": {"sha256": "c" * 64}}
            def read(meta, suffix=None):
                pdf_ocr.validate_ocr_object_path(meta["path"], suffix)
                return objects[meta["path"]]
            with patch.object(stages, "read_object", side_effect=read):
                result = stages.assemble_book(book, {"2": pages[1]}, bundle)
            for path in bundle.rglob("*"):
                if path.is_file():
                    objects[path.relative_to(bundle).as_posix()] = path.read_bytes()
            index = v3.verify_text_bundle(result["text_layer"], read, source, 2)
            self.assertEqual(index["revision"], "raw")
            complete = v3.decode(read(index["book_text"]))
            self.assertEqual([p["text"] for p in complete["pages"]], ["Native", "Recognized"])
            manifest = json.loads(objects[result["ocr_manifest"]])
            compatibility = json.loads(gzip.decompress(read(manifest["book_text"])))
            self.assertEqual(manifest["text_layer"], result["text_layer"])
            self.assertEqual(compatibility["text_layer"], result["text_layer"])
            with patch.object(publication, "read_bucket_bytes", side_effect=lambda path, token, bucket: objects[path]):
                publication.verify_ready_objects(result)
                objects[result["text_layer"]["path"]] = b"corrupt companion"
                with self.assertRaises(ValueError):
                    publication.verify_ready_objects(result)
