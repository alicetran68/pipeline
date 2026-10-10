import copy
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf
from PIL import Image

from scripts import pdf_reading_v3 as v3, pdf_text_layer as text, shared
from tests.test_pdf_text_layer import raw_page


class ReadingV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        self.objects = {}

    def add(self, path, raw, bucket=shared.PDF_PAGES_BUCKET, role="runtime"):
        self.objects[(bucket, path)] = raw
        return {"bucket": bucket, "path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "role": role}

    def read(self, ref):
        return self.objects[(ref["bucket"], ref["path"])]

    def generated(self):
        for path in self.bundle.rglob("*"):
            if path.is_file():
                self.objects[(shared.PDF_PAGES_BUCKET, path.relative_to(self.bundle).as_posix())] = path.read_bytes()

    def layers(self, count):
        layers, pages = [], []
        for number in range(1, count + 1):
            payload = {**raw_page("Text"), "page": number}
            layers.append(text.from_page(payload, "a" * 64))
            pages.append({"p": number})
        return layers, pages

    def test_text_partitions_deep_page_and_full_verification(self):
        layers, pages = self.layers(260)
        ref = v3.build_text_bundle(layers, pages, Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), self.bundle)
        self.generated()
        manifest = v3.verify_text_bundle(ref, self.read, "a" * 64, 260)
        target = v3.partition_for(manifest, 259)
        self.assertEqual((target["start"], target["end"]), (257, 260))
        self.assertEqual(len(manifest["partitions"]), 3)
        with self.assertRaises(ValueError):
            v3.partition_for(manifest, 261)

    def test_corrupt_page_and_wrong_source_block_verification(self):
        layers, pages = self.layers(1)
        ref = v3.build_text_bundle(layers, pages, Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), self.bundle)
        self.generated()
        with self.assertRaises(ValueError):
            v3.verify_text_bundle(ref, self.read, "b" * 64, 1)
        path = next(key for key in self.objects if key[1].endswith("text/page-000001.json.gz"))
        self.objects[path] = b"corrupted"
        with self.assertRaises(ValueError):
            v3.verify_text_bundle(ref, self.read, "a" * 64, 1)

    def test_search_partitions_are_text_only_complete_and_checked_against_pages(self):
        layers, pages = self.layers(70)
        ref = v3.build_text_bundle(layers, pages, Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), self.bundle)
        self.generated()
        manifest = v3.verify_text_bundle(ref, self.read, "a" * 64, 70)
        self.assertEqual([(p["start"], p["end"]) for p in manifest["search_partitions"]],
                         [(1, 32), (33, 64), (65, 70)])
        part = manifest["search_partitions"][-1]
        data = v3.decode(self.read(part["resource"]))
        self.assertEqual(set(data["pages"][0]), {"page", "text", "text_generation"})
        data["pages"][-1]["text"] = "wrong effective text"
        bad = self.add(part["resource"]["path"], gzip.compress(json.dumps(data).encode()))
        manifest["search_partitions"][-1]["resource"] = bad
        bad_index = self.add(ref["path"], json.dumps(manifest).encode())
        with self.assertRaisesRegex(ValueError, "search partition differs"):
            v3.verify_text_bundle(bad_index, self.read, "a" * 64, 70)

    def test_search_partition_byte_budget_preserves_oversized_whole_page(self):
        layers, pages = self.layers(3)
        for n in range(3):
            layers[n] = text.from_page({**raw_page("x" * 300000), "page": n + 1}, "a" * 64)
        ref = v3.build_text_bundle(layers, pages, Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), self.bundle)
        self.generated()
        manifest = v3.verify_text_bundle(ref, self.read, "a" * 64, 3)
        self.assertEqual(len(manifest["search_partitions"]), 3)

    def test_review_evidence_keeps_qualified_input_bucket(self):
        layers, pages = self.layers(1)
        layers[0] = text.from_page(raw_page("Text", .1), "a" * 64)
        evidence = self.add("objects/aa/source/inputs/ocr-input/page-000001.png", b"evidence",
                            shared.PDF_OCR_INPUT_BUCKET, "review")
        pages[0]["i"] = evidence["path"]
        pages[0]["is"] = evidence["sha256"]
        pages[0]["ib"] = evidence["bytes"]
        ref = v3.build_text_bundle(layers, pages, Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), self.bundle)
        self.generated()
        manifest = v3.verify_text_bundle(ref, self.read, "a" * 64, 1, verify_evidence=True)
        review = v3.decode(self.read(manifest["review"]))
        self.assertEqual(review["tasks"][0]["evidence"][0]["bucket"], shared.PDF_OCR_INPUT_BUCKET)

    def spec(self):
        with pymupdf.open() as doc:
            doc.new_page(width=100, height=200)
            doc.new_page(width=100, height=200).set_rotation(90)
            pdf = doc.tobytes()
        primary = self.add("objects/aa/book/primary/document.pdf", pdf)
        mapping = v3.page_map(pdf, "a" * 64)
        previews = []
        for geometry in mapping["pages"]:
            image = Image.new("RGB", (int(geometry["width"]), int(geometry["height"])), "white")
            stream = io.BytesIO()
            image.save(stream, "PNG")
            ref = self.add(f"objects/aa/book/preview/page-{geometry['page']:06d}.png", stream.getvalue())
            previews.append({"page": geometry["page"], "page_identity": geometry["identity"],
                             "codec": "png", "width": image.width, "height": image.height, "resource": ref})
        return {"source_sha256": "a" * 64, "source_key": "repo\0book.pdf", "primary": primary,
                "previews": previews, "require_complete_preview": True}

    def test_full_reading_bundle_preserves_real_rotation(self):
        spec = self.spec()
        ref, manifest = v3.build_reading(spec, self.read, self.bundle)
        self.generated()
        mapping = json.loads(gzip.decompress(self.read(manifest["page_map"])))
        self.assertEqual(mapping["pages"][1]["rotation"], 90)
        self.assertTrue(manifest["preview"]["complete"])
        self.assertEqual(manifest["components"]["searchable_pdf"], "skipped")
        self.assertEqual(v3.decode(self.read(ref)), manifest)
        self.assertEqual(v3.verify_reading(ref, self.read), manifest)
        self.assertEqual(v3.build_reading(spec, self.read, self.bundle)[0], ref)

    def test_missing_preview_is_pending_and_never_claimed_complete(self):
        spec = self.spec()
        spec["previews"].pop()
        with self.assertRaises(ValueError):
            v3.build_reading(spec, self.read, self.bundle)
        spec["require_complete_preview"] = False
        _, manifest = v3.build_reading(spec, self.read, self.bundle)
        self.assertFalse(manifest["preview"]["complete"])
        self.assertEqual(manifest["components"]["document"], "ready")

    def test_duplicate_wrong_identity_codec_and_processing_preview_rejected(self):
        spec = self.spec()
        duplicate = copy.deepcopy(spec)
        duplicate["previews"].append(duplicate["previews"][0])
        with self.assertRaises(ValueError):
            v3.build_reading(duplicate, self.read, self.bundle)
        for change in ({"page_identity": "b" * 64}, {"codec": "webp"}, {"width": 200}):
            altered = copy.deepcopy(spec)
            altered["previews"][0].update(change)
            with self.assertRaises(ValueError):
                v3.build_reading(altered, self.read, self.bundle)
        altered = copy.deepcopy(spec)
        altered["previews"][0]["resource"]["bucket"] = shared.PDF_OCR_INPUT_BUCKET
        with self.assertRaises(ValueError):
            v3.build_reading(altered, self.read, self.bundle)

    def test_resource_path_traversal_unsupported_bucket_and_digest_rejected(self):
        valid = self.add("objects/aa/book/document.pdf", b"pdf")
        for change in ({"path": "objects/../secret"}, {"bucket": "foreign/bucket"},
                       {"path": "https://example.org/pdf"}, {"sha256": "invalid"}, {"bytes": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                v3.resource({**valid, **change})

    def test_backfill_uses_existing_page_objects_without_ocr_or_render(self):
        pages = []
        for number in range(1, 3):
            payload = {**raw_page("Text"), "page": number}
            raw = gzip.compress(json.dumps(payload).encode())
            ref = self.add(f"objects/aa/raw/ocr/page-{number:06d}.json.gz", raw)
            pages.append({"p": number, "source": "ocr", "o": ref["path"], "os": ref["sha256"], "ob": ref["bytes"]})
        manifest = {"kind": "pdf-ocr", "complete": True, "source_sha256": "a" * 64,
                    "page_count": 2, "language": "en", "pages": pages}
        ref = v3.backfill_text(manifest, self.read, self.bundle)
        self.generated()
        self.assertEqual(v3.verify_text_bundle(ref, self.read, "a" * 64, 2)["page_count"], 2)
        with self.assertRaises(ValueError):
            v3.backfill_text({**manifest, "pages": pages[:1]}, self.read, self.bundle)
        with self.assertRaises(ValueError):
            v3.backfill_text(manifest, self.read, self.bundle, {"rotation": 90})

    def test_stream_generation_is_resumable_and_validates_real_pixels(self):
        spec = self.spec()
        spec["previews"] = []
        one = v3.generate_previews(spec, self.read, self.bundle, start=2, end=2, max_pixels=10000)
        self.assertEqual([p["page"] for p in one["previews"]], [2])
        self.assertLessEqual(one["previews"][0]["width"] * one["previews"][0]["height"], 10000)
        resumed = v3.generate_previews(one, self.read, self.bundle, start=1, end=1, max_pixels=10000)
        self.generated()
        ref, manifest = v3.build_reading(resumed, self.read, self.bundle)
        self.generated()
        self.assertTrue(v3.verify_reading(ref, self.read)["preview"]["complete"])
        self.assertEqual([p["page"] for p in resumed["previews"]], [1, 2])

    def test_bitonal_page_gets_png_without_modifying_pdf(self):
        image = Image.new("1", (100, 200), 1)
        stream = io.BytesIO()
        image.save(stream, "TIFF", compression="group4")
        with pymupdf.open() as doc:
            page = doc.new_page(width=100, height=200)
            page.insert_image(page.rect, stream=stream.getvalue())
            raw = doc.tobytes()
        primary = self.add("objects/aa/binary/document.pdf", raw)
        spec = {"source_sha256": "a" * 64, "source_key": "book", "primary": primary,
                "require_complete_preview": True}
        generated = v3.generate_previews(spec, self.read, self.bundle, max_pixels=10000)
        self.assertEqual(generated["previews"][0]["codec"], "png")
        self.assertEqual(self.read(primary), raw)
        self.generated()
        ref, _ = v3.build_reading(generated, self.read, self.bundle)
        self.generated()
        v3.verify_reading(ref, self.read)
        generated["previews"][0]["codec"] = "webp"
        with self.assertRaises(ValueError):
            v3.build_reading(generated, self.read, self.bundle)

    def test_false_completeness_and_corrupt_preview_partition_rejected(self):
        ref, manifest = v3.build_reading(self.spec(), self.read, self.bundle)
        self.generated()
        incomplete = copy.deepcopy(manifest)
        incomplete["preview"]["partitions"] = []
        with self.assertRaises(ValueError):
            v3.validate_reading_manifest(incomplete)
        partition = manifest["preview"]["partitions"][0]["resource"]
        self.objects[(partition["bucket"], partition["path"])] = b"corrupt"
        with self.assertRaises(ValueError):
            v3.verify_reading(ref, self.read)

    def test_text_geometry_must_match_pdf_before_reading_publication(self):
        spec = self.spec()
        layers, pages = self.layers(2)
        ref = v3.build_text_bundle(layers, pages, Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), self.bundle)
        self.generated()
        spec["text_layer"] = ref
        # Page two is rotated to landscape; the supplied text coordinates are portrait.
        with self.assertRaises(ValueError):
            v3.build_reading(spec, self.read, self.bundle)

    def test_accepted_bundle_updates_search_and_retains_parent(self):
        layers, pages = self.layers(2)
        ref = v3.build_text_bundle(layers, pages, Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), self.bundle)
        self.generated()
        proposal = {"kind": "pdf-text-correction", "version": 1, "page": 1,
                    "base_generation": layers[0]["generation"], "raw_sha256": layers[0]["raw_sha256"],
                    "page_identity": layers[0]["page_identity"],
                    "replacements": [{"region_id": "b0", "before": "Text", "after": "Corrected text"}]}
        accepted = v3.accept_text_bundle(ref, [proposal], self.read, self.bundle, actor="reviewer")
        self.generated()
        manifest = v3.verify_text_bundle(accepted, self.read, "a" * 64, 2)
        self.assertEqual(manifest["parent"], ref)
        self.assertEqual(manifest["revision"], "effective")
        index = v3.decode(self.read(manifest["book_text"]))
        self.assertEqual([p["text"] for p in index["pages"]], ["Corrected text", "Text"])
        self.assertEqual(index["quality"], "partially-reviewed")
        repacked = v3.repartition_text_bundle(accepted, self.read, self.bundle)
        self.generated()
        refreshed = v3.verify_text_bundle(repacked, self.read, "a" * 64, 2)
        self.assertEqual(refreshed["generation"], manifest["generation"])
        self.assertEqual(refreshed["parent"], accepted)
        self.assertEqual(refreshed["revision"], "effective")
        search = v3.decode(self.read(refreshed["search_partitions"][0]["resource"]))
        self.assertEqual([p["text"] for p in search["pages"]], ["Corrected text", "Text"])
        with self.assertRaises(ValueError):
            v3.accept_text_bundle(accepted, [proposal], self.read, self.bundle, actor="reviewer")
        with self.assertRaises(ValueError):
            v3.accept_text_bundle(ref, [proposal, proposal], self.read, self.bundle, actor="reviewer")


if __name__ == "__main__":
    unittest.main()
