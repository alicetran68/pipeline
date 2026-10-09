import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub import HfApi
from PIL import Image, ImageDraw

from scripts import pdf_ocr, pdf_ocr_stages as stages, plan_pdf_ocr, shared
from scripts.build_reader_assets_index import build_index


class PdfReaderPresentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.objects = {}

    def source(self, modes=("1", "1")):
        images = []
        for mode in modes:
            image = Image.new(mode, (80, 120), "white")
            ImageDraw.Draw(image).text((5, 30), "Binary text", fill="black")
            images.append(image)
        path = self.root / ("source-" + "-".join(modes) + ".pdf")
        images[0].save(path, "PDF", save_all=True, append_images=images[1:], resolution=72)
        for image in images:
            image.close()
        return path

    def store(self, bundle):
        self.objects.update({path.relative_to(bundle).as_posix(): path.read_bytes()
                             for path in bundle.rglob("*") if path.is_file()})

    def read(self, meta, suffix=None):
        pdf_ocr.validate_ocr_object_path(meta["path"], suffix)
        data = self.objects[meta["path"]]
        self.assertEqual(len(data), meta["bytes"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), meta["sha256"])
        return data

    def item(self, source):
        return {"key": "VoiceOfML/Test\0scan.pdf", "repo": "VoiceOfML/Test", "path": "scan.pdf",
                "source_kind": "upstream", "source_extension": "pdf", "source_revision": "revision",
                "probe": pdf_ocr.probe_pdf(source), "reader_presentation": pdf_ocr.reader_presentation(source)}

    def test_real_ccitt_pdf_keeps_original_and_publishes_independent_ocr_without_reader_images(self):
        source = self.source()
        original = source.read_bytes()
        item = self.item(source)
        self.assertEqual(item["reader_presentation"]["strategy"], "preserve-pdf")
        self.assertEqual(item["reader_presentation"]["bitonal_images"], 2)
        self.assertIn("CCITTFaxDecode", item["reader_presentation"]["filters"])
        bundle = self.root / "render"
        with patch.object(pdf_ocr, "JXL_ENABLED", True), \
                patch.object(pdf_ocr, "encode_jxl", side_effect=AssertionError("must not recompress binary PDF")):
            rendered = stages.render_book(item, source, bundle)
            self.store(bundle)
            manifest = stages.validate_render(rendered, json.loads(self.read(rendered["render_manifest"])))
        self.assertIsNone(rendered["page_manifest"])
        self.assertFalse(rendered["image_rendered"])
        self.assertEqual(rendered["ocr_pages"], 2)
        self.assertEqual(list(bundle.rglob("*.webp")), [])
        self.assertEqual(list(bundle.rglob("*.jxl")), [])
        for page in manifest["pages"]:
            self.assertIn("i", page)
            self.assertNotIn("w", page)
        with patch.object(stages, "read_object", side_effect=self.read):
            queue = stages.plan_images({rendered["key"]: rendered}, {}, {})
            self.assertEqual(queue["total_ocr_pages"], 2)
            output = self.root / "ocr"
            with patch.object(pdf_ocr, "ocr_page", return_value=[
                    {"t": "recognized binary text", "b": [0, 0, 1, 1], "c": .9, "s": "ocr"}]):
                recognized = stages.recognize_task(queue["shards"][0][0], output)
            self.store(output)
            progress = stages.collect_progress(queue, [recognized])
            completed_bundle = self.root / "complete"
            completed = stages.assemble_book(queue["books"][0], progress[rendered["key"]]["pages"], completed_bundle)
        self.store(completed_bundle)
        self.assertEqual(completed["status"], "ready")
        self.assertFalse(completed["stream"])
        ocr = json.loads(self.objects[completed["ocr_manifest"]])
        self.assertNotIn("page_manifest", ocr)
        text = json.loads(gzip.decompress(self.read(ocr["book_text"])))
        self.assertTrue(text["complete"])
        self.assertEqual([page["text"] for page in text["pages"]], ["recognized binary text"] * 2)
        compact = build_index({"files": {}}, ocr_manifest={"files": {rendered["key"]: completed}})["f"][rendered["key"]]
        self.assertEqual(compact["s"], 3)
        self.assertIn("o", compact)
        self.assertNotIn("p", compact)
        self.assertEqual(source.read_bytes(), original)

    def test_small_rgb_is_not_assumed_binary_and_mixed_bitonal_book_is_preserved(self):
        color = self.source(("RGB",))
        self.assertLess(color.stat().st_size, 10 * 1024**2)
        self.assertEqual(pdf_ocr.reader_presentation(color)["strategy"], "page-stream")
        mixed = pdf_ocr.reader_presentation(self.source(("RGB", "1")))
        self.assertEqual(mixed["strategy"], "preserve-pdf")
        self.assertEqual(mixed["inspected_pages"], 2)
        self.assertEqual(mixed["bitonal_images"], 1)
        failed = pdf_ocr.reader_presentation(self.root / "missing.pdf")
        self.assertEqual(failed["strategy"], "preserve-pdf")
        self.assertFalse(failed["complete"])

    def test_binary_pdf_with_embedded_text_is_not_forced_into_reader_images(self):
        source = self.source()
        probe = {"page_count": 2, "classification": "native-text", "page_chars": [80, 80]}
        with patch.object(plan_pdf_ocr, "download_source", return_value=source), \
                patch.object(pdf_ocr, "probe_pdf", return_value=probe):
            queue = plan_pdf_ocr.plan([self.item(source)], workers=1, native_text_stream=True)
        planned = queue["shards"][0]["records"][0]
        self.assertEqual(planned["reader_presentation"]["strategy"], "preserve-pdf")
        self.assertFalse(planned.get("force_image_render", False))

    def test_legacy_monolithic_ocr_cannot_recompress_bitonal_pdf(self):
        source = self.source()
        with patch.object(pdf_ocr, "render_page") as render:
            with self.assertRaisesRegex(ValueError, "independent plan-render/plan-ocr"):
                pdf_ocr.build_item(self.item(source), source, self.root / "legacy")
        render.assert_not_called()

    def test_preserved_render_ranges_assemble_without_webp_and_reject_recompressed_pages(self):
        source = self.source()
        digest, size = shared.hash_file(source)
        book = {**self.item(source), "source_sha256": digest, "source_bytes": size,
                "render_profile": stages.render_profile(), "profile": pdf_ocr.asset_profile(), "page_count": 2}
        descriptors = {}
        for page in (1, 2):
            bundle = self.root / f"range-{page}"
            result = stages.render_book({**book, "start": page, "end": page}, source, bundle)
            self.store(bundle)
            descriptors[stages.range_id(page, page)] = result["descriptor"]
        with patch.object(stages, "read_object", side_effect=self.read):
            result = stages.assemble_render_book(book, descriptors, self.root / "assembled")
        self.assertFalse(result["image_rendered"])
        self.assertIsNone(result["page_manifest"])
        manifest = json.loads((self.root / "assembled" / result["render_manifest"]["path"]).read_bytes())
        manifest["pages"][0]["w"] = "forbidden.webp"
        with self.assertRaisesRegex(ValueError, "recompressed reader image"):
            stages.validate_render(result, manifest)

    def test_canonical_publish_retires_old_page_route_but_keeps_completed_text(self):
        source = self.source()
        bundle = self.root / "render"
        result = stages.render_book(self.item(source), source, bundle)
        key = result["key"]
        old = {**result, "reader_presentation": {"strategy": "page-stream"},
               "status": "ready", "stream": True, "image_rendered": True,
               "ocr_manifest": "objects/old/ocr-manifest.json",
               "render_manifest": {"path": "old-render-manifest.json"},
               "page_manifest": {"path": "objects/old/page-manifest.json"}}
        sidecar = {"v": 1, "f": {key: {"s": 2, "m": "p", "p": "objects/old/page-manifest.json",
                                       "o": old["ocr_manifest"], "b": shared.PDF_PAGES_BUCKET}}}
        api = HfApi()
        written = {}
        def registry(_api, _repo, name):
            return {"version": 1, "files": {key: copy.deepcopy(old)}
                    if name == stages.publication.OCR_MANIFEST_NAME else {}}
        with patch.object(stages, "load_registry", side_effect=registry), \
                 patch.object(stages.publication, "load_sidecar", return_value=sidecar), \
                 patch.object(stages, "publish_json", side_effect=lambda path, data, token: written.update({path: data})), \
                 patch.object(stages, "publish_bytes"), patch.object(stages, "publish_catalog", return_value="gen"):
            stages.save_registry(api, "unused", stages.RENDER_REGISTRY, {key: result}, publish_streams=True)
        current = written[stages.index_path(stages.publication.OCR_MANIFEST_NAME)]["files"][key]
        self.assertEqual(current["ocr_manifest"], old["ocr_manifest"])
        self.assertEqual(current["render_manifest"], result["render_manifest"])
        self.assertIsNone(current["page_manifest"])
        self.assertFalse(current["stream"])
        self.assertEqual(sidecar["f"][key]["s"], 3)
        self.assertNotIn("p", sidecar["f"][key])
        generated = {**result, "source_kind": "generated", "reader_assets_path": "derived/test/" + "a" * 32 + "/document.pdf",
                     "reader_assets_bucket": shared.PDF_PAGES_BUCKET}
        route = shared.merge_pdf_ocr_sidecar_entry(
            {"s": 2, "m": "p", "p": "objects/old/page-manifest.json"},
            {**generated, "ocr_manifest": old["ocr_manifest"]})
        self.assertEqual(route["p"], generated["reader_assets_path"])
        self.assertEqual(route["ob"], shared.PDF_PAGES_BUCKET)
        route = shared.preserve_pdf_sidecar_entry(
            {"s": 2, "m": "p", "p": "objects/old/page-manifest.json", "o": old["ocr_manifest"]},
            {**generated, "reader_assets_bucket": shared.READER_ASSETS_BUCKET})
        self.assertEqual(route["b"], shared.READER_ASSETS_BUCKET)
        self.assertEqual(route["ob"], shared.PDF_PAGES_BUCKET)
