import copy
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import yaml

from scripts import pdf_ocr, pdf_ocr_stages as stages, plan_pdf_ocr, shared
from scripts.build_reader_assets_index import build_index


class PdfOcrStagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.objects = {}

    def store(self, bundle):
        for path in bundle.rglob("*"):
            if path.is_file():
                self.objects[path.relative_to(bundle).as_posix()] = path.read_bytes()

    def read(self, meta, suffix=None):
        pdf_ocr.validate_ocr_object_path(meta["path"], suffix)
        data = self.objects[meta["path"]]
        if len(data) != meta["bytes"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError("object checksum mismatch")
        return data

    def item(self):
        return {"key": "repo\0small.pdf", "repo": "repo", "path": "small.pdf",
                "source_kind": "upstream", "source_revision": "revision"}

    def render_fixture(self, native=False, jxl=False, force_image=False, native_only=False):
        source = self.root / "source.pdf"
        source.write_bytes(b"%PDF-test-source")
        bundle = self.root / "render"
        def render(_source, page, directory, reader_pixels=None, reader_jxl=False):
            path = directory / f"page-{page:06d}.png"
            with Image.new("RGB", (200, 300), "white") as image:
                image.save(path)
                image.resize((100, 150)).save(path.with_suffix(".webp"))
            return path, 200, 300
        def encode(png, destination):
            with Image.open(png) as image:
                self.assertEqual(image.size, (200, 300))
            destination.write_bytes(b"jxl")
        item = {**self.item(), "probe": {"page_count": 2, "page_chars": [60, 60] if native_only else
                                         [60 if native else 0, 0],
                                         "classification": "native-text" if native_only else "mixed" if native else "scan"},
                **({"force_image_render": True} if force_image else {})}
        with patch.object(pdf_ocr, "render_page", side_effect=render), \
                patch.object(pdf_ocr, "JXL_ENABLED", jxl), \
                patch.object(pdf_ocr, "encode_jxl", side_effect=encode), \
                patch.object(pdf_ocr, "native_page", return_value={"width": 200, "height": 300,
                             "blocks": [], "text": "原生文字"}), \
                patch.object(pdf_ocr, "ocr_page", side_effect=AssertionError("renderer must not OCR")):
            result = stages.render_book(item, source, bundle)
        self.store(bundle)
        return result

    def test_render_persists_png_webp_jxl_and_native_text_before_ocr(self):
        result = self.render_fixture(native=True, jxl=True)
        manifest = json.loads(self.read(result["render_manifest"]))
        stages.validate_render(result, manifest)
        self.assertEqual(result["status"], "ready")
        self.assertLess(result["source_bytes"], 100 * 1024 ** 2)
        self.assertEqual(result["ocr_pages"], 1)
        for page in manifest["pages"]:
            for field in ("i", "w", "j"):
                self.read(stages.page_meta(page, field))
        native = manifest["pages"][0]
        self.assertEqual(json.loads(gzip.decompress(self.read(stages.page_meta(native, "o"))))["text"], "原生文字")
        self.assertNotIn("o", manifest["pages"][1])

    def test_failed_structure_native_pdf_gets_image_stream_and_jxl(self):
        result = self.render_fixture(native_only=True, jxl=True, force_image=True)
        manifest = json.loads(self.read(result["render_manifest"]))
        self.assertTrue(manifest["image_rendered"])
        self.assertTrue(result["image_rendered"])
        self.assertEqual(result["ocr_pages"], 0)
        self.assertIsNotNone(result["page_manifest"])
        for page in manifest["pages"]:
            for field in ("i", "w", "j"):
                self.read(stages.page_meta(page, field))
            self.assertEqual(page["source"], "native")
            self.read(stages.page_meta(page, "o"))
        with patch.object(stages, "read_object", side_effect=self.read):
            queue = stages.plan_images({result["key"]: result}, {}, {})
            self.assertEqual(queue["total_ocr_pages"], 0)
            book = stages.assemble_book(queue["books"][0], {}, self.root / "native-images-text")
        self.store(self.root / "native-images-text")
        text = json.loads(gzip.decompress(self.read(json.loads(self.objects[book["ocr_manifest"]])["book_text"])))
        self.assertEqual([page["text"] for page in text["pages"]], ["原生文字", "原生文字"])
        self.assertTrue(book["stream"])

    def test_old_native_range_without_images_is_not_reused(self):
        source = self.root / "source.pdf"
        source.write_bytes(b"%PDF-test-source")
        item = {**self.item(), "probe": {"page_count": 2, "page_chars": [60, 60],
                                        "classification": "native-text"}, "force_image_render": True,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "page_count": 2,
                "profile": pdf_ocr.asset_profile(), "render_profile": stages.render_profile()}
        old = {**item, "start": 1, "end": 2}
        old["force_image_render"] = False
        with patch.object(pdf_ocr, "native_page", return_value={"width": 200, "height": 300,
                 "blocks": [], "text": "原生文字"}):
            result = stages.render_book(old, source, self.root / "old-range")
        self.store(self.root / "old-range")
        progress = {item["key"]: {**stages.range_identity(item),
                    "ranges": {stages.range_id(1, 2): result["descriptor"]}}}
        with patch.object(stages, "read_object", side_effect=self.read):
            queue = stages.plan_render_ranges({"shards": [{"records": [item]}]}, progress)
        self.assertEqual(queue["saved_ranges"], {})
        self.assertEqual(len(queue["shards"][0]["records"]), 1)

    def test_force_reprobe_does_not_reuse_saved_render_ranges(self):
        queue = {"shards": [{"records": [{**self.item(), "key": "repo\\0book.pdf",
                                             "probe": {"page_count": 2, "classification": "scan"},
                                             "page_count": 2, "source_sha256": "a" * 64,
                                             "source_revision": "revision", "source_kind": "upstream"}]}]}
        planned = stages.plan_render_ranges(queue, {"repo\\0book.pdf": {"ranges": {"000001-000002": {}}}},
                                            force_reprobe=True)
        self.assertEqual(len(planned["shards"]), 1)

    def test_partial_native_text_prevents_source_pixel_cap(self):
        source = self.root / "mixed-input.pdf"
        source.write_bytes(b"%PDF-test-source")
        dimensions = []
        def render(_source, number, directory, reader_pixels=None, reader_jxl=False):
            dimensions.append(reader_pixels)
            path = directory / f"page-{number:06d}.png"
            with Image.new("RGB", (200, 300), "white") as image:
                image.save(path)
                image.save(path.with_suffix(".webp"))
            return path, 200, 300
        item = {**self.item(), "probe": {"classification": "scan", "page_count": 2,
                                        "page_chars": [12, 0]}}
        with patch.object(pdf_ocr, "scan_reader_images", return_value={1: (100, 150), 2: (100, 150)}), \
                patch.object(pdf_ocr, "render_page", side_effect=render), \
                patch.object(pdf_ocr, "JXL_ENABLED", False):
            stages.render_book(item, source, self.root / "mixed-input-render")
        self.assertEqual(dimensions, [None, (100, 150)])

    def test_failed_structure_native_ready_render_is_rebuilt_for_images(self):
        item = {**self.item(), "force_image_render": True}
        previous = {**item, "status": "ready", "render_profile": stages.render_profile(),
                    "image_rendered": False}
        self.assertEqual([x["key"] for x in stages.pending_render([item], {item["key"]: previous}, {})],
                         [item["key"]])
        previous["image_rendered"] = True
        self.assertEqual(stages.pending_render([item], {item["key"]: previous}, {}), [])

    def test_png_only_ocr_resume_and_complete_book_publication(self):
        result = self.render_fixture(native=True)
        with patch.object(stages, "read_object", side_effect=self.read), \
                patch.object(stages, "source_path", side_effect=AssertionError("OCR must not download PDF")), \
                patch.object(pdf_ocr, "render_page", side_effect=AssertionError("OCR must not render")):
            queue = stages.plan_images({result["key"]: result}, {}, {}, target=500)
            self.assertEqual(queue["total_ocr_pages"], 1)
            book = queue["books"][0]
            incomplete = stages.assemble_book(book, {}, self.root / "incomplete")
            self.assertEqual(incomplete["status"], "failed")
            def recognize(path, width, height):
                self.assertEqual(path.suffix, ".png")
                self.assertEqual((width, height), (200, 300))
                return [{"t": "识别结果", "b": [0, 0, 1, 1], "c": 1, "s": "ocr"}]
            with patch.object(pdf_ocr, "ocr_page", side_effect=recognize) as engine:
                output = self.root / "ocr"
                recognized = stages.recognize_task(queue["shards"][0][0], output)
                self.store(output)
            engine.assert_called_once()
            progress = stages.collect_progress(queue, [recognized])
            resumed = stages.plan_images({result["key"]: result}, {}, progress)
            self.assertEqual(resumed["shard_count"], 0)
            finished = self.root / "finished"
            completed = stages.assemble_book(resumed["books"][0], progress[result["key"]]["pages"], finished)
            self.assertEqual(completed["status"], "ready")
            self.store(finished)
            manifest = json.loads(self.objects[completed["ocr_manifest"]])
            text = json.loads(gzip.decompress(self.read(manifest["book_text"])))
            self.assertEqual((text["version"], text["kind"], text["complete"]), (2, "pdf-book-text", True))
            self.assertEqual((text["language"], text["ocr_version"]), pdf_ocr.resolve_ocr_config("ch", "rapidocr_onnxruntime")[:2])
            self.assertEqual([p["text"] for p in text["pages"]], ["原生文字", "识别结果"])
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["page_manifest"], result["page_manifest"])

    def test_page_checksum_failure_does_not_publish_ready(self):
        result = self.render_fixture()
        with patch.object(stages, "read_object", side_effect=self.read):
            queue = stages.plan_images({result["key"]: result}, {}, {})
            task = queue["shards"][0][0]
            self.objects[task["pages"][0]["i"]] = b"corrupt"
            with patch.object(pdf_ocr, "ocr_page", return_value=[]) as engine:
                recognized = stages.recognize_task(task, self.root / "ocr")
            self.assertEqual(len(recognized["pages"]), 1)
            self.assertEqual(len(recognized["errors"]), 1)
            self.assertEqual(engine.call_count, 1)
            progress = stages.collect_progress(queue, [recognized])
            self.assertEqual(stages.assemble_book(queue["books"][0], progress[result["key"]]["pages"],
                                                  self.root / "result")["status"], "failed")
            retried = stages.plan_images({result["key"]: result}, {}, progress)
            self.assertEqual(retried["total_ocr_pages"], 1)

    def test_500_page_tasks_split_large_books_and_keep_every_page(self):
        result = self.render_fixture()
        manifest = json.loads(self.read(result["render_manifest"]))
        template = manifest["pages"][0]
        pages = []
        for number in range(1, 1202):
            page = copy.deepcopy(template)
            page["p"] = number
            for field in ("i", "w"):
                page[field] = page[field].replace("000001", f"{number:06d}")
            pages.append(page)
        manifest["pages"] = pages
        manifest["page_count"] = result["page_count"] = len(pages)
        with patch.object(stages, "read_object", return_value=json.dumps(manifest).encode()):
            queue = stages.plan_images({result["key"]: result}, {}, {})
        self.assertEqual(queue["shard_count"], 3)
        numbers = [p["p"] for shard in queue["shards"] for task in shard for p in task["pages"]]
        self.assertEqual(sorted(numbers), list(range(1, 1202)))
        self.assertEqual(max(len(task["pages"]) for shard in queue["shards"] for task in shard), 500)

    def test_large_render_book_ranges_resume_and_publish_only_when_complete(self):
        source = self.root / "source.pdf"
        source.write_bytes(b"%PDF-range-source")
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        item = {**self.item(), "source_sha256": sha, "source_bytes": source.stat().st_size,
                "probe": {"page_count": 5, "page_chars": [0] * 5, "classification": "scan"},
                "page_count": 5, "profile": pdf_ocr.asset_profile()}
        queue = {"version": 1, "kind": "pdf-render-queue", "shards": [{"records": [item]}]}
        def render(_source, number, directory, reader_pixels=None, reader_jxl=False):
            path = directory / f"page-{number:06d}.png"
            with Image.new("RGB", (200, 300), "white") as image:
                image.save(path)
                image.save(path.with_suffix(".webp"))
            return path, 200, 300
        with patch.object(stages, "RENDER_RANGE_THRESHOLD", 2), \
                patch.object(stages, "RENDER_RANGE_PAGES", 2), \
                patch.object(pdf_ocr, "JXL_ENABLED", False), \
                patch.object(pdf_ocr, "render_page", side_effect=render):
            planned = stages.plan_render_ranges(queue, {})
            tasks = [task for shard in planned["shards"] for task in shard["records"]]
            self.assertEqual([(task["start"], task["end"]) for task in sorted(tasks, key=lambda t: t["start"])],
                             [(1, 2), (3, 4), (5, 5)])
            book = planned["books"][0]
            descriptors = {}
            for task in tasks[:2]:
                bundle = self.root / f"range-{task['start']}"
                result = stages.render_book(task, source, bundle)
                self.store(bundle)
                descriptors[stages.range_id(task["start"], task["end"])] = result["descriptor"]
            with patch.object(stages, "read_object", side_effect=self.read):
                self.assertIsNone(stages.assemble_render_book(book, descriptors, self.root / "incomplete"))
                progress = {book["key"]: {**stages.range_identity(book), "ranges": descriptors}}
                resumed = stages.plan_render_ranges(queue, progress)
                remaining = [task for shard in resumed["shards"] for task in shard["records"]]
                self.assertEqual([(t["start"], t["end"]) for t in remaining], [(5, 5)])
            bundle = self.root / "range-last"
            final_range = stages.render_book(remaining[0], source, bundle)
            self.store(bundle)
            descriptors[stages.range_id(5, 5)] = final_range["descriptor"]
            with patch.object(stages, "read_object", side_effect=self.read):
                final_bundle = self.root / "render-final"
                result = stages.assemble_render_book(book, descriptors, final_bundle)
                self.store(final_bundle)
                manifest = json.loads(self.read(result["render_manifest"]))
                stages.validate_render(result, manifest)
                self.assertEqual([p["p"] for p in manifest["pages"]], list(range(1, 6)))
                self.assertEqual(manifest["page_manifest"]["version"], 2)
                with self.assertRaises(ValueError):
                    stages.validate_range(book, 3, 4, descriptors[stages.range_id(1, 2)])
                self.objects[descriptors[stages.range_id(3, 4)]["path"]] = b"damaged"
                retry = stages.plan_render_ranges(queue, {book["key"]: {
                    **stages.range_identity(book), "ranges": descriptors}})
                missing = [task for shard in retry["shards"] for task in shard["records"]]
                self.assertEqual([(t["start"], t["end"]) for t in missing], [(3, 4)])

    def test_render_plan_balances_very_large_books_by_page_range(self):
        book = {**self.item(), "source_sha256": "a" * 64, "page_count": 2001,
                "profile": pdf_ocr.asset_profile()}
        queue = stages.plan_render_ranges({"shards": [{"records": [book]}]}, {})
        tasks = [t for shard in queue["shards"] for t in shard["records"]]
        self.assertEqual(len(tasks), 9)
        self.assertEqual(sum(t["end"] - t["start"] + 1 for t in tasks), 2001)
        self.assertGreater(queue["shard_count"], 1)
        self.assertLessEqual(max(s["page_count"] for s in queue["shards"]), 500)

    def test_render_publication_saves_partial_progress_then_only_complete_book(self):
        previous = self.render_fixture()
        manifest = json.loads(self.read(previous["render_manifest"]))
        book = {**self.item(), "source_sha256": previous["source_sha256"],
                "source_bytes": previous["source_bytes"], "page_count": 2,
                "profile": previous["profile"], "render_profile": previous["render_profile"],
                "probe": {"classification": "scan", "page_count": 2, "page_chars": [0, 0]}}
        descriptors = {}
        range_bundle = self.root / "range-descriptors"
        root = stages.root_for(book["source_sha256"], book["key"], book["render_profile"])
        for page in manifest["pages"]:
            number = page["p"]
            destination = range_bundle / root / f"render-range-{stages.range_id(number, number)}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pdf_ocr.write_json(destination, {**stages.range_identity(book), "version": 1,
                                             "kind": "pdf-render-range", "start": number, "end": number,
                                             "pages": [page]})
            descriptors[number] = stages.metadata(destination, range_bundle)
        self.store(range_bundle)
        updates = []
        with patch.object(stages, "RENDER_RANGE_THRESHOLD", 1), \
                patch.object(stages, "RENDER_RANGE_PAGES", 1), \
                patch.object(stages, "read_object", side_effect=self.read), \
                patch.object(stages, "upload_objects", side_effect=self.store), \
                patch.object(stages, "save_registry", side_effect=lambda api, repo, name, data, **kw:
                             updates.append((name, data))):
            for selected, saved in ((1, {}), (2, {stages.range_id(1, 1): descriptors[1]})):
                task = {**book, "start": selected, "end": selected}
                queue = {"version": 1, "kind": "pdf-render-queue", "books": [book],
                         "shards": [{"records": [task]}], "saved_ranges": {book["key"]: saved},
                         "failed": []}
                location = self.root / f"queue-{selected}.json"
                pdf_ocr.write_json(location, queue)
                results_dir = self.root / f"results-{selected}"
                results_dir.mkdir()
                pdf_ocr.write_json(results_dir / f"results-{selected}.json", {"version": 1, "results": [{
                    **stages.range_identity(book), "start": selected, "end": selected,
                    "status": "range", "descriptor": descriptors[selected]}]})
                with patch.object(sys, "argv", ["pdf_ocr_stages.py", "publish-render", "--queue", str(location),
                                                "--results-dir", str(results_dir), "--output", str(self.root)]):
                    stages.main()
                state = updates[-1][1][book["key"]]
                if selected == 1:
                    self.assertEqual(state["status"], "failed")
                    self.assertNotIn("page_manifest", state)
                else:
                    self.assertEqual(state["status"], "ready")
                    finished = json.loads(self.read(state["render_manifest"]))
                    self.assertEqual([page["p"] for page in finished["pages"]], [1, 2])
        self.assertEqual([name for name, _ in updates], [stages.RENDER_PROGRESS_REGISTRY,
                          stages.RENDER_REGISTRY, stages.RENDER_PROGRESS_REGISTRY, stages.RENDER_REGISTRY])

    def test_missing_native_or_duplicate_page_is_rejected(self):
        result = self.render_fixture(native=True)
        manifest = json.loads(self.read(result["render_manifest"]))
        del manifest["pages"][0]["o"]
        with self.assertRaises(KeyError):
            stages.validate_render(result, manifest)
        manifest = json.loads(self.read(result["render_manifest"]))
        manifest["pages"][1]["p"] = 1
        with self.assertRaises(ValueError):
            stages.validate_render(result, manifest)
        manifest = json.loads(self.read(result["render_manifest"]))
        manifest["pages"][1]["i"] = manifest["pages"][1]["i"].replace(
            result["source_sha256"], "b" * 64)
        with self.assertRaisesRegex(ValueError, "outside book profile"):
            stages.validate_render(result, manifest)
        jxl = self.render_fixture(jxl=True)
        manifest = json.loads(self.read(jxl["render_manifest"]))
        del manifest["pages"][0]["j"]
        with self.assertRaises(KeyError):
            stages.validate_render(jxl, manifest)

    def test_read_png_uses_resolve_without_api_metadata_lookup(self):
        data = b"image"
        meta = {"path": "objects/aa/" + "a" * 64 + "/" + "b" * 16 + "/ocr-input/page-000001.png",
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        response = Mock(content=data)
        with patch.object(stages, "get_session") as session, patch.object(stages, "hf_raise_for_status"):
            session.return_value.get.return_value = response
            self.assertEqual(stages.read_object(meta), data)
            url = session.return_value.get.call_args.args[0]
            self.assertIn("/resolve/objects/", url)
            self.assertNotIn("/api/", url)

    def test_transfers_scope_each_book_not_whole_bucket(self):
        result = self.render_fixture()
        with patch.object(stages, "HfApi") as api:
            stages.upload_objects(self.root / "render")
        call = api.return_value.sync_bucket.call_args
        self.assertTrue(call.args[1].endswith(str(Path(result["render_manifest"]["path"]).parent)))
        self.assertNotEqual(call.args[1], stages.BUCKET)

    def test_planner_includes_small_pdf_and_rebuilds_old_ocr_for_v2_index(self):
        item = {**self.item(), "source_bytes": 1024}
        self.assertEqual(stages.pending_render([item], {}, {}), [item])
        old = {**item, "status": "ready", "profile": pdf_ocr.asset_profile(),
               "ocr_manifest": "objects/aa/" + "a" * 64 + "/" + "b" * 16 + "/ocr-manifest.json"}
        self.assertEqual(stages.pending_render([item], {}, {item["key"]: old}), [item])
        self.assertEqual(stages.pending_render([item], {}, {item["key"]: {**old, "ocr_manifest": ""}}), [item])

    def test_render_partitions_keep_small_and_large_files_independent(self):
        small = {**self.item(), "source_bytes": stages.SMALL_RENDER_MAX_SOURCE_BYTES - 1}
        large = {**self.item(), "key": "repo\0large.pdf", "path": "large.pdf",
                 "source_bytes": stages.SMALL_RENDER_MAX_SOURCE_BYTES}
        self.assertTrue(stages.render_partition_matches(small, "small"))
        self.assertFalse(stages.render_partition_matches(small, "large"))
        self.assertFalse(stages.render_partition_matches(large, "small"))
        self.assertTrue(stages.render_partition_matches(large, "large"))
        self.assertTrue(stages.render_partition_matches(small, "all"))

    def test_render_workflow_partitions_and_enables_native_text_streams(self):
        root = Path(__file__).resolve().parents[1]
        large = (root / ".github/workflows/pdf-render-inputs.yml").read_text()
        small = (root / ".github/workflows/pdf-render-small-inputs.yml").read_text()
        self.assertIn("plan-render --partition large --native-text-stream", large)
        self.assertIn("plan-render --partition small --native-text-stream", small)
        self.assertIn("--source-repo", small)
        self.assertIn("--source-path-prefix", small)
        self.assertIn("fonts-noto-cjk", small)
        ocr = (root / ".github/workflows/pdf-ocr-assets.yml").read_text()
        self.assertIn("--source-repo", ocr)
        self.assertIn("--source-path-prefix", ocr)
        self.assertIn("group: pdf-render-small-inputs", small)
        self.assertIn("group: reader-assets", small)

    def test_native_text_stream_plan_marks_pages_for_images_without_ocr(self):
        item = {**self.item(), "source_bytes": 1024}
        probe = {"page_count": 2, "page_chars": [80, 80], "classification": "native-text"}
        with patch.object(plan_pdf_ocr, "download_source", return_value=self.root / "source.pdf"), \
                patch.object(plan_pdf_ocr.shared, "hash_file", return_value=("a" * 64, 1024)), \
                patch.object(pdf_ocr, "probe_pdf", return_value=probe):
            (self.root / "source.pdf").write_bytes(b"pdf")
            queue = plan_pdf_ocr.plan([item], workers=1, native_text_stream=True)
        self.assertTrue(queue["shards"][0]["records"][0]["force_image_render"])

    def test_gbk_repaired_generated_pdf_is_not_misclassified_as_ocr_input(self):
        entry = {"source_kind": "generated",
                 "reader_assets_path": "objects/aa/" + "a" * 64 + "/gbk-font-repair-v1/document.pdf"}
        self.assertTrue(stages.skip_ocr_for_generated_text_pdf(entry))
        self.assertFalse(stages.skip_ocr_for_generated_text_pdf({"source_kind": "upstream"}))

    def test_failed_books_do_not_starve_untouched_backlog(self):
        failed = {**self.item(), "key": "repo\0a.pdf", "path": "a.pdf"}
        untouched = {**self.item(), "key": "repo\0z.pdf", "path": "z.pdf"}
        rendered = {failed["key"]: {**failed, "render_profile": stages.render_profile(), "status": "failed"}}
        self.assertEqual([x["key"] for x in stages.pending_render(
            [failed, untouched], rendered, {}, retry_failed=True)],
            [untouched["key"], failed["key"]])

        with patch.object(stages, "read_object", return_value=b"{}"), \
                patch.object(stages, "validate_render", return_value={"pages": []}):
            entries = {key: {**record, "status": "ready", "render_profile": "current", "profile": "p",
                             "source_sha256": "a" * 64, "page_count": 0,
                             "render_manifest": {"sha256": "b" * 64}}
                       for key, record in [(failed["key"], failed), (untouched["key"], untouched)]}
            queue = stages.plan_images(entries, {failed["key"]: {"status": "failed"}}, {}, limit=1)
            recovery = stages.plan_images(entries, {failed["key"]: {"status": "failed"}}, {},
                                          limit=1, retry_failed_only=True)
        self.assertEqual(queue["books"][0]["key"], untouched["key"])
        self.assertEqual([book["key"] for book in recovery["books"]], [failed["key"]])

    def test_native_only_pdf_builds_complete_book_without_png_or_ocr_worker(self):
        source = self.root / "native.pdf"
        source.write_bytes(b"%PDF-native")
        bundle = self.root / "native-render"
        item = {**self.item(), "probe": {"page_count": 2, "page_chars": [80, 80],
                                        "classification": "native-text"}}
        def native(_source, page):
            return {"width": 500, "height": 700,
                    "blocks": [{"t": "竖排正文", "b": [.7, .1, .75, .5], "c": 1, "s": "native"}],
                    "text": f"竖排正文{page}"}
        with patch.object(pdf_ocr, "render_page", side_effect=AssertionError("native PDF must not render")), \
                patch.object(pdf_ocr, "native_page", side_effect=native):
            result = stages.render_book(item, source, bundle)
        self.assertIsNone(result["page_manifest"])
        self.store(bundle)
        with patch.object(stages, "read_object", side_effect=self.read):
            queue = stages.plan_images({result["key"]: result}, {}, {})
            self.assertEqual(queue["shard_count"], 0)
            self.assertEqual(queue["total_ocr_pages"], 0)
            finished = self.root / "native-finished"
            completed = stages.assemble_book(queue["books"][0], {}, finished)
            self.assertTrue(completed["ocr_manifest"])
            self.assertFalse(completed["stream"])
            self.store(finished)
            text = json.loads(gzip.decompress(self.read(json.loads(self.objects[completed["ocr_manifest"]])["book_text"])))
            self.assertEqual(len(text["pages"]), 2)
            self.assertEqual((text["version"], text["kind"], text["complete"]), (2, "pdf-book-text", True))
            self.assertTrue(text["pages"][0]["text_spans"])
            self.assertIn("text_spans", text["pages"][0])

    def test_sidecar_rebuild_preserves_rendered_stream_without_advertising_ocr(self):
        result = self.render_fixture()
        base = {"files": {result["key"]: {"status": "ready", "reader_mode": "pdf",
                                         "path": "ordinary.pdf"}}}
        for status in ("rendered", "failed"):
            entry = {**result, "status": status}
            state = {"version": 1, "files": {entry["key"]: entry}}
            pdf_ocr.validate_manifest(state)
            index = build_index(base, ocr_manifest=state)
            compact = index["f"][entry["key"]]
            self.assertEqual(compact["p"], result["page_manifest"]["path"])
            self.assertEqual(compact["b"], "vomebook/pdf-pages")
            self.assertNotIn("o", compact)

    def test_completed_ocr_page_stream_replaces_existing_pdf_route(self):
        result = self.render_fixture()
        existing = {"s": 2, "m": "p", "p": "objects/old/document.pdf",
                    "b": "vomebook/pdf-optimized"}
        base = {"files": {result["key"]: {"status": "ready", "reader_mode": "pdf",
                                           "path": "ordinary.pdf"}}}
        ocr = {"files": {result["key"]: {**result, "status": "ready",
                                           "ocr_manifest": "objects/old/ocr-manifest.json"}}}
        merged = build_index(base, ocr_manifest=ocr)["f"][result["key"]]
        self.assertEqual(merged["p"], result["page_manifest"]["path"])
        self.assertEqual(merged["b"], "vomebook/pdf-pages")
        self.assertEqual(merged["o"], "objects/old/ocr-manifest.json")
        direct = shared.merge_pdf_ocr_sidecar_entry(existing, ocr["files"][result["key"]])
        self.assertEqual(direct["p"], result["page_manifest"]["path"])
        self.assertEqual(direct["b"], "vomebook/pdf-pages")
        self.assertEqual(direct["o"], merged["o"])

    def test_render_refresh_persists_stream_route_with_existing_ready_ocr(self):
        result = self.render_fixture()
        old_manifest = {**result["page_manifest"], "path": "objects/old/page-manifest.json"}
        previous = {**result, "status": "ready", "ocr_manifest": "objects/old/ocr-manifest.json",
                    "page_manifest": old_manifest}
        sidecar_path = self.root / "refresh-sidecar.json.gz"
        sidecar_path.write_bytes(gzip.compress(json.dumps({"v": 1, "f": {
            result["key"]: {"s": 2, "m": "p", "p": "ordinary.pdf", "b": "vomebook/pdf-optimized",
                            "o": previous["ocr_manifest"], "ob": "vomebook/pdf-pages"}}}).encode()))
        api = Mock()
        api.repo_info.return_value.sha = "pinned-revision"
        api.hf_hub_download.return_value = str(sidecar_path)
        def state(_api, _repo, name, _revision):
            return {"version": 1, "files": {result["key"]: previous}
                    if name == "pdf_ocr_manifest.json" else {}}
        with patch.object(stages, "load_registry", side_effect=state):
            stages.save_registry(api, "test/repo", stages.RENDER_REGISTRY,
                                 {result["key"]: result}, publish_streams=True)
        operations = {op.path_in_repo: op.path_or_fileobj for op in api.create_commit.call_args.kwargs["operations"]}
        stored = json.loads(operations["pdf_ocr_manifest.json"])["files"][result["key"]]
        self.assertEqual(stored["status"], "ready")
        self.assertEqual(stored["ocr_manifest"], previous["ocr_manifest"])
        self.assertEqual(stored["page_manifest"], result["page_manifest"])
        route = json.loads(gzip.decompress(operations["reader_assets.json.gz"]))["f"][result["key"]]
        rebuilt = build_index({"files": {}}, ocr_manifest={"files": {result["key"]: stored}})["f"][result["key"]]
        self.assertEqual(route["p"], rebuilt["p"])
        self.assertEqual(route["p"], result["page_manifest"]["path"])
        self.assertEqual(route["o"], rebuilt["o"])

    def test_failed_native_optimization_replaces_pdf_route_and_keeps_text(self):
        result = {**self.render_fixture(native_only=True, force_image=True),
                  "range_status": "failed", "classification": "native-text"}
        existing = {**result, "status": "ready", "ocr_manifest": "objects/old/ocr-manifest.json",
                    "page_manifest": None}
        base = {"files": {result["key"]: {"status": "ready", "reader_mode": "pdf",
                                     "path": "ordinary.pdf"}}}
        merged = build_index(base, ocr_manifest={"files": {result["key"]: {**existing,
                         "page_manifest": result["page_manifest"]}}})["f"][result["key"]]
        self.assertEqual(merged["p"], result["page_manifest"]["path"])
        self.assertEqual(merged["o"], existing["ocr_manifest"])
        sidecar_path = self.root / "existing-reader.json.gz"
        sidecar_path.write_bytes(gzip.compress(json.dumps({"v": 1, "f": {
            result["key"]: {"s": 2, "m": "p", "p": "ordinary.pdf", "o": existing["ocr_manifest"]}}}).encode()))
        api = Mock()
        api.repo_info.return_value.sha = "pinned-revision"
        api.hf_hub_download.return_value = str(sidecar_path)
        def state(_api, _repo, name, _revision):
            return {"version": 1, "files": {result["key"]: existing} if name == "pdf_ocr_manifest.json" else {}}
        with patch.object(stages, "load_registry", side_effect=state):
            stages.save_registry(api, "test/repo", stages.RENDER_REGISTRY,
                                 {result["key"]: result}, publish_streams=True)
        operations = {op.path_in_repo: op.path_or_fileobj for op in api.create_commit.call_args.kwargs["operations"]}
        stored = json.loads(operations["pdf_ocr_manifest.json"])["files"][result["key"]]
        self.assertEqual(stored["status"], "ready")
        self.assertEqual(stored["ocr_manifest"], existing["ocr_manifest"])
        self.assertEqual(stored["page_manifest"], result["page_manifest"])
        route = json.loads(gzip.decompress(operations["reader_assets.json.gz"]))["f"][result["key"]]
        self.assertEqual(route["p"], result["page_manifest"]["path"])
        self.assertEqual(route["o"], existing["ocr_manifest"])

    def test_workflows_separate_rendering_and_recognition(self):
        root = Path(__file__).resolve().parents[1]
        render_text = (root / ".github/workflows/pdf-render-inputs.yml").read_text()
        ocr_text = (root / ".github/workflows/pdf-ocr-assets.yml").read_text()
        render = yaml.safe_load(render_text)
        ocr = yaml.safe_load(ocr_text)
        self.assertIn("pdf_ocr_stages.py render", render_text)
        self.assertNotIn("requirements-pdf-ocr", render_text)
        self.assertNotIn("poppler-utils", ocr_text)
        self.assertNotIn("fetch_and_parse", ocr_text)
        self.assertIn("Render PDF OCR Inputs", ocr_text)
        self.assertIn("lang:", ocr_text)
        self.assertIn("PDF_OCR_LANG", ocr_text)
        self.assertIn("backend:", ocr_text)
        self.assertIn("PDF_OCR_BACKEND", ocr_text)
        stages_text = (root / "scripts/pdf_ocr_stages.py").read_text()
        self.assertIn('load_registry(api, repo, "pdf_range_manifest.json", revision)', stages_text)
        self.assertFalse(ocr[True]["workflow_dispatch"]["inputs"]["retry_failed_only"]["default"])
        self.assertIn("--retry-failed-only", ocr_text)
        self.assertIn('default: "auto"', ocr_text)
        self.assertIn('default: "rapidocr_onnxruntime"', ocr_text)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", ocr["jobs"]["plan"]["if"])
        self.assertIn("!cancelled()", ocr["jobs"]["publish"]["if"])
        self.assertEqual(ocr[True]["workflow_dispatch"]["inputs"]["limit"]["default"], "100")
        self.assertIn("inputs.limit || '100'", ocr_text)
        self.assertEqual(render["jobs"]["publish"]["concurrency"]["group"],
                         ocr["jobs"]["publish"]["concurrency"]["group"])
        self.assertEqual(render["jobs"]["build"]["strategy"]["max-parallel"], 10)
        self.assertEqual(ocr[True]["workflow_dispatch"]["inputs"]["target_pages"]["default"], "2000")

    def test_scheduled_render_drains_pending_in_webp_batches(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / ".github/workflows/pdf-render-inputs.yml").read_text()
        workflow = yaml.safe_load(text)
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["limit"]["default"], "100")
        self.assertFalse(inputs["generate_jxl"]["default"])
        self.assertEqual(workflow["env"]["PDF_JXL_ENABLED"], "${{ inputs.generate_jxl == true }}")
        self.assertEqual(workflow["jobs"]["plan"]["steps"][5]["env"]["CHECKPOINT"], "${{ inputs.checkpoint || '0' }}")

    def test_render_registry_stream_and_pending_ocr_are_committed_together(self):
        result = self.render_fixture()
        sidecar_path = self.root / "reader.json.gz"
        sidecar_path.write_bytes(gzip.compress(json.dumps({"v": 1, "f": {}}).encode()))
        api = Mock()
        api.repo_info.return_value.sha = "pinned-revision"
        api.hf_hub_download.return_value = str(sidecar_path)
        with patch.object(stages, "load_registry", side_effect=lambda *args: {"version": 1, "files": {}}):
            stages.save_registry(api, "test/repo", stages.RENDER_REGISTRY,
                                 {result["key"]: result}, publish_streams=True)
        call = api.create_commit.call_args.kwargs
        self.assertEqual(call["parent_commit"], "pinned-revision")
        operations = {op.path_in_repo: op.path_or_fileobj for op in call["operations"]}
        self.assertEqual(set(operations), {stages.RENDER_REGISTRY, "pdf_ocr_manifest.json", "reader_assets.json.gz"})
        ocr_state = json.loads(operations["pdf_ocr_manifest.json"])
        self.assertEqual(ocr_state["files"][result["key"]]["status"], "rendered")
        reader = json.loads(gzip.decompress(operations["reader_assets.json.gz"]))
        self.assertNotIn("o", reader["f"][result["key"]])

    def test_old_generation_progress_is_not_reused(self):
        result = self.render_fixture()
        progress = {result["key"]: {"generation": "old", "pages": {"1": {"o": "stale"}}}}
        with patch.object(stages, "read_object", side_effect=self.read):
            queue = stages.plan_images({result["key"]: result}, {}, progress)
        self.assertEqual(queue["total_ocr_pages"], 2)
        with self.assertRaisesRegex(ValueError, "generation"):
            stages.collect_progress(queue, [{"key": result["key"], "generation": "old", "pages": []}])

    def test_reader_image_refresh_reuses_matching_ocr_png_and_rebuilds_text_index(self):
        result = self.render_fixture(native=True)
        with patch.object(stages, "read_object", side_effect=self.read):
            first = stages.plan_images({result["key"]: result}, {}, {})
            with patch.object(pdf_ocr, "ocr_page", return_value=[
                    {"t": "recognized", "b": [0, 0, 1, 1], "c": 1, "s": "ocr"}]):
                recognized = stages.recognize_task(first["shards"][0][0], self.root / "old-ocr")
            self.store(self.root / "old-ocr")
            progress = stages.collect_progress(first, [recognized])
            old = stages.assemble_book(first["books"][0], progress[result["key"]]["pages"], self.root / "old-text")
            self.store(self.root / "old-text")

            manifest = json.loads(self.read(result["render_manifest"]))
            new_manifest = {**result["page_manifest"], "sha256": "f" * 64}
            manifest["page_manifest"] = new_manifest
            raw = json.dumps(manifest).encode()
            new_result = {**result, "page_manifest": new_manifest,
                          "render_manifest": {**result["render_manifest"],
                                              "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
            self.objects[result["render_manifest"]["path"]] = raw
            jxl_only = {**new_result, "profile": old["profile"].replace("-jxl-0-", "-jxl-1-")}
            self.assertEqual(set(stages.reuse_recognized_pages(old, jxl_only, manifest["pages"])), {"2"})
            changed_layout = {**jxl_only, "profile": jxl_only["profile"].rsplit("-layout-", 1)[0]
                              + "-layout-v1-" + "e" * 16}
            self.assertEqual(stages.reuse_recognized_pages(old, changed_layout, manifest["pages"]), {})
            queue = stages.plan_images({result["key"]: new_result}, {result["key"]: old}, {})
            self.assertEqual(queue["total_ocr_pages"], 0)
            self.assertEqual(queue["shard_count"], 0)
            self.assertEqual(queue["books"][0]["saved"]["2"]["i"], manifest["pages"][1]["i"])
            complete = stages.assemble_book(queue["books"][0], queue["books"][0]["saved"], self.root / "new-text")
            self.store(self.root / "new-text")
            index = json.loads(gzip.decompress(self.read(json.loads(self.objects[complete["ocr_manifest"]])["book_text"])))
            self.assertEqual([page["text"] for page in index["pages"]], ["原生文字", "recognized"])
            self.assertEqual(json.loads(self.objects[complete["ocr_manifest"]])["page_manifest"], new_manifest)

            changed = copy.deepcopy(manifest)
            changed["pages"][1]["is"] = "e" * 64
            raw = json.dumps(changed).encode()
            self.objects[result["render_manifest"]["path"]] = raw
            altered = {**new_result, "render_manifest": {**new_result["render_manifest"],
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
            pending = stages.plan_images({result["key"]: altered}, {result["key"]: old}, {})
            self.assertEqual(pending["total_ocr_pages"], 1)

    def test_result_discovery_handles_one_flat_artifact_and_multiple_nested_artifacts(self):
        results = self.root / "results"
        results.mkdir()
        flat = results / "results-0.json"
        flat.write_text('{"version":1,"results":[]}')
        nested = results / "artifact-1" / "results-1.json"
        nested.parent.mkdir()
        nested.write_text('{"version":1,"results":[]}')
        self.assertEqual(set(stages.result_paths([], results, True)), {flat, nested})
        with self.assertRaisesRegex(ValueError, "no result artifacts"):
            stages.result_paths([], self.root / "missing", True)

    def test_layout_change_invalidates_saved_recognition_not_rendered_png(self):
        entry = self.render_fixture()
        with patch.object(stages, "read_object", side_effect=self.read):
            first = stages.plan_images({entry["key"]: entry}, {}, {})
            changed = stages.plan_images({entry["key"]: entry}, {}, {}, overrides={entry["key"]: {
                "default": {"writing_mode": "vertical-rl"}, "pages": {"1": {"rotation": 90}}}})
        self.assertNotEqual(first["books"][0]["profile"], changed["books"][0]["profile"])
        self.assertNotEqual(stages.generation_for(first["books"][0]), stages.generation_for(changed["books"][0]))
        self.assertEqual(first["books"][0]["render_manifest"], changed["books"][0]["render_manifest"])


if __name__ == "__main__":
    unittest.main()
