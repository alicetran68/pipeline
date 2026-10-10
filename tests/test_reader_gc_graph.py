import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub import HfApi
import yaml

from scripts import publish_pdf_ocr_assets as publication
from scripts import reader_gc_graph as gc
from scripts import reader_bucket
from scripts import pdf_ocr_stages as stages


ASSETS, PDF, INPUT = gc.BUCKETS
SIDECAR = "reader-index/reader_assets.json.gz"


class MemoryStore:
    def __init__(self):
        self.objects = {bucket: {} for bucket in gc.BUCKETS}
        self.put(ASSETS, SIDECAR, {"v": 1, "f": {}})

    def put(self, bucket, path, payload):
        raw = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        self.objects[bucket][path] = gzip.compress(raw) if path.endswith(".gz") else raw

    def read_bytes(self, bucket, path):
        return self.objects[bucket][path]

    def list_files(self, bucket, prefixes):
        return {path for path in self.objects[bucket] if path.startswith(prefixes)}

    def put_json(self, bucket, path, payload):
        self.put(bucket, path, payload)

    def delete(self, bucket, paths):
        for path in paths:
            self.objects[bucket].pop(path, None)


class ReaderGcGraphTests(unittest.TestCase):
    def test_v3_automation_and_model_proposals_keep_pinned_inputs(self):
        store = MemoryStore()
        primary = "derived/migration-test/" + "a" * 32 + "/document.pdf"
        source = "objects/aa/source/ocr/page-000001.json.gz"
        store.put(PDF, primary, b"pdf")
        store.put(PDF, source, b"raw text")
        def ref(path):
            return {"bucket": PDF, "path": path, "sha256": "a" * 64, "bytes": 1, "role": "provenance"}
        store.put(ASSETS, "reader-index/v3/automation.json", {"version": 1, "kind": "reader-v3-automation",
            "tasks": {"task": {"spec": {"primary": ref(primary)}}}, "days": {}})
        store.put(ASSETS, "reader-index/v3/corrections/state.json", {"version": 1, "kind": "reader-v3-correction-state",
            "tasks": {"task": {"text_layer": ref(source)}}, "days": {}})
        store.put(ASSETS, "reader-index/v3/corrections/proposals/task/proposal.json", {
            "version": 1, "kind": "pdf-text-correction", "page": 1,
            "base_generation": "a" * 64, "raw_sha256": "b" * 64, "page_identity": "c" * 64,
            "replacements": [], "evidence": {"primary": ref(primary)}})
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertTrue(all(not value["candidates"] for value in report["buckets"].values()))

    def test_v3_text_review_retains_cross_bucket_evidence_and_raw_provenance(self):
        from scripts import pdf_reading_v3 as v3, pdf_text_layer as text
        from tests.test_pdf_text_layer import raw_page
        store = MemoryStore()
        raw_path = "objects/aa/raw/ocr/page-000001.json.gz"
        raw = raw_page("Text", .1)
        raw_bytes = gzip.compress(json.dumps(raw).encode())
        raw_ref = {"bucket": PDF, "path": raw_path, "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                   "bytes": len(raw_bytes), "role": "provenance"}
        store.objects[PDF][raw_path] = raw_bytes
        layer = text.from_page(raw, "a" * 64, "en", raw_ref)
        evidence_path = "objects/aa/input/ocr-input/page-000001.png"
        evidence = b"lossless evidence"
        store.put(INPUT, evidence_path, evidence)
        page = {"p": 1, "i": evidence_path, "is": hashlib.sha256(evidence).hexdigest(), "ib": len(evidence)}
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            ref = v3.build_text_bundle([layer], [page], Path("objects/aa/" + "a" * 64 + "/" + "b" * 16), bundle)
            for path in bundle.rglob("*"):
                if path.is_file():
                    store.objects[PDF][path.relative_to(bundle).as_posix()] = path.read_bytes()
        store.put(ASSETS, "reader-index/text-generation.json", {"resources": [ref]})
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertTrue(all(not b["candidates"] for b in report["buckets"].values()))

    def test_pdf_range_only_reference_protects_png_jxl_and_native_text(self):
        store = MemoryStore()
        root = "objects/aa/book/render"
        descriptor = f"{root}/render-range-000001-000001.json"
        png, jxl, text = f"{root}/ocr-input/page-000001.png", f"{root}/pages/page-000001.jxl", f"{root}/ocr/page-000001.json.gz"
        store.put(ASSETS, "reader-index/pdf_render_progress.json", {"version": 1, "files": {
            "book": {"ranges": {"000001-000001": {"path": descriptor}}},
        }})
        store.put(PDF, descriptor, {"version": 1, "kind": "pdf-render-range", "path": "upstream.pdf", "pages": [
            {"p": 1, "i": png, "j": jxl, "o": text, "ob": 50},
        ]})
        for bucket, path in ((INPUT, png), (INPUT, jxl), (PDF, text)):
            store.put(bucket, path, b"payload")
        store.put(INPUT, "objects/unused/ocr-input/page-000001.png", b"orphan")
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertEqual(report["buckets"][INPUT]["candidates"], ["objects/unused/ocr-input/page-000001.png"])
        self.assertEqual(report["buckets"][PDF]["candidates"], [])

    def test_all_ebook_extensions_and_fallback_resources_are_kept(self):
        store = MemoryStore()
        for ext in ("epub", "mobi", "azw3", "fb2", "chm"):
            root = f"chapters/ebook/{ext}/hash/version"
            manifest = f"{root}/chapter-manifest.json"
            fallback = f"native/ebook/{ext}/hash/document.epub"
            store.put(ASSETS, f"chapters/ebook/{ext}/index.json", {"files": [
                {"manifest": manifest, "fallback": fallback, "bucket": ASSETS},
            ]})
            store.put(ASSETS, manifest, {"kind": "epub-chapters", "chapters": [
                {"path": "chapters/chapter-0001.xhtml"}], "search_index": {"path": "epub-search-index.json.gz"}})
            for path in ("chapters/chapter-0001.xhtml", "resources/chapter-0001/image.jpg",
                         "resources/chapter-0001/font.woff2", "epub-search-index.json.gz"):
                store.put(ASSETS, f"{root}/{path}", b"resource")
            store.put(ASSETS, fallback, b"epub")
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertEqual(report["buckets"][ASSETS]["candidates"], [])

    def test_static_pdf_web_office_text_spreadsheet_and_media_bundles(self):
        store = MemoryStore()
        for category, name in (("documents/pdf/pptx", "document.pdf"),
                               ("documents/web/mhtml", "document.html"),
                               ("documents/office/docx", "document.docx"),
                               ("documents/text/txt", "document.txt"),
                               ("documents/spreadsheet/xlsx", "document.html"),
                               ("media/audio/mp3", "audio.mp3"),
                               ("media/video/mp4", "video.mp4"),
                               ("media/swf/swf", "animation.swf")):
            root = f"{category}/hash"
            store.put(ASSETS, f"{category}/index.json", {"files": [{"object": f"{root}/{name}"}]})
            store.put(ASSETS, f"{root}/{name}", b"document")
            store.put(ASSETS, f"{root}/resources/cover.png", b"resource")
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertEqual(report["buckets"][ASSETS]["candidates"], [])

    def test_image_manifest_is_not_assumed_to_be_a_pdf_manifest(self):
        store = MemoryStore()
        root = "pages/image/tiff/hash"
        store.put(ASSETS, SIDECAR, {"v": 1, "f": {"image": {
            "s": 2, "p": f"{root}/page-manifest.json", "b": ASSETS}}})
        store.put(ASSETS, f"{root}/page-manifest.json", {"kind": "image-page-stream", "page_count": 1,
            "pages": [{"path": "pages/page-000001.webp"}]})
        store.put(ASSETS, f"{root}/pages/page-000001.webp", b"image")
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])

    def test_sidecar_associated_pdf_and_ocr_have_separate_buckets(self):
        store = MemoryStore()
        manifest = "objects/aa/book/stream/page-manifest.json"
        pdf = "documents/pdf/ppt/hash/document.pdf"
        ocr = "objects/aa/book/text/ocr-manifest.json"
        store.put(ASSETS, SIDECAR, {"v": 1, "f": {"book": {
            "p": manifest, "b": PDF, "pd": pdf, "pdb": ASSETS, "o": ocr, "ob": PDF}}})
        store.put(PDF, manifest, {"kind": "pdf-pages", "page_count": 1})
        store.put(PDF, "objects/aa/book/stream/pages/page-000001.webp", b"image")
        store.put(ASSETS, pdf, b"pdf")
        store.put(PDF, ocr, {"kind": "pdf-ocr", "book_text": {"path": "objects/aa/book/text/ocr/book-text.json.gz"}})
        store.put(PDF, "objects/aa/book/text/ocr/book-text.json.gz", b"text")
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertTrue(all(not data["candidates"] for data in report["buckets"].values()))

    def test_derived_pdf_shard_checkpoint_and_pending_review_are_roots(self):
        store = MemoryStore()
        pdf = "derived/weekly-djvu/hash/document.pdf"
        evidence = "objects/aa/book/render/ocr-input/page-000001.png"
        store.put(PDF, "reader-index/derived_pdf_manifest.json", {"files": [{"new_path": pdf, "bucket": PDF}]})
        store.put(PDF, pdf, b"pdf")
        store.put(ASSETS, "reader-index/review_queue.json", {"resources": [{"path": evidence, "bucket": INPUT}]})
        store.put(INPUT, evidence, b"evidence")
        store.put(ASSETS, "pages/image/index.json", {"files": []})
        store.put(ASSETS, "pages/image/index-00.json", {"files": [{"object": "pages/image/png/partial/image.webp"}]})
        store.put(ASSETS, "pages/image/png/partial/image.webp", b"checkpoint")
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertTrue(all(not data["candidates"] for data in report["buckets"].values()))

    def test_missing_or_invalid_manifest_blocks_all_candidates(self):
        for raw in (None, b"invalid JSON", b'{"kind":"future-format"}'):
            with self.subTest(raw=raw):
                store = MemoryStore()
                path = "objects/book/render-manifest.json"
                store.put(ASSETS, SIDECAR, {"v": 1, "f": {"book": {"p": path, "b": PDF}}})
                if raw is not None:
                    store.put(PDF, path, raw)
                store.put(INPUT, "objects/unclaimed/input.png", b"input")
                report = gc.ReferenceGraph(store).build()
                self.assertFalse(report["graph_complete"])
                self.assertEqual(report["buckets"][INPUT]["candidates"], [])

    def test_permission_failure_or_skipped_input_is_incomplete(self):
        store = MemoryStore()
        original = store.list_files
        def denied(bucket, prefixes):
            if bucket == INPUT:
                raise PermissionError("denied")
            return original(bucket, prefixes)
        with patch.object(store, "list_files", side_effect=denied):
            self.assertFalse(gc.ReferenceGraph(store).build()["graph_complete"])
        self.assertFalse(gc.ReferenceGraph(store, include_input=False).build()["graph_complete"])

    def test_root_change_and_new_root_during_scan_are_detected(self):
        store = MemoryStore()
        original = store.read_bytes
        reads = 0
        def racing_read(bucket, path):
            nonlocal reads
            if path == SIDECAR:
                reads += 1
                if reads == 2:
                    return gzip.compress(b'{"v":1,"f":{"new":{"s":4}}}')
            return original(bucket, path)
        with patch.object(store, "read_bytes", side_effect=racing_read):
            report = gc.ReferenceGraph(store).build()
        self.assertTrue(any("root changed" in value for value in report["blockers"]))
        original_list = store.list_files
        pdf_lists = 0
        def racing_list(bucket, prefixes):
            nonlocal pdf_lists
            if bucket == PDF:
                pdf_lists += 1
                if pdf_lists == 2:
                    store.put(PDF, "reader-index/new.json", {"files": {}})
            return original_list(bucket, prefixes)
        with patch.object(store, "list_files", side_effect=racing_list):
            report = gc.ReferenceGraph(store).build()
        self.assertTrue(any("root inventory changed" in value for value in report["blockers"]))

    def test_unknown_formats_are_reported_and_never_deleted(self):
        store = MemoryStore()
        store.put(ASSETS, "new-format/content.bin", b"data")
        store.put(PDF, "objects/orphan/page.svg", b"svg")
        graph = gc.ReferenceGraph(store)
        report = graph.build()
        self.assertEqual(report["buckets"][ASSETS]["unmanaged"], ["new-format/content.bin"])
        self.assertEqual(report["buckets"][PDF]["formats"][".svg"], 1)
        self.assertFalse(report["deletion_enabled"])

    def test_ambiguous_unqualified_paths_preserve_both_buckets(self):
        store = MemoryStore()
        path = "objects/shared/document.pdf"
        store.put(ASSETS, "reader-index/manifest.json", {"files": {"book": {"path": path}}})
        store.put(ASSETS, path, b"pdf1")
        store.put(PDF, path, b"pdf2")
        report = gc.ReferenceGraph(store).build()
        self.assertEqual(report["ambiguous_paths"], [path])
        self.assertTrue(all(not data["candidates"] for data in report["buckets"].values()))

    def test_released_processing_root_does_not_keep_objects_but_uploaded_does(self):
        store = MemoryStore()
        root = "objects/aa/book/render"
        path = root + "/pages/page-000001.webp"
        store.put(PDF, path, b"page")
        store.put(ASSETS, "reader-index/processing/released.json", {
            "kind": "reader-processing-roots", "status": "released", "generation": "generation", "resources": [
                {"bucket": PDF, "root": root}],
        })
        store.put(ASSETS, "reader-index/reader_catalog.json", {
            "version": 1, "kind": "reader-catalog", "current": {"generation": "generation"},
            "generations": {"generation": {"generation": "generation", "status": "current", "references": []}},
        })
        report = gc.ReferenceGraph(store).build()
        self.assertEqual(report["buckets"][PDF]["candidates"], [path])
        store.put(ASSETS, "reader-index/processing/uploaded.json", {
            "kind": "reader-processing-roots", "status": "uploaded", "resources": [
                {"bucket": PDF, "root": root}],
        })
        report = gc.ReferenceGraph(store).build()
        self.assertEqual(report["buckets"][PDF]["candidates"], [])

    def test_gc_observation_requires_two_pass_grace_and_never_deletes(self):
        store = MemoryStore()
        orphan = "objects/orphan/document.pdf"
        store.put(PDF, orphan, b"orphan")
        report = gc.ReferenceGraph(store).build()
        state, first = gc.plan_retention({"version": 1, "kind": "reader-gc-state", "orphans": {}}, report, 14, 0)
        self.assertEqual((first["marked"], first["past_grace"], first["deletion_authorized"]), (1, 0, False))
        state["orphans"][f"{PDF}:{orphan}"]["first_seen"] = (date.today() - timedelta(days=15)).isoformat()
        report = gc.ReferenceGraph(store).build()
        _state, second = gc.plan_retention(state, report, 14, 0)
        self.assertEqual((second["past_grace"], second["deletion_authorized"]), (1, False))
        self.assertIn(orphan, store.objects[PDF])

    def test_gc_observation_aborts_when_a_root_is_unreadable(self):
        store = MemoryStore()
        store.put(PDF, "reader-index/pdf_ocr_manifest.json", b"bad")
        report = gc.ReferenceGraph(store).build()
        with self.assertRaises(RuntimeError):
            gc.plan_retention({"version": 1, "kind": "reader-gc-state", "orphans": {}}, report, 14, 0)

    def test_released_record_without_generation_keeps_inputs_and_blocks_graph(self):
        store = MemoryStore()
        root = "objects/aa/book/render"
        store.put(INPUT, root + "/ocr-input/page-000001.png", b"input")
        store.put(ASSETS, "reader-index/processing/released.json", {
            "kind": "reader-processing-roots", "status": "released", "generation": "missing",
            "resources": [{"bucket": INPUT, "root": root}],
        })
        report = gc.ReferenceGraph(store).build()
        self.assertFalse(report["graph_complete"])
        self.assertEqual(report["buckets"][INPUT]["unreferenced"], [])

    def test_scheduled_gc_is_artifact_only_without_blocking_writers(self):
        root = Path(__file__).resolve().parents[1]
        workflow = yaml.safe_load((root / ".github/workflows/reader-gc.yml").read_text())
        self.assertIn("reader-gc-report", workflow["concurrency"]["group"])
        self.assertIn("reader-sidecar", workflow["concurrency"]["group"])
        self.assertIn("record_observations", workflow["concurrency"]["group"])
        commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["gc"]["steps"])
        self.assertIn("python scripts/reader_gc_graph.py", commands)
        self.assertNotIn("--apply", commands)
        self.assertIn("--record-observations", commands)
        artifacts = [step for step in workflow["jobs"]["gc"]["steps"] if "upload-artifact" in step.get("uses", "")]
        self.assertEqual(artifacts[0]["if"], "always()")

    def test_upload_protection_precedes_objects_and_failure_retains_roots(self):
        store = MemoryStore()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "objects/aa/book/render"
            root.mkdir(parents=True)
            (root / "ocr-input").mkdir()
            events = []
            def protect(path, record, token):
                events.append("protect:" + record["status"])
                store.put(ASSETS, path, record)
            def upload(*args, **kwargs):
                events.append("upload")
                raise ValueError("failed transfer")
            with patch.object(stages, "HfApi") as api, patch.object(stages, "publish_json", side_effect=protect):
                api.return_value.sync_bucket.side_effect = upload
                with self.assertRaises(ValueError):
                    stages.upload_objects(Path(directory))
            self.assertEqual(events, ["protect:processing", "upload"])
            store.put(INPUT, "objects/aa/book/render/ocr-input/page-000001.png", b"partial")
            report = gc.ReferenceGraph(store).build()
            self.assertTrue(report["graph_complete"], report["blockers"])
            self.assertEqual(report["buckets"][INPUT]["candidates"], [])

    def test_partial_worker_handoff_keeps_progress_owned_objects(self):
        store = MemoryStore()
        root = "objects/aa/book/render"
        path = "reader-index/processing/worker.json"
        descriptor = root + "/render-range-000001-000001.json"
        result = {"key": "book", "status": "range", "descriptor": {"path": descriptor},
                  "processing_roots": [path]}
        store.put(ASSETS, path, {"kind": "reader-processing-roots", "status": "uploaded",
                                "resources": [{"bucket": PDF, "root": root}]})
        store.put(ASSETS, "reader-index/pdf_render_progress.json", {
            "files": {"book": {"ranges": {"1-1": {"path": descriptor}}}}})
        store.put(PDF, descriptor, {"kind": "pdf-render-range", "pages": []})
        catalog, generation = gc.reader_lifecycle.catalog_generation({"v": 1, "f": {}})
        store.put(ASSETS, "reader-index/reader_catalog.json", catalog)
        with patch.object(stages.publication, "load_sidecar", return_value={"v": 1, "f": {}}), \
                patch.object(stages, "publish_catalog", return_value=generation), \
                patch.object(stages, "read_bucket_json", side_effect=lambda p, token: json.loads(store.read_bytes(ASSETS, p))), \
                patch.object(stages, "publish_json", side_effect=lambda p, value, token: store.put(ASSETS, p, value)):
            stages.handoff_worker_results(HfApi(), "unused", [result])
        self.assertEqual(json.loads(store.read_bytes(ASSETS, path))["status"], "released")
        report = gc.ReferenceGraph(store).build()
        self.assertTrue(report["graph_complete"], report["blockers"])
        self.assertEqual(report["buckets"][PDF]["candidates"], [])

    def test_foreign_processing_root_cannot_be_released(self):
        with patch.object(publication, "read_bucket_json", return_value={
                "kind": "reader-processing-roots", "status": "uploaded",
                "resources": [{"bucket": PDF, "root": "objects/foreign/book/profile"}]}), \
                patch.object(publication, "publish_json") as write:
            with self.assertRaises(ValueError):
                publication.release_processing_roots([{
                    "ocr_manifest": "objects/own/book/profile/ocr-manifest.json",
                    "processing_roots": ["reader-index/processing/foreign.json"]}], "generation", None)
            write.assert_not_called()


class CanonicalOcrPublicationTests(unittest.TestCase):
    def test_ready_publication_verifies_manifest_text_checksums_and_page_order(self):
        root = "objects/aa/" + "a" * 64 + "/" + "b" * 16
        text_path = root + "/ocr/book-text.json.gz"
        text = gzip.compress(json.dumps({"kind": "pdf-book-text", "version": 2,
            "complete": True, "page_count": 2, "pages": [{"page": 1}, {"page": 2}]}).encode())
        manifest = json.dumps({"kind": "pdf-ocr", "complete": True, "page_count": 2,
            "source_sha256": "a" * 64, "pages": [{"p": 1}, {"p": 2}], "book_text": {
                "path": text_path, "bytes": len(text), "sha256": hashlib.sha256(text).hexdigest()}}).encode()
        result = {"ocr_manifest": root + "/ocr-manifest.json", "page_count": 2,
            "source_sha256": "a" * 64, "ocr_manifest_bytes": len(manifest),
            "ocr_manifest_sha256": hashlib.sha256(manifest).hexdigest()}
        with patch.object(publication, "read_bucket_bytes", side_effect=[manifest, text]) as read:
            publication.verify_ready_objects(result)
        self.assertTrue(all(call.kwargs["bucket"] == PDF for call in read.call_args_list))
        malformed = json.loads(manifest)
        malformed["pages"] = [{"p": 2}, {"p": 1}]
        raw = json.dumps(malformed).encode()
        with patch.object(publication, "read_bucket_bytes", return_value=raw):
            with self.assertRaises(ValueError):
                publication.verify_ready_objects({**result, "ocr_manifest_bytes": len(raw),
                    "ocr_manifest_sha256": hashlib.sha256(raw).hexdigest()})

    def test_catalog_snapshot_precedes_pointer_and_does_not_claim_consumer_acceptance(self):
        sidecar = {"v": 1, "f": {"book": {"p": "objects/book/document.pdf", "b": PDF}}}
        writes = []
        with patch.object(reader_bucket, "read_json", side_effect=FileNotFoundError), \
                patch.object(reader_bucket, "publish_indexes", side_effect=lambda data, token: writes.append(("snapshot", data))), \
                patch.object(reader_bucket, "publish_json", side_effect=lambda path, data, token: writes.append(("catalog", data))):
            generation = reader_bucket.publish_catalog(sidecar)
        self.assertEqual([kind for kind, data in writes], ["snapshot", "catalog"])
        catalog = writes[1][1]
        entry = catalog["generations"][generation]
        self.assertEqual(entry["acks"], {"hf": False, "pages": False})
        compressed = writes[0][1][entry["snapshot"]["path"]]
        self.assertEqual(json.loads(gzip.decompress(compressed)), sidecar)
        self.assertEqual(entry["snapshot"]["bytes"], len(compressed))

    def test_acknowledgment_rejects_wrong_generation_or_sidecar_identity(self):
        catalog, generation = gc.reader_lifecycle.catalog_generation({"v": 1, "f": {}})
        with patch.object(reader_bucket, "read_json", return_value=catalog), \
                patch.object(reader_bucket, "publish_json") as write:
            for expected, digest in (("stale", catalog["generations"][generation]["sidecar_sha256"]),
                                     (generation, "wrong")):
                with self.assertRaises(ValueError):
                    reader_bucket.acknowledge_catalog("pages", expected, digest)
            write.assert_not_called()

    def test_real_api_publishes_only_canonical_bucket_and_repairs_sidecar(self):
        api = HfApi()
        result = {"key": "book", "status": "ready", "profile": "test", "source_sha256": "a" * 64,
                  "classification": "scan", "ocr_manifest": "objects/text/ocr-manifest.json"}
        existing = {"version": 1, "files": {"book": result}}
        with patch.object(publication, "read_bucket_json", return_value=existing), \
                patch.object(publication, "read_bucket_bytes", return_value=publication.encode_sidecar({"v": 1, "f": {}})), \
                patch.object(publication, "verify_ready_objects"), \
                patch.object(publication, "publish_indexes") as upload, \
                patch.object(publication, "publish_catalog", return_value="gen"), \
                patch.object(api, "repo_info") as repo_info, patch.object(api, "create_commit") as commit:
            publication.publish(api, "unused", [result])
        repo_info.assert_not_called()
        commit.assert_not_called()
        payloads = upload.call_args.args[0]
        self.assertEqual(set(payloads), {reader_bucket.INDEX_FILES["ocr"], reader_bucket.INDEX_FILES["sidecar"]})
        sidecar = json.loads(gzip.decompress(payloads[reader_bucket.INDEX_FILES["sidecar"]]))
        self.assertEqual(sidecar["f"]["book"]["o"], result["ocr_manifest"])

    def test_unreadable_ready_manifest_blocks_registry_publication(self):
        with patch.object(publication, "read_bucket_bytes", side_effect=FileNotFoundError), \
                patch.object(publication, "publish_indexes") as write:
            with self.assertRaises(FileNotFoundError):
                publication.publish(HfApi(), "unused", [{
                    "status": "ready", "ocr_manifest": "objects/aa/" + "a" * 64 + "/" + "b" * 16 + "/ocr-manifest.json"}])
            write.assert_not_called()

    def test_corrupt_or_unavailable_bucket_never_falls_back_to_dataset(self):
        api = HfApi()
        for error in (PermissionError("denied"), ValueError("corrupt"), OSError("network")):
            with self.subTest(error=error), patch.object(publication, "read_bucket_json", side_effect=error), \
                    patch.object(api, "hf_hub_download") as download:
                with self.assertRaises(type(error)):
                    publication.load_remote(api, "unused", publication.OCR_MANIFEST_NAME, {})
                download.assert_not_called()
        with patch.object(publication, "read_bucket_bytes", return_value=b"not gzip"), \
                patch.object(api, "hf_hub_download") as download:
            with self.assertRaises(OSError):
                publication.load_sidecar(api, "unused")
            download.assert_not_called()

    def test_staged_registry_does_not_turn_corruption_into_empty_state(self):
        api = HfApi()
        with patch.object(stages, "read_bucket_json", return_value={"version": 1, "files": []}):
            with self.assertRaises(ValueError):
                stages.load_registry(api, "unused", stages.RENDER_REGISTRY)
        with patch.object(stages, "read_bucket_json", side_effect=OSError("network")):
            with self.assertRaises(OSError):
                stages.load_registry(api, "unused", stages.RENDER_REGISTRY)

    def test_index_batch_uses_only_current_bucket(self):
        observed = []
        def upload(bucket, add, token):
            from pathlib import Path
            observed.append((bucket, {remote: Path(local).read_bytes() for local, remote in add}))
        with patch.object(reader_bucket, "batch_bucket_files", side_effect=upload):
            reader_bucket.publish_indexes({"reader-index/a.json": b"{}", "reader-index/b.json.gz": b"gzip"})
        self.assertEqual(observed, [(ASSETS, {"reader-index/a.json": b"{}", "reader-index/b.json.gz": b"gzip"})])


if __name__ == "__main__":
    unittest.main()
