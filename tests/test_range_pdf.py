from pathlib import Path
import shutil
import tempfile
import unittest
import copy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject, NameObject,
                          NullObject, NumberObject, TextStringObject)

from scripts import pdf_range, pdf_range_assets, build_reader_assets_index, pdf_range_state


class RangePdfTests(unittest.TestCase):
    def test_source_fingerprint_and_engine_identity_control_incremental_queue(self):
        item = {"key": "book", "input_token": "sha256:aaa", "input_profile": "office-v1"}
        files, pending = pdf_range_assets.plan({"book": item}, {"files": {}}, "qpdf-test", 10)
        self.assertEqual(len(pending), 1)
        files["book"]["status"] = "no-gain"
        self.assertFalse(pdf_range_assets.plan({"book": {**item, "input_revision": "new-repo-commit"}},
                                              {"files": files}, "qpdf-test", 10)[1])
        self.assertTrue(pdf_range_assets.plan({"book": {**item, "input_token": "sha256:bbb"}},
                                             {"files": files}, "qpdf-test", 10)[1])
        self.assertTrue(pdf_range_assets.plan({"book": item}, {"files": files}, "new-qpdf", 10)[1])

    def test_pending_batches_make_progress_and_failed_inputs_require_retry(self):
        items = {str(i): {"key": str(i), "input_token": str(i), "input_profile": "upstream"} for i in range(3)}
        files, pending = pdf_range_assets.plan(items, {"files": {}}, "qpdf-test", 1)
        files[pending[0]["key"]]["status"] = "failed"
        _, following = pdf_range_assets.plan(items, {"files": files}, "qpdf-test", 1)
        self.assertNotEqual(following[0]["key"], pending[0]["key"])
        _, retries = pdf_range_assets.plan(items, {"files": files}, "qpdf-test", 3, retry_failed=True)
        self.assertEqual(len(retries), 3)

    def test_blocked_retry_is_scoped_and_bypasses_unsupported_cache(self):
        items = {status: {"key": status, "input_token": status, "input_profile": "upstream"}
                 for status in ("failed", "unsupported", "optimized", "unchanged", "pending")}
        previous = {key: {**item, "identity": pdf_range_assets.identity(item, "qpdf"), "status": key}
                    for key, item in items.items()}
        for version in ("qpdf", "upgraded-qpdf"):
            files, pending = pdf_range_assets.plan(items, {"files": previous}, version, 10, retry_blocked=True)
            self.assertEqual({row["key"] for row in pending}, {"failed", "unsupported"})
            self.assertEqual(files["optimized"], previous["optimized"])
            for row in pending:
                self.assertEqual(row["_previous_identity"], previous[row["key"]]["identity"])

    def test_stale_blocked_retry_advances_past_terminal_results(self):
        items = {key: {"key": key, "input_token": key, "input_profile": "upstream"}
                 for key in ("old-failed", "old-unsupported", "current-failed", "current-unsupported", "pending", "optimized")}
        previous = {key: {**item, "identity": pdf_range_assets.identity(item, "old" if key.startswith("old-") else "current"),
                          "status": key.split("-")[-1]}
                    for key, item in items.items()}
        _, batch = pdf_range_assets.plan(items, {"files": previous}, "current", 1, retry_stale_blocked=True)
        self.assertEqual(len(batch), 1)
        first = batch[0]
        previous[first["key"]] = {**first, "status": "unsupported"}
        _, following = pdf_range_assets.plan(items, {"files": previous}, "current", 1, retry_stale_blocked=True)
        self.assertEqual(len(following), 1)
        self.assertNotEqual(following[0]["key"], first["key"])
        previous[following[0]["key"]] = {**following[0], "status": "failed"}
        self.assertFalse(pdf_range_assets.plan(items, {"files": previous}, "current", 10, retry_stale_blocked=True)[1])
        items["current-failed"]["input_token"] = "changed-content"
        _, changed = pdf_range_assets.plan(items, {"files": previous}, "current", 10, retry_stale_blocked=True)
        self.assertEqual([row["key"] for row in changed], ["current-failed"])

    def test_policy_upgrade_shares_capacity_with_never_assessed_inputs(self):
        items = {key: {"key": key, "input_token": key, "input_profile": "upstream", "repo": "repo"}
                 for key in ("a-old-1", "a-old-2", "z-new-1", "z-new-2")}
        state = {"files": {key: {**row, "status": "unchanged" if key.startswith("a") else "pending",
                                  "identity": pdf_range_assets.identity(row, "old-tool")}
                           for key, row in items.items()}}
        _, batch = pdf_range_assets.plan(items, state, "new-tool", 2)
        self.assertEqual([row["key"] for row in batch], ["z-new-1", "a-old-1"])
        for row in batch:
            state["files"][row["key"]] = {**row, "status": "unchanged"}
        _, following = pdf_range_assets.plan(items, state, "new-tool", 2)
        self.assertEqual([row["key"] for row in following], ["z-new-2", "a-old-2"])

    def test_isolated_assessment_reports_unsupported_without_starting_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=400, height=600)
            writer._root_object[NameObject("/UnknownCatalog")] = NumberObject(1)
            writer.write(source)
            with source.open("ab") as output:
                output.write(b" " * pdf_range.MIN_BYTES)
            report, chosen = pdf_range_assets.assess_isolated(source, work, work)
            self.assertEqual(report["status"], "unsupported")
            self.assertIn("/UnknownCatalog", report["reason"])
            self.assertIsNone(chosen)

    def test_isolated_assessment_kills_a_stuck_process(self):
        real_popen = subprocess.Popen
        children = []

        def stuck(*args, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(child)
            return child

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pdf_range_assets.subprocess, "Popen", side_effect=stuck):
            work = Path(directory)
            report, chosen = pdf_range_assets.assess_isolated(work / "source.pdf", work, work, timeout=.1)
            self.assertEqual(report["reason"], "assessment-total-timeout")
            self.assertIsNone(chosen)
            self.assertIsNotNone(children[0].poll())

    def test_retry_does_not_keep_old_candidate_diagnostics_or_output_paths(self):
        item = {"key": "book", "source_bytes": 10, "status": "failed", "path": "old.pdf",
                "candidates": {"objects": {"error": "old error"}}, "sha256": "old-hash"}
        result = pdf_range_assets.process(item, None, None, None)
        self.assertEqual(result, {"key": "book", "source_bytes": 10,
                                  "status": "unchanged", "reason": "below-4-mib"})

    def test_failure_details_preserve_actionable_qpdf_output(self):
        error = subprocess.CalledProcessError(2, ["qpdf"], stderr=b"page 49: errors while decoding content stream")
        detail = pdf_range.failure_details(error)
        self.assertEqual(detail["error_category"], "qpdf-error")
        self.assertIn("page 49", detail["error"])
        self.assertEqual(pdf_range.failure_details(ValueError("document content or resource signature differs"))[
            "error_category"], "content-mismatch")

    def test_small_files_do_not_starve_the_assessment_budget(self):
        items = {str(i): {"key": str(i), "input_token": str(i), "input_profile": "upstream",
                          "source_bytes": 1000} for i in range(20)}
        items["large"] = {"key": "large", "input_token": "large", "input_profile": "upstream",
                          "source_bytes": 80 * pdf_range.MI}
        files, pending = pdf_range_assets.plan(items, {"files": {}}, "qpdf", 1)
        self.assertEqual([row["key"] for row in pending], ["large"])
        self.assertEqual(files["0"]["status"], "unchanged")

    def test_generated_and_original_repositories_share_backfill_slots(self):
        items = {f"A/{i}": {"key": f"A/{i}", "repo": "A", "source_kind": "upstream",
                             "input_token": f"a{i}", "input_profile": "upstream"} for i in range(10)}
        items["Z/book"] = {"key": "Z/book", "repo": "Z", "source_kind": "generated",
                           "input_token": "generated", "input_profile": "djvu"}
        files, pending = pdf_range_assets.plan(items, {"files": {}}, "v", 2)
        self.assertEqual({row["source_kind"] for row in pending}, {"upstream", "generated"})
        files["Z/book"]["status"] = "unchanged"
        _, next_batch = pdf_range_assets.plan(items, {"files": files}, "v", 2)
        self.assertEqual(len(next_batch), 2)
        self.assertTrue(all(row["source_kind"] == "upstream" for row in next_batch))

    def test_tool_upgrade_keeps_same_content_route_until_reassessment(self):
        item = {"key": "book", "input_token": "sha256:aaa", "input_profile": "upstream"}
        previous = {**item, "identity": pdf_range_assets.identity(item, "old"), "status": "optimized",
                    "path": "objects/aa/hash/profile/document.pdf"}
        files, pending = pdf_range_assets.plan({"book": item}, {"files": {"book": previous}}, "new", 10)
        self.assertEqual(files["book"]["status"], "optimized")
        self.assertEqual(len(pending), 1)
        files, _ = pdf_range_assets.plan({"book": {**item, "input_token": "sha256:bbb"}},
                                         {"files": {"book": previous}}, "new", 10)
        self.assertEqual(files["book"]["status"], "pending")
        self.assertNotIn("path", files["book"])

    def test_no_gain_results_are_shared_by_content_without_repository_commit_churn(self):
        item = {"key": "a", "input_token": "sha256:aaa", "input_profile": "upstream",
                "input_revision": "first", "input_repo": "repo", "input_path": "a.pdf"}
        cached = {**item, "identity": pdf_range_assets.identity(item, "v"), "status": "no-gain"}
        files, pending = pdf_range_assets.plan({"a": {**item, "input_revision": "next"},
                                                "b": {**item, "key": "b", "input_path": "b.pdf"}},
                                               {"files": {"a": cached}}, "v", 10)
        self.assertFalse(pending)
        self.assertEqual(files["a"]["input_revision"], "first")
        self.assertEqual(files["b"]["key"], "b")
        self.assertEqual(files["b"]["status"], "no-gain")

    def test_retired_range_mapping_never_overrides_reader_pdf(self):
        base = {"files": {"book": {"status": "ready", "reader_mode": "pdf", "path": "original.pdf",
                                   "sha256": "aaa", "profile": "repair-v1"}}}
        state = {"files": {"book": {"status": "optimized", "source_kind": "generated",
                                    "input_sha256": "aaa", "input_profile": "repair-v1",
                                    "path": "objects/aa/hash/range/document.pdf"}}}
        self.assertEqual(build_reader_assets_index.build_index(base, range_manifest=state)["f"]["book"]["p"],
                         "original.pdf")
        base["files"]["book"]["sha256"] = "bbb"
        self.assertEqual(build_reader_assets_index.build_index(base, range_manifest=state)["f"]["book"]["p"],
                         "original.pdf")
        self.assertEqual(build_reader_assets_index.build_index({"files": {}}, range_manifest=state)["f"], {})

    def test_large_gain_does_not_override_changed_pixels_or_more_requests(self):
        def report(amount, requests):
            metric = {"bytes": amount, "requests": requests}
            return {"renders": ["same"], "pages": 10, "outline_entries": 1,
                    "snapshots": {key: dict(metric) for key in ("startup", "idle", "jump", "final")}}
        before, after = report(80 * pdf_range.MI, 80), report(3 * pdf_range.MI, 3)
        self.assertTrue(pdf_range.improvement(before, after, 90 * pdf_range.MI, 89 * pdf_range.MI))
        for mutation in ("pixels", "requests", "background", "growth"):
            candidate = copy.deepcopy(after)
            if mutation == "pixels": candidate["renders"] = ["changed"]
            if mutation == "requests": candidate["snapshots"]["startup"]["requests"] = 81
            if mutation == "background": candidate["snapshots"]["idle"]["bytes"] += 1
            output_size = 100 * pdf_range.MI if mutation == "growth" else 89 * pdf_range.MI
            self.assertFalse(pdf_range.improvement(before, candidate, 90 * pdf_range.MI, output_size))

    def test_strong_gain_can_end_candidate_search_early(self):
        def report(amount, requests):
            metric = {"bytes": amount, "requests": requests}
            return {"renders": ["same"], "pages": 10, "outline_entries": 1,
                    "snapshots": {key: dict(metric) for key in ("startup", "idle", "jump", "final")}}
        before = report(80 * pdf_range.MI, 80)
        strong = report(pdf_range.MI - 1, 20)
        ordinary = report(60 * pdf_range.MI, 70)
        self.assertTrue(pdf_range.strong_improvement(before, strong, 90 * pdf_range.MI, 90 * pdf_range.MI))
        self.assertFalse(pdf_range.strong_improvement(before, ordinary, 90 * pdf_range.MI, 90 * pdf_range.MI))
        before["invalid_destinations"] = [[0, 2]]
        self.assertFalse(pdf_range.improvement(before, strong, 90 * pdf_range.MI, 90 * pdf_range.MI))
        strong["invalid_destinations"] = [[0, 2]]
        self.assertTrue(pdf_range.improvement(before, strong, 90 * pdf_range.MI, 90 * pdf_range.MI))

    def test_heavy_candidates_are_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(list(pdf_range.candidate_methods()), list(pdf_range.METHODS))
        with patch.dict("os.environ", {"PDF_RANGE_TRY_HEAVY": "0"}):
            self.assertEqual(list(pdf_range.candidate_methods()), ["objects"])

    def test_qpdf_exit_three_is_deferred_to_pdf_validation(self):
        completed = subprocess.CompletedProcess(["qpdf"], 3, b"", b"WARNING: repaired input")
        with patch.object(pdf_range.subprocess, "run", return_value=completed):
            self.assertIn("repaired input", pdf_range.run_qpdf(["qpdf"], timeout=1))
        failed = subprocess.CompletedProcess(["qpdf"], 2, b"", b"fatal")
        with patch.object(pdf_range.subprocess, "run", return_value=failed):
            with self.assertRaises(subprocess.CalledProcessError):
                pdf_range.run_qpdf(["qpdf"], timeout=1)

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_conversion_preserves_content_images_geometry_and_outline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.pdf", root / "document.pdf"
            writer = PdfWriter()
            image = DecodedStreamObject()
            image.set_data(bytes([255, 0, 0, 0, 255, 0]))
            image.update({NameObject("/Type"): NameObject("/XObject"),
                          NameObject("/Subtype"): NameObject("/Image"),
                          NameObject("/Width"): NumberObject(2),
                          NameObject("/Height"): NumberObject(1),
                          NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                          NameObject("/BitsPerComponent"): NumberObject(8)})
            image_ref = writer._add_object(image.flate_encode())
            for index in range(3):
                page = writer.add_blank_page(width=400 + index, height=600)
                page.cropbox.lower_left = (20, 30)
                page.cropbox.upper_right = (380, 580)
                page.rotate(180)
                page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"):
                    DictionaryObject({NameObject("/Im1"): image_ref})})
                content = DecodedStreamObject()
                content.set_data(b"q 200 0 0 100 20 30 cm /Im1 Do Q")
                page[NameObject("/Contents")] = writer._add_object(content.flate_encode())
            writer.add_metadata({"/Title": "Range fixture"})
            writer.add_outline_item("Third page", 2)
            writer.write(source)
            original = source.read_bytes()
            subprocess.run(["qpdf", "--object-streams=generate", "--stream-data=preserve", str(source), str(target)], check=True)
            self.assertEqual(pdf_range.content_signature(source), pdf_range.content_signature(target))
            self.assertEqual(source.read_bytes(), original)
            before, after = PdfReader(source), PdfReader(target)
            self.assertEqual(len(after.pages), 3)
            for a, b in zip(before.pages, after.pages):
                self.assertEqual(a.mediabox, b.mediabox)
                self.assertEqual(a.get_contents().get_data(), b.get_contents().get_data())
                self.assertEqual(a["/Contents"]._data, b["/Contents"]._data)
                self.assertEqual(a["/Resources"]["/XObject"]["/Im1"]._data,
                                 b["/Resources"]["/XObject"]["/Im1"]._data)
            self.assertEqual(after.metadata.title, before.metadata.title)
            self.assertEqual(after.outline[0].title, "Third page")
            self.assertEqual(after.get_destination_page_number(after.outline[0]), 2)
            self.assertTrue(after.xref_objStm, "page dictionaries should use object streams")

    def test_all_generated_pdf_formats_use_asset_fingerprint_and_original_path(self):
        records = [{"Repo": "VoiceOfML/Test", "File": "slides.ppt", "Path": "Test", "Extension": "ppt", "Size": 20}]
        # Use the common metadata path helper, as generated corpus shapes vary.
        from scripts import reader_assets
        path = reader_assets.relative_path(records[0])
        key = reader_assets.asset_key("VoiceOfML/Test", path)
        base = {"files": {key: {"status": "ready", "reader_mode": "pdf", "sha256": "abc", "bytes": 5000000,
                               "profile": "office-v1", "path": "objects/aa/bb/office/document.pdf"}}}
        class NoNetwork:
            def list_repo_tree(self, **kwargs):
                raise AssertionError("generated PDF does not need original-file download")
        items, _ = pdf_range_assets.discover(records, {}, base, {"files": {}}, NoNetwork(),
                                             pdf_range_state.empty_state(), "assets", "fixed")
        self.assertEqual(items[key]["input_token"], "sha256:abc")
        self.assertEqual(items[key]["input_path"], base["files"][key]["path"])
        self.assertEqual(items[key]["source_path"], path)

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_local_open_action_and_page_labels_survive_object_renumbering(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, changed = [Path(directory) / name for name in ("source.pdf", "target.pdf", "changed.pdf")]
            writer = PdfWriter()
            for _ in range(3):
                writer.add_blank_page(width=400, height=600)
            action = DictionaryObject({NameObject("/S"): NameObject("/GoTo"),
                NameObject("/D"): ArrayObject([writer.pages[2].indirect_reference, NameObject("/Fit")])})
            writer._root_object[NameObject("/OpenAction")] = writer._add_object(action)
            writer._root_object[NameObject("/PageLabels")] = DictionaryObject({NameObject("/Nums"):
                ArrayObject([NumberObject(0), DictionaryObject({NameObject("/S"): NameObject("/r")})])})
            writer.write(source)
            for options in (["--object-streams=generate"], ["--linearize"]):
                subprocess.run(["qpdf", *options, str(source), str(target)], check=True, capture_output=True)
                self.assertEqual(pdf_range.content_signature(source), pdf_range.content_signature(target))
            action[NameObject("/D")][0] = writer.pages[0].indirect_reference
            writer.write(changed)
            self.assertNotEqual(pdf_range.content_signature(source), pdf_range.content_signature(changed))
            action[NameObject("/S")] = NameObject("/JavaScript")
            writer.write(changed)
            with self.assertRaisesRegex(pdf_range.UnsupportedPDF, "non-local opening"):
                pdf_range.content_signature(changed)

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_xmp_compression_is_normalized_but_metadata_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, changed = [Path(directory) / name for name in ("source.pdf", "target.pdf", "changed.pdf")]
            writer = PdfWriter()
            writer.add_blank_page(width=400, height=600)
            stream = DecodedStreamObject()
            stream.set_data(b'<x:xmpmeta xmlns:x="adobe:ns:meta/">original</x:xmpmeta>')
            stream.update({NameObject("/Type"): NameObject("/Metadata"), NameObject("/Subtype"): NameObject("/XML")})
            writer._root_object[NameObject("/Metadata")] = writer._add_object(stream.flate_encode())
            writer.write(source)
            subprocess.run(["qpdf", "--object-streams=generate", "--stream-data=preserve", str(source), str(target)],
                           check=True, capture_output=True)
            self.assertEqual(pdf_range.content_signature(source), pdf_range.content_signature(target))
            stream.set_data(b'<x:xmpmeta xmlns:x="adobe:ns:meta/">changed</x:xmpmeta>')
            writer._root_object[NameObject("/Metadata")] = writer._add_object(stream)
            writer.write(changed)
            self.assertNotEqual(pdf_range.content_signature(source), pdf_range.content_signature(changed))

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_cyclic_tags_annotations_forms_and_named_destinations_remain_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, changed = [Path(directory) / name for name in ("source.pdf", "target.pdf", "changed.pdf")]
            writer = PdfWriter()
            for _ in range(3):
                writer.add_blank_page(width=400, height=600)
            tag_root = DictionaryObject({NameObject("/Type"): NameObject("/StructTreeRoot")})
            tag_ref = writer._add_object(tag_root)
            tag = DictionaryObject({NameObject("/Type"): NameObject("/StructElem"), NameObject("/S"): NameObject("/P"),
                NameObject("/P"): tag_ref, NameObject("/Pg"): writer.pages[0].indirect_reference,
                NameObject("/Alt"): TextStringObject("original tag")})
            tag_root[NameObject("/K")] = ArrayObject([writer._add_object(tag)])
            writer._root_object[NameObject("/StructTreeRoot")] = tag_ref
            annotation = DictionaryObject({NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Text"), NameObject("/Contents"): TextStringObject("original note"),
                NameObject("/P"): writer.pages[1].indirect_reference})
            writer.pages[1][NameObject("/Annots")] = ArrayObject([writer._add_object(annotation)])
            field = DictionaryObject({NameObject("/FT"): NameObject("/Tx"), NameObject("/T"): TextStringObject("name"),
                NameObject("/V"): TextStringObject("original value")})
            writer._root_object[NameObject("/AcroForm")] = DictionaryObject({NameObject("/Fields"):
                ArrayObject([writer._add_object(field)])})
            writer.add_named_destination("chapter", 2)
            writer._root_object[NameObject("/OpenAction")] = TextStringObject("chapter")
            writer.add_outline_item("third", 2)
            writer.write(source)
            original = pdf_range.content_signature(source)
            for options in pdf_range.METHODS.values():
                subprocess.run(["qpdf", *options, "--stream-data=preserve", str(source), str(target)],
                               check=True, capture_output=True)
                self.assertEqual(original, pdf_range.content_signature(target))
            for obj, key, value in ((tag, "/Alt", TextStringObject("changed tag")),
                                    (tag, "/Pg", writer.pages[2].indirect_reference),
                                    (annotation, "/Contents", TextStringObject("changed note")),
                                    (field, "/V", TextStringObject("changed value"))):
                old = obj.raw_get(key)
                obj[NameObject(key)] = value
                writer.write(changed)
                self.assertNotEqual(original, pdf_range.content_signature(changed), key)
                obj[NameObject(key)] = old
            destination = writer.get_named_dest_root()[1].get_object()["/D"]
            destination[0] = writer.pages[0].indirect_reference
            writer.write(changed)
            self.assertNotEqual(original, pdf_range.content_signature(changed))

    def test_signature_fields_and_active_content_are_still_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            for field in (DictionaryObject({NameObject("/FT"): NameObject("/Sig")}),
                          DictionaryObject({NameObject("/S"): NameObject("/JavaScript"),
                                            NameObject("/JS"): TextStringObject("app.alert('x')")})):
                writer = PdfWriter()
                page = writer.add_blank_page(width=400, height=600)
                page[NameObject("/Annots")] = ArrayObject([writer._add_object(field)])
                writer.write(source)
                with self.assertRaises(pdf_range.UnsupportedPDF):
                    pdf_range.content_signature(source)
                key = "/FT" if "/FT" in field else "/S"
                field[NameObject(key)] = writer._add_object(field[key])
                writer.write(source)
                with self.assertRaises(pdf_range.UnsupportedPDF):
                    pdf_range.content_signature(source)

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_attachments_preserve_payload_names_and_shared_filespecs(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, changed = [Path(directory) / name for name in ("source.pdf", "target.pdf", "changed.pdf")]
            writer = PdfWriter(); writer.add_blank_page(width=400, height=600)
            stream = DecodedStreamObject(); stream.set_data(b"original archive\x00\xff")
            stream[NameObject("/Type")] = NameObject("/EmbeddedFile")
            stream_ref = writer._add_object(stream)
            file = DictionaryObject({NameObject("/Type"): NameObject("/Filespec"),
                NameObject("/F"): TextStringObject("archive.zip"), NameObject("/UF"): TextStringObject("archive.zip"),
                NameObject("/EF"): DictionaryObject({NameObject("/F"): stream_ref, NameObject("/UF"): stream_ref})})
            file_ref = writer._add_object(file)
            leaf = DictionaryObject({NameObject("/Names"): ArrayObject([TextStringObject("archive.zip"), file_ref])})
            writer._root_object[NameObject("/Names")] = DictionaryObject({NameObject("/EmbeddedFiles"):
                DictionaryObject({NameObject("/Kids"): ArrayObject([writer._add_object(leaf)])})})
            writer.write(source)
            original = pdf_range.content_signature(source)
            for options in pdf_range.METHODS.values():
                subprocess.run(["qpdf", *options, "--stream-data=preserve", str(source), str(target)],check=True,capture_output=True)
                self.assertEqual(original, pdf_range.content_signature(target))
            for obj, key, value in [(file, "/UF", TextStringObject("changed.zip")),
                                     (leaf, "/Names", ArrayObject([]))]:
                old = obj.raw_get(key); obj[NameObject(key)] = value; writer.write(changed)
                self.assertNotEqual(original, pdf_range.content_signature(changed)); obj[NameObject(key)] = old
            stream.set_data(b"changed archive\x00\xff"); writer.write(changed)
            self.assertNotEqual(original, pdf_range.content_signature(changed))
            file[NameObject("/AA")] = DictionaryObject(); writer.write(changed)
            with self.assertRaises(pdf_range.UnsupportedPDF): pdf_range.content_signature(changed)

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_passive_catalog_and_orphan_page_destination_remain_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, changed = [Path(directory) / name for name in ("source.pdf", "target.pdf", "changed.pdf")]
            writer = PdfWriter(); page = writer.add_blank_page(width=400, height=600)
            orphan = DictionaryObject({NameObject("/Type"): NameObject("/Page"),
                NameObject("/MediaBox"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(200), NumberObject(300)])})
            orphan_ref = writer._add_object(orphan)
            writer._root_object[NameObject("/OpenAction")] = ArrayObject([page.indirect_reference, NameObject("/Fit")])
            spider = DictionaryObject({NameObject("/V"): NumberObject(1)})
            writer._root_object[NameObject("/SpiderInfo")] = spider
            color = ArrayObject([NameObject("/CalGray"), DictionaryObject({NameObject("/Gamma"): NumberObject(2)})])
            writer._root_object[NameObject("/DefaultGray")] = color
            thread = DictionaryObject(); thread_ref = writer._add_object(thread)
            bead = DictionaryObject({NameObject("/T"): thread_ref, NameObject("/P"): page.indirect_reference,
                NameObject("/R"): ArrayObject([NumberObject(i) for i in [0,0,200,300]])})
            bead_ref = writer._add_object(bead)
            bead[NameObject("/N")] = bead_ref; bead[NameObject("/V")] = bead_ref
            thread[NameObject("/F")] = bead_ref
            writer._root_object[NameObject("/Threads")] = ArrayObject([thread_ref])
            writer.write(source); original = pdf_range.content_signature(source)
            for options in pdf_range.METHODS.values():
                subprocess.run(["qpdf", *options, "--stream-data=preserve", str(source), str(target)],check=True,capture_output=True)
                self.assertEqual(original, pdf_range.content_signature(target))
            for obj, key, value in [(spider, "/V", NumberObject(2)), (bead, "/P", orphan_ref),
                                     (color[1], "/Gamma", NumberObject(3)),
                                     (writer._root_object, "/OpenAction", ArrayObject([orphan_ref, NameObject("/Fit")]))]:
                old=obj.raw_get(key);obj[NameObject(key)]=value;writer.write(changed)
                self.assertNotEqual(original,pdf_range.content_signature(changed));obj[NameObject(key)]=old
            writer._root_object[NameObject("/OpenAction")] = ArrayObject([orphan_ref, NameObject("/Fit")])
            writer.write(source); original=pdf_range.content_signature(source)
            subprocess.run(["qpdf", "--object-streams=generate", "--stream-data=preserve", str(source), str(target)],check=True,capture_output=True)
            self.assertEqual(original,pdf_range.content_signature(target))
            writer._root_object[NameObject("/OpenAction")] = ArrayObject([NullObject(), NameObject("/Fit")])
            writer.write(changed)
            with self.assertRaises(pdf_range.UnsupportedPDF): pdf_range.content_signature(changed)

    def test_passive_structure_upgrade_preserves_only_unchanged_completed_inputs(self):
        item = {"key":"book", "input_token":"same", "input_profile":"upstream", "source_bytes":10*pdf_range.MI}
        old = pdf_range_assets.identity(item,"qpdf","pdfjs-6.3.289-range-1m-scene-v2-policy-v3")
        for status in ["optimized","unchanged","no-gain","failed","unsupported"]:
            state={"files":{"book":{**item,"identity":old,"status":status}}}
            _,queue=pdf_range_assets.plan({"book":item},state,"qpdf",10)
            self.assertEqual(bool(queue), status in {"failed","unsupported"})
            if status in {"optimized","unchanged","no-gain"}:
                self.assertTrue(pdf_range_assets.plan({"book":{**item,"input_token":"changed"}},state,"qpdf",10)[1])
                self.assertTrue(pdf_range_assets.plan({"book":item},state,"new-qpdf",10)[1])
    def test_null_dictionary_entries_are_absent_but_array_slots_and_lengths_matter(self):
        with tempfile.TemporaryDirectory() as directory:
            source, changed = [Path(directory) / name for name in ("source.pdf", "changed.pdf")]
            writer = PdfWriter()
            writer.add_blank_page(width=400, height=600)
            writer._root_object[NameObject("/Metadata")] = NullObject()
            writer._root_object[NameObject("/OpenAction")] = NullObject()
            properties = DictionaryObject({NameObject("/Length"): NumberObject(4),
                NameObject("/Items"): ArrayObject([NullObject(), NumberObject(5)])})
            writer._root_object[NameObject("/PieceInfo")] = properties
            writer.write(source)
            original = pdf_range.content_signature(source)
            del writer._root_object["/Metadata"]
            del writer._root_object["/OpenAction"]
            writer.write(changed)
            self.assertEqual(original, pdf_range.content_signature(changed))
            properties[NameObject("/Length")] = NumberObject(3)
            writer.write(changed)
            self.assertNotEqual(original, pdf_range.content_signature(changed))
            properties[NameObject("/Length")] = NumberObject(4)
            properties["/Items"].pop(0)
            writer.write(changed)
            self.assertNotEqual(original, pdf_range.content_signature(changed))

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_passwordless_encryption_preserves_permissions_and_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = [Path(directory) / name for name in ("source.pdf", "target.pdf")]
            writer = PdfWriter()
            writer.add_blank_page(width=400, height=600)
            writer.encrypt("", "owner-secret", algorithm="AES-128")
            writer.write(source)
            original = pdf_range.content_signature(source)
            for options in pdf_range.METHODS.values():
                subprocess.run(["qpdf", *options, "--stream-data=preserve", str(source), str(target)],
                               check=True, capture_output=True)
                self.assertEqual(original, pdf_range.content_signature(target))
                self.assertTrue(PdfReader(target).is_encrypted)
            subprocess.run(["qpdf", "--decrypt", str(source), str(target)], check=True, capture_output=True)
            self.assertNotEqual(original, pdf_range.content_signature(target))
            writer.encrypt("reader-secret", "owner-secret", algorithm="AES-256")
            writer.write(source)
            with self.assertRaisesRegex(pdf_range.UnsupportedPDF, "password required"):
                pdf_range.content_signature(source)

    @unittest.skipUnless(shutil.which("qpdf"), "qpdf is required")
    def test_reconstruction_fallback_requires_original_graph_and_render_equivalence(self):
        def measurement(amount):
            return {"renders": ["same pixels and text"], "pages": 1, "outline_entries": 0,
                    "snapshots": {key: {"bytes": amount, "requests": 1}
                                  for key in ("startup", "idle", "jump", "final")}}

        real_qpdf, real_reconstruct = pdf_range.run_qpdf, pdf_range.reconstruct_pdf
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                source = work / "source.pdf"
                writer = PdfWriter()
                page = writer.add_blank_page(width=400, height=600)
                contents = DecodedStreamObject()
                contents.set_data(b" " * pdf_range.MIN_BYTES)
                page[NameObject("/Contents")] = writer._add_object(contents)
                writer.write(source)
                original_bytes = source.read_bytes()

                def qpdf(args, *, timeout):
                    if args[-2] == str(source):
                        raise subprocess.CalledProcessError(2, args, stderr=b"invalid object in page tree")
                    return real_qpdf(args, timeout=timeout)

                def reconstruct(source, target):
                    if not changed:
                        return real_reconstruct(source, target)
                    altered = PdfWriter()
                    altered.add_blank_page(width=401, height=600)
                    altered.write(target)

                with patch.dict("os.environ", {"PDF_RANGE_TRY_HEAVY": "0"}), \
                        patch("playwright.sync_api.sync_playwright"), \
                        patch.object(pdf_range, "run_qpdf", side_effect=qpdf), \
                        patch.object(pdf_range, "reconstruct_pdf", side_effect=reconstruct), \
                        patch.object(pdf_range, "benchmark", side_effect=[measurement(4 * pdf_range.MI),
                                     measurement(pdf_range.MI // 2)]) as benchmark:
                    report, chosen = pdf_range.assess(source, work, work)
                self.assertEqual(source.read_bytes(), original_bytes)
                if changed:
                    self.assertIsNone(chosen)
                    self.assertFalse(report["reconstruction"]["validated"])
                    self.assertEqual(report["reconstruction"]["error_category"], "content-mismatch")
                    self.assertEqual(benchmark.call_count, 1)
                else:
                    self.assertEqual(report["status"], "optimized")
                    self.assertTrue(report["reconstruction"]["validated"])
                    self.assertEqual(report["method"], "reconstructed-objects")
                    self.assertEqual(pdf_range.content_signature(source), pdf_range.content_signature(chosen))

    def test_upstream_inventory_cache_tracks_actual_file_fingerprint(self):
        from scripts import reader_assets
        row = {"Repo": "VoiceOfML/Test", "File": "scan.pdf", "Path": "Test", "Extension": "pdf", "Size": 10}
        path = reader_assets.relative_path(row)
        api = Mock()
        api.list_repo_tree.return_value = [SimpleNamespace(path=path, blob_id="blob", size=10,
                                                           lfs=SimpleNamespace(sha256="content"))]
        args = ([row], {"VoiceOfML/Test": "rev1"}, {"files": {}}, {"files": {}}, api)
        items, inventories = pdf_range_assets.discover(*args, pdf_range_state.empty_state(), "assets", "rev")
        self.assertEqual(next(iter(items.values()))["input_token"], "sha256:content")
        pdf_range_assets.discover(*args, {"inventories": inventories}, "assets", "rev")
        self.assertEqual(api.list_repo_tree.call_count, 1)
        pdf_range_assets.discover([row], {"VoiceOfML/Test": "rev2"}, {"files": {}}, {"files": {}},
                                  api, {"inventories": inventories}, "assets", "rev")
        self.assertEqual(api.list_repo_tree.call_count, 2)

    def test_regular_publication_does_not_publish_retired_optimized_routes(self):
        from scripts import publish_reader_assets, reader_assets
        import gzip
        import json
        api = Mock()
        api.repo_info.return_value = SimpleNamespace(sha="parent")
        base = {"version": 1, "files": {}}
        state = {"version": 1, "files": {"scan": {"status": "optimized", "source_kind": "upstream",
                                                   "path": "objects/aa/hash/range/document.pdf"}}}
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "bundle.json").write_text(json.dumps({"version": 1, "results": []}))
            with patch.object(publish_reader_assets, "remote_manifest", return_value=base), \
                 patch.object(publish_reader_assets, "remote_pdf_manifest", return_value=base), \
                 patch.object(publish_reader_assets, "remote_pdf_ocr_manifest", return_value={"version": 1, "files": {}}), \
                 patch.object(publish_reader_assets, "remote_state", return_value=state):
                publish_reader_assets.publish_bundle(api, "assets", bundle)
        operations = api.create_commit.call_args.kwargs["operations"]
        sidecar = next(op.path_or_fileobj for op in operations if op.path_in_repo == "reader_assets.json.gz")
        self.assertNotIn("scan", json.loads(gzip.decompress(sidecar))["f"])

    def test_new_range_state_is_not_a_reader_route(self):
        from scripts import build_reader_assets_index, pdf_range_state, shared
        state = {"version": 1, "artifact_bucket": shared.PDF_RANGE_BUCKET, "files": {
            "scan": {"status": "optimized", "path": "objects/aa/" + "a" * 64
                      + "/pdf-range-v1-document/document.pdf"}
        }}
        index = build_reader_assets_index.build_index({"files": {}}, range_manifest=state)
        self.assertNotIn("scan", index["f"])
        self.assertEqual(pdf_range_state.empty_state()["artifact_bucket"], shared.PDF_RANGE_BUCKET)


if __name__ == "__main__":
    unittest.main()
