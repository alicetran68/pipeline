import json
import requests
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub.errors import HfHubHTTPError

from scripts import pdf_health


class PdfHealthTests(unittest.TestCase):
    def test_source_enumeration_requires_pinned_revision(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            search = root / "search.json"
            revisions = root / "commits.json"
            search.write_text(json.dumps({"rp": ["r"], "fd": [[]],
                                          "rc": [[0, "book", "pdf", 0, 12, False]]}), encoding="utf-8")
            revisions.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing pinned revision"):
                pdf_health.records_from_sources(search, revisions)

    def test_planner_is_incremental_and_rechecks_changed_revision(self):
        records = [
            {"key": "r\0same.pdf", "source_revision": "one", "declared_bytes": 10},
            {"key": "r\0fixed.pdf", "source_revision": "two", "declared_bytes": 20},
        ]
        report = {"version": 1, "files": {
            "r\0same.pdf": {"source_revision": "one", "status": "healthy"},
            "r\0fixed.pdf": {"source_revision": "one", "status": "corrupt"},
        }}
        queue = pdf_health.plan(records, report)
        selected = [item for shard in queue["shards"] for item in shard["records"]]
        self.assertEqual([item["key"] for item in selected], ["r\0fixed.pdf"])
        self.assertEqual(queue["total_records"], 2)
        self.assertEqual(queue["pending_records"], 1)
        self.assertEqual(queue["shard_count"], 18)

    def test_planner_skips_unchanged_healthy_content_but_retries_transient_and_corrupt(self):
        records = [
            {"key": "r\0healthy.pdf", "source_revision": "two", "declared_bytes": 10},
            {"key": "r\0corrupt.pdf", "source_revision": "two", "declared_bytes": 20},
            {"key": "r\0transient.pdf", "source_revision": "one", "declared_bytes": 30},
        ]
        report = {"version": 1, "files": {
            "r\0healthy.pdf": {"source_revision": "one", "declared_bytes": 10, "status": "healthy"},
            "r\0corrupt.pdf": {"source_revision": "one", "declared_bytes": 20, "status": "corrupt"},
            "r\0transient.pdf": {"source_revision": "one", "declared_bytes": 30, "status": "download-failed"},
        }}
        self.assertEqual([item["key"] for item in pdf_health.pending_records(records, report)],
                         ["r\0corrupt.pdf", "r\0transient.pdf"])

    def test_conversion_ready_records_skip_audit_and_refresh_report(self):
        from scripts import pdf_assets
        ready = {"status": "ready", "strategy": "sampled-webp", "source_revision": "one",
                 "render_profile": pdf_assets.PDF_PROFILE,
                 "decision_profile": pdf_assets.PDF_DECISION_PROFILE}
        manifest = {"version": 1, "files": {
            "r\0proven.pdf": dict(ready),
            "r\0changed.pdf": {**ready, "source_revision": "old"},
            "r\0legacy.pdf": {**ready, "render_profile": "pdf-pages-v1"},
        }}
        records = [
            {"key": "r\0proven.pdf", "source_revision": "one", "declared_bytes": 10},
            {"key": "r\0changed.pdf", "source_revision": "one", "declared_bytes": 20},
            {"key": "r\0legacy.pdf", "source_revision": "one", "declared_bytes": 30},
            {"key": "r\0fresh.pdf", "source_revision": "one", "declared_bytes": 40},
        ]
        report = {"version": 1, "files": {
            "r\0proven.pdf": {"source_revision": "one", "status": "tool-error", "reason": "qpdf-timeout"},
        }}
        queue = pdf_health.plan(records, report, manifest)
        selected = [item["key"] for shard in queue["shards"] for item in shard["records"]]
        self.assertNotIn("r\0proven.pdf", selected)
        self.assertIn("r\0changed.pdf", selected)
        self.assertIn("r\0legacy.pdf", selected)
        self.assertIn("r\0fresh.pdf", selected)
        self.assertEqual(queue["conversion_ready"], 1)
        entries = queue["_conversion_ready_entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["key"], "r\0proven.pdf")
        self.assertEqual(entries[0]["status"], "healthy")
        self.assertEqual(entries[0]["reason"], "conversion-ready")
        api = Mock()
        api.repo_info.return_value = Mock(sha="parent")
        stale = {"version": 1, "files": {
            "r\0proven.pdf": {"source_revision": "one", "status": "tool-error", "reason": "qpdf-timeout"}}}
        with patch.object(pdf_health, "remote_report", return_value=stale), \
                patch.object(pdf_health.time, "sleep"):
            merged = pdf_health.publish(api, "repo", [], {"r\0proven.pdf"}, entries)
        self.assertEqual(merged["files"]["r\0proven.pdf"]["status"], "healthy")
        self.assertEqual(merged["files"]["r\0proven.pdf"]["reason"], "conversion-ready")
        self.assertEqual(api.create_commit.call_count, 1)

    def test_conversion_ready_reuses_same_content_across_source_revision(self):
        from scripts import pdf_assets
        manifest = {"version": 1, "files": {
            "r\0proven.pdf": {
                "status": "ready", "strategy": "sampled-webp", "source_revision": "old",
                "source_sha256": "same", "source_bytes": 100,
                "render_profile": pdf_assets.PDF_PROFILE,
                "decision_profile": pdf_assets.PDF_DECISION_PROFILE,
            }
        }}
        records = [{"key": "r\0proven.pdf", "source_revision": "new", "declared_bytes": 100}]
        report = {"version": 1, "files": {
            "r\0proven.pdf": {"source_revision": "new", "sha256": "same", "declared_bytes": 100,
                               "status": "tool-error", "reason": "qpdf-timeout"}
        }}
        self.assertEqual(pdf_health.pending_records(records, report, manifest), [])
        report["files"]["r\0proven.pdf"]["sha256"] = "different"
        self.assertEqual(pdf_health.pending_records(records, report, manifest), records)

    def test_merge_report_prunes_removed_files(self):
        remote = {"version": 1, "files": {"keep": {"status": "healthy"},
                                            "removed": {"status": "corrupt"}}}
        merged = pdf_health.merge_report(remote, [], {"keep"})
        self.assertEqual(set(merged["files"]), {"keep"})

    def test_plan_caps_batch_without_hiding_pending_total_and_weights_by_size(self):
        total = pdf_health.BATCH_SIZE + 1
        records = [{"key": f"r\0{x:03}.pdf", "source_revision": "one", "declared_bytes": x + 1}
                   for x in range(total)]
        queue = pdf_health.plan(records, {"version": 1, "files": {}})
        self.assertEqual(queue["selected_records"], pdf_health.BATCH_SIZE)
        self.assertEqual(queue["pending_records"], total)
        self.assertEqual(queue["remaining_after_batch"], 1)
        self.assertEqual(sum(len(shard["records"]) for shard in queue["shards"]), pdf_health.BATCH_SIZE)

    def inspect_with_tools(self, qpdf=(0, "No syntax errors"), info=None, text=(0, ""), payload=None):
        info = info or (0, "Pages: 3\nEncrypted: no\nPDF version: 1.7\n")
        payload = payload or b"%PDF-1.7\nbody\n%%EOF\n"
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "book.pdf"
            path.write_bytes(payload)
            with patch.object(pdf_health, "_run", side_effect=[info, qpdf, text, text, text]):
                return pdf_health.inspect_pdf(path, len(payload))

    def test_classifier_distinguishes_healthy_warning_encrypted_and_corrupt(self):
        self.assertEqual(self.inspect_with_tools()["status"], "healthy")
        warning = self.inspect_with_tools(qpdf=(0, "WARNING: object repaired"))
        self.assertEqual((warning["status"], warning["reason"]), ("warning", "qpdf-warning"))
        encrypted = self.inspect_with_tools(info=(0, "Pages: 3\nEncrypted: yes\nPDF version: 1.6\n"),
                                            qpdf=(3, "invalid password"))
        self.assertEqual(encrypted["status"], "encrypted")
        for detail in ("xref table not found", "unable to find trailer dictionary"):
            corrupt = self.inspect_with_tools(qpdf=(2, detail))
            self.assertEqual(corrupt["status"], "corrupt")
            self.assertEqual(corrupt["reason"], "qpdf-check-failed")

    def test_truncation_and_trailing_zero_signs_are_recorded(self):
        truncated = self.inspect_with_tools(payload=b"%PDF-1.7\ntruncated")
        self.assertEqual(truncated["status"], "corrupt")
        self.assertIn("missing-eof", truncated["reasons"])
        zeros = self.inspect_with_tools(payload=b"%PDF-1.7\n%%EOF\n\0\0")
        self.assertEqual(zeros["status"], "healthy")
        self.assertEqual(zeros["trailing_zero_bytes"], 2)

    def test_timeout_and_missing_tool_are_tool_errors(self):
        timeout = self.inspect_with_tools(info=(124, "timeout after 1s"))
        self.assertEqual((timeout["status"], timeout["reason"]), ("tool-error", "pdfinfo-timeout"))
        missing = self.inspect_with_tools(qpdf=(127, "qpdf not found"))
        self.assertEqual((missing["status"], missing["reason"]), ("tool-error", "qpdf-unavailable"))

    def test_nonzero_pdfinfo_is_never_healthy(self):
        result = self.inspect_with_tools(info=(1, "Pages: 3\nEncrypted: no\nPDF version: 1.7\nsyntax warning"))
        self.assertEqual(result["status"], "warning")
        self.assertIn("pdfinfo-check-failed", result["reasons"])

    def test_diagnostics_are_bounded(self):
        result = self.inspect_with_tools(qpdf=(0, "\n".join("x" * 1000 + str(i) for i in range(30))))
        self.assertLessEqual(len(result["diagnostics"]), pdf_health.MAX_DIAGNOSTICS)
        self.assertTrue(all(len(line) <= pdf_health.MAX_DIAGNOSTIC_CHARS for line in result["diagnostics"]))

    def test_successful_extracted_text_is_not_retained(self):
        result = self.inspect_with_tools(text=(0, "private page contents"))
        self.assertEqual(result["diagnostics"], [])
        self.assertEqual(result["sample_pages"], [1, 2, 3])

    def test_publish_retries_parent_race_and_preserves_unrelated_entries(self):
        response = requests.Response()
        response.status_code = 409
        response.request = requests.Request("POST", "https://huggingface.co/commit").prepare()
        error = HfHubHTTPError("race", response=response)
        api = Mock()
        api.repo_info.side_effect = [Mock(sha="one"), Mock(sha="two")]
        api.create_commit.side_effect = [error, None]
        remotes = [
            {"version": 1, "files": {"other\0a.pdf": {"status": "healthy"}}},
            {"version": 1, "files": {"other\0a.pdf": {"status": "healthy"},
                                      "other\0b.pdf": {"status": "warning"}}},
        ]
        result = {"key": "r\0book.pdf", "source_revision": "rev", "status": "healthy"}
        with patch.object(pdf_health, "remote_report", side_effect=remotes), patch.object(pdf_health.time, "sleep"):
            merged = pdf_health.publish(api, "repo", [result])
        self.assertEqual(set(merged["files"]), {"other\0a.pdf", "other\0b.pdf", "r\0book.pdf"})
        self.assertEqual([call.kwargs["parent_commit"] for call in api.create_commit.call_args_list], ["one", "two"])

if __name__ == "__main__":
    unittest.main()
