import copy
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import Mock, patch

from scripts import reader_v3_automation as auto, publish_reader_v3 as publication
from tests import test_publish_reader_v3 as fixtures


def entry(key="VoiceOfML/books\0book.pdf", source="a" * 64, count=97):
    return {"key": key, "repo": key.split("\0")[0], "path": key.split("\0")[1],
            "source_kind": "upstream", "source_revision": "b" * 40, "source_sha256": source,
            "source_bytes": 100, "page_count": count, "status": "ready",
            "ocr_manifest": f"objects/{source[:2]}/{source}/{'c' * 16}/ocr-manifest.json",
            "ocr_manifest_sha256": "d" * 64, "ocr_manifest_bytes": 100}


class AutomationTests(unittest.TestCase):
    def setUp(self):
        self.store = fixtures.Store()
        self.now = datetime(2026, 10, 10, tzinfo=timezone.utc)
        self.add_entries(entry())

    def add_entries(self, *entries):
        self.store.objects[(publication.ASSETS, auto.REGISTRY)] = publication.encode(
            {"version": 1, "files": {e["key"]: e for e in entries}})

    def test_dry_run_never_builds_or_writes(self):
        builder = Mock()
        result = auto.run(self.store, build=builder, now=self.now)
        self.assertEqual(result["eligible"], 1)
        self.assertEqual(self.store.writes, [])
        builder.assert_not_called()

    def test_staged_checkpoint_retries_promotion_without_rebuilding(self):
        fixture = fixtures.V3PublicationTests()
        fixture.setUp()
        try:
            ref = fixture.candidate(key=entry()["key"])
            staged = publication.stage(self.store, ref, fixture.bundle, apply=True)
            build = Mock(return_value=staged)
            self.store.fail_pointer = True
            first = auto.run(self.store, apply=True, build=build, now=self.now)
            self.assertEqual(first["processed"][0]["status"], "retry")
            self.store.fail_pointer = False
            second = auto.run(self.store, apply=True, build=build, now=self.now + timedelta(hours=3))
            self.assertEqual(second["processed"][0]["status"], "published")
            self.assertEqual(build.call_count, 1)
            self.assertEqual(second["budget"]["attempts"], 1)
        finally:
            fixture.temp.cleanup()

    def test_interrupted_build_charges_budget_and_daily_reset_is_utc(self):
        build = Mock(side_effect=OSError("transport"))
        auto.run(self.store, apply=True, build=build, now=self.now)
        auto.run(self.store, apply=True, build=build, now=self.now + timedelta(hours=3))
        result = auto.run(self.store, apply=True, build=build, now=self.now + timedelta(hours=10))
        self.assertEqual(result["eligible"], 0)
        self.assertEqual(build.call_count, 2)
        auto.run(self.store, apply=True, build=build, now=self.now + timedelta(days=1))
        task = next(iter(auto.load_state(self.store)["tasks"].values()))
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["attempts"], 3)

    def test_existing_book_is_protected_even_when_raw_recipe_or_source_changes(self):
        fixture = fixtures.V3PublicationTests()
        fixture.setUp()
        try:
            ref = fixture.candidate(key=entry()["key"])
            staged = publication.stage(self.store, ref, fixture.bundle, apply=True)
            publication.promote(self.store, staged["candidate"], None, apply=True)
            self.add_entries(entry(source="e" * 64))
            build = Mock()
            result = auto.run(self.store, apply=True, build=build, now=self.now)
            self.assertEqual(result["eligible"], 0)
            build.assert_not_called()
            self.assertEqual(next(iter(auto.load_state(self.store)["tasks"].values()))["status"], "protected")
        finally:
            fixture.temp.cleanup()

    def test_page_budget_never_truncates_a_book(self):
        self.add_entries(entry(count=1001))
        result = auto.run(self.store, now=self.now)
        self.assertEqual(result["tasks"], 1)
        self.assertEqual(result["eligible"], 0)

    def test_unusable_converted_text_requires_review_without_repeated_builds(self):
        builder = Mock(side_effect=publication.PublicationReviewRequired())
        report = auto.run(self.store, apply=True, build=builder, now=self.now)
        self.assertEqual(report["processed"][0]["status"], "needs-review")
        auto.run(self.store, apply=True, build=builder, now=self.now + timedelta(days=1))
        self.assertEqual(builder.call_count, 1)

    def test_explicit_retry_retains_day_charges_and_failure_history(self):
        auto.run(self.store, apply=True, build=Mock(side_effect=OSError()), now=self.now)
        state = auto.load_state(self.store)
        identity = next(iter(state["tasks"]))
        before = copy.deepcopy(state["days"])
        with patch.dict("os.environ", {"GITHUB_ACTOR": "reviewer"}):
            auto.retry(self.store, {"task_ids": [identity]}, apply=True)
        state = auto.load_state(self.store)
        self.assertEqual(state["days"], before)
        self.assertEqual(state["tasks"][identity]["attempts"], 0)
        self.assertEqual(state["tasks"][identity]["retries"][0]["actor"], "reviewer")

    def test_converted_primary_uses_qualified_derivative_and_original_identity(self):
        value = entry(key="VoiceOfML/books\0book.djvu")
        value.update(source_kind="generated", reader_assets_bucket="vomebook/pdf-pages-v2",
                     reader_assets_path="derived/migration-test/" + "a" * 32 + "/document.pdf", source_url="")
        spec, count = auto.spec_for(value["key"], value)
        self.assertEqual(count, 97)
        self.assertEqual(spec["primary"]["path"], value["reader_assets_path"])
        self.assertNotIn("primary_source", spec)
        for change in ({"source_kind": "upstream"}, {"reader_assets_bucket": "melsm/pdf-archive-v2"},
                       {"reader_assets_path": "derived/../document.pdf"}, {"source_url": "https://evil.example/book.pdf"}):
            with self.assertRaises(ValueError):
                auto.spec_for(value["key"], {**value, **change})
