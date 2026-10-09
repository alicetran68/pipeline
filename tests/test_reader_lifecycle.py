import unittest
from datetime import datetime, timezone

from scripts import reader_lifecycle


class ReaderLifecycleTests(unittest.TestCase):
    def test_reader_assets_are_final_and_have_no_staging_consumers(self):
        record = reader_lifecycle.asset_record({
            "key": "repo\0book.pdf", "path": "objects/aa/document.pdf",
            "source_bytes": 8 * 1024 * 1024,
        })
        self.assertEqual(record["phase"], "final")
        self.assertEqual(record["consumers"], {})

    def test_orphan_marking_forgets_referenced_paths(self):
        marked = reader_lifecycle.mark_orphans(
            {"version": 1, "files": {}, "orphans": {"melsm:objects/a": {"since": "2020-01-01"}}},
            {"vomebook/pdf-pages:objects/b"}, "2026-09-29",
        )
        self.assertEqual(set(marked["orphans"]), {"vomebook/pdf-pages:objects/b"})

    def test_catalog_generation_is_content_addressed_and_old_generation_waits_for_acks(self):
        sidecar = {"v": 1, "f": {"book": {
            "p": "objects/a/page-manifest.json", "b": reader_lifecycle.PDF_PAGES_BUCKET,
        }}}
        first, first_id = reader_lifecycle.catalog_generation(sidecar, now="2026-10-09T00:00:00+00:00")
        same, same_id = reader_lifecycle.catalog_generation(sidecar, first,
                                                            now="2026-10-09T00:01:00+00:00")
        self.assertEqual(first_id, same_id)
        self.assertEqual(first, same)
        newer = {"v": 1, "f": {"book": {
            "p": "objects/b/page-manifest.json", "b": reader_lifecycle.PDF_PAGES_BUCKET,
        }}}
        second, second_id = reader_lifecycle.catalog_generation(newer, first,
                                                                 now="2026-10-10T00:00:00+00:00")
        self.assertNotEqual(first_id, second_id)
        old = second["generations"][first_id]
        clock = datetime(2026, 11, 10, tzinfo=timezone.utc)
        self.assertTrue(reader_lifecycle.generation_live(old, second_id, clock))
        old["replacement_acks"] = {"hf": True, "pages": True}
        self.assertFalse(reader_lifecycle.generation_live(old, second_id, clock))

    def test_processing_root_requires_qualified_resources_and_releases_idempotently(self):
        with self.assertRaises(ValueError):
            reader_lifecycle.processing_record([{"root": "objects/a"}], "upload", {})
        record = reader_lifecycle.processing_record([{
            "bucket": reader_lifecycle.PDF_PAGES_BUCKET, "root": "objects/a"}], "upload", {})
        released = reader_lifecycle.release_processing(record, "generation", "2026-10-09T00:00:00+00:00")
        self.assertEqual(released["status"], "released")
        self.assertEqual(reader_lifecycle.release_processing(released, "generation"), released)


if __name__ == "__main__":
    unittest.main()
