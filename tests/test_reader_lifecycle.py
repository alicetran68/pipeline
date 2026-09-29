import unittest

from scripts import reader_lifecycle


class ReaderLifecycleTests(unittest.TestCase):
    def test_staging_record_tracks_render_and_range_consumers(self):
        record = reader_lifecycle.staging_record({
            "key": "repo\0book.pdf", "path": "objects/aa/document.pdf",
            "source_bytes": 8 * 1024 * 1024, "bucket_staging": True,
        })
        self.assertEqual(record["phase"], "staging")
        self.assertEqual(record["consumers"], {"render": "pending", "range": "pending"})

    def test_completed_consumers_make_staging_object_collectible(self):
        record = reader_lifecycle.staging_record({
            "key": "repo\0book.pdf", "path": "objects/aa/document.pdf",
            "source_bytes": 1, "bucket_staging": True,
        })
        record = {**record, "consumers": {"render": "done", "range": "not-needed"}}
        self.assertTrue(reader_lifecycle.collectible(record))

    def test_orphan_marking_forgets_referenced_paths(self):
        marked = reader_lifecycle.mark_orphans(
            {"version": 1, "files": {}, "orphans": {"melsm:objects/a": {"since": "2020-01-01"}}},
            {"vomebook/pdf-pages:objects/b"}, "2026-09-29",
        )
        self.assertEqual(set(marked["orphans"]), {"vomebook/pdf-pages:objects/b"})


if __name__ == "__main__":
    unittest.main()
