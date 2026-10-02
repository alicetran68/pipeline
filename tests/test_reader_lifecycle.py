import unittest

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


if __name__ == "__main__":
    unittest.main()
