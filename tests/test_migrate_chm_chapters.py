import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts import migrate_chm_chapters, reader_assets


class ChmMigrationTests(unittest.TestCase):
    def test_only_unmigrated_ready_chm_epubs_are_selected(self):
        manifest = {"files": {
            "a": {"status": "ready", "source_extension": "chm", "reader_mode": "epub"},
            "b": {"status": "ready", "source_extension": "epub", "reader_mode": "foliate"},
            "c": {"status": "failed", "source_extension": "chm", "reader_mode": "epub"},
            "d": {"status": "ready", "source_extension": "chm", "reader_mode": "epub",
                  "chapter_manifest": "objects/aa/x/y/z/epub-chapters/chapter-manifest.json",
                  "chapter_bucket": reader_assets.READER_EBOOK_BUCKET},
        }}
        selected = migrate_chm_chapters.migration_candidates(manifest, shard_count=1, shard_index=0)
        self.assertEqual([key for key, _ in selected], ["a"])

    def test_bucket_upload_uses_direct_add_without_listing_existing_bucket(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            file = bundle / "objects/aa/hash/profile/epub-chapters/chapter-manifest.json"
            file.parent.mkdir(parents=True)
            file.write_text("{}", encoding="utf-8")
            api = Mock()
            self.assertEqual(migrate_chm_chapters.upload_bucket_files(api, bundle, "token"), 1)
            api.batch_bucket_files.assert_called_once()
            self.assertEqual(api.batch_bucket_files.call_args.kwargs["add"][0][1],
                             "ebook-chapters/objects/aa/hash/profile/epub-chapters/chapter-manifest.json")


if __name__ == "__main__":
    unittest.main()
