import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from scripts import prune_pdf_bucket as gc


class PdfBucketGcTests(unittest.TestCase):
    def test_object_root_accepts_only_canonical_object_paths(self):
        root = "objects/aa/" + "a" * 64 + "/" + "b" * 16
        self.assertEqual(gc.object_root(root + "/pages/page-000001.webp"), root)
        for path in ("", "objects/aa/hash/root/file", "other/aa/" + "a" * 64 + "/" + "b" * 16):
            self.assertIsNone(gc.object_root(path))

    def test_referenced_roots_walks_manifests_and_sidecars(self):
        first = "objects/aa/" + "a" * 64 + "/" + "b" * 16
        second = "objects/bb/" + "c" * 64 + "/" + "d" * 16
        roots = gc.referenced_roots({"p": first + "/page-manifest.json"},
                                    {"pages": [{"o": second + "/ocr/page-000001.json.gz"}]})
        self.assertEqual(roots, {first, second})

    def test_eligible_roots_keeps_referenced_recent_and_unknown_dates(self):
        old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2026, 9, 25, tzinfo=timezone.utc)
        roots = [("old", old), ("protected", old), ("recent", recent), ("unknown", None)]
        self.assertEqual(gc.eligible_roots(roots, {"protected"}, date(2026, 9, 28), 14, 100), ["old"])

    def test_apply_requires_explicit_unreferenced_gate(self):
        with patch.dict("os.environ", {"HF_TOKEN": "token"}, clear=False), \
             patch.object(gc.HfApi, "repo_info") as info:
            with patch("sys.argv", ["prune_pdf_bucket", "--apply"]):
                with self.assertRaisesRegex(ValueError, "allow-unreferenced"):
                    gc.main()
            info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
