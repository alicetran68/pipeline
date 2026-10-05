import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import pilot_epub_chapters


class ChmChapterPilotTests(unittest.TestCase):
    def test_chm_publishes_chapter_tree_and_native_epub_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            source_bytes = b"source chm"
            source_digest = hashlib.sha256(source_bytes).hexdigest()
            item = {"repo": "VoiceOfML/Test", "path": "book.chm", "revision": "rev",
                    "source_bytes": len(source_bytes), "extension": "chm"}

            def download(_url, target, _token):
                target.write_bytes(source_bytes)

            def convert(queue_item, output):
                chapter_root = output / "objects" / "chapters" / "epub-chapters"
                (chapter_root / "chapters").mkdir(parents=True)
                (chapter_root / "chapters" / "chapter-0001.xhtml").write_text("<p>body</p>")
                (chapter_root / "epub-search-index.json.gz").write_bytes(b"search")
                manifest = {"version": 1, "kind": "epub-chapters", "chapters": [{
                    "index": 1, "title": "Book", "path": "chapters/chapter-0001.xhtml",
                    "bytes": 11, "sha256": "a" * 64,
                }]}
                (chapter_root / "chapter-manifest.json").write_text(json.dumps(manifest))
                native = output / "objects" / "document.epub"
                native.parent.mkdir(parents=True, exist_ok=True)
                native.write_bytes(b"native epub")
                return {"path": "objects/document.epub", "profile": queue_item["profile"],
                        "chapter_manifest": "objects/chapters/epub-chapters/chapter-manifest.json",
                        "chapter_bundle_error": None}

            with patch.object(pilot_epub_chapters, "download", side_effect=download), \
                    patch.object(pilot_epub_chapters, "convert_item", side_effect=convert):
                entry, uploads = pilot_epub_chapters.build_one(item, work / "book", "token", "vomebook/reader-assets-v2")

            self.assertEqual(entry["source_sha256"], source_digest)
            self.assertEqual(entry["source_extension"], "chm")
            self.assertTrue(entry["fallback"].startswith("native/ebook/chm/"))
            self.assertTrue(any(path.endswith("chapter-manifest.json") and path.startswith("chapters/ebook/chm/")
                                for path in uploads))
            self.assertIn(entry["fallback"], uploads)


if __name__ == "__main__":
    unittest.main()
