import io
import json
import unittest
from unittest.mock import Mock, patch

from scripts.reader_bucket_store import S3BucketStore


class ReaderBucketStoreTests(unittest.TestCase):
    def store(self, client):
        store = S3BucketStore.__new__(S3BucketStore)
        store._location = Mock(return_value=("vomebook", "reader-assets-v2"))
        store._client = Mock(return_value=client)
        return store

    def test_head_inventory_distinguishes_missing_from_permissions(self):
        class Missing(Exception):
            response = {"Error": {"Code": "404"}}
        class Denied(Exception):
            response = {"Error": {"Code": "AccessDenied"}}
        client = Mock()
        client.head_object.side_effect = [None, Missing()]
        store = self.store(client)
        self.assertEqual(store.existing_files("bucket", {"a", "b"}), {"a"})
        client.head_object.side_effect = Denied()
        with self.assertRaises(Denied):
            store.existing_files("bucket", {"a"})

    def test_read_closes_the_response_body(self):
        body = io.BytesIO(b"json")
        client = Mock()
        client.get_object.return_value = {"Body": body}
        self.assertEqual(self.store(client).read_bytes("bucket", "path"), b"json")
        self.assertTrue(body.closed)

    def test_delete_checks_per_object_errors(self):
        client = Mock()
        client.delete_objects.return_value = {"Errors": [{"Key": "a", "Code": "AccessDenied"}]}
        with self.assertRaises(RuntimeError):
            self.store(client).delete("bucket", ["a"])

    def test_integrity_scan_preserves_page_and_chapter_contracts(self):
        store = self.store(Mock())
        manifests = {
            "objects/a/page-manifest.json": {"page_count": 2},
            "ebook-chapters/objects/b/chapter-manifest.json": {
                "chapters": [{"path": "chapters/chapter-0001.xhtml"}],
                "search_index": {"path": "epub-search-index.json.gz"},
            },
        }
        store.read_bytes = Mock(side_effect=lambda bucket, path: json.dumps(manifests[path]).encode())
        self.assertEqual(store.manifest_references("bucket", set(manifests)), {
            "objects/a/page-manifest.json": {
                "objects/a/pages/page-000001.webp", "objects/a/pages/page-000002.webp"},
            "ebook-chapters/objects/b/chapter-manifest.json": {
                "ebook-chapters/objects/b/chapters/chapter-0001.xhtml",
                "ebook-chapters/objects/b/epub-search-index.json.gz"},
        })


if __name__ == "__main__":
    unittest.main()
