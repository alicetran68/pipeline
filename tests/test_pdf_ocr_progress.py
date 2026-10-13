import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import pdf_ocr_progress as progress, pdf_ocr_stages as stages
from scripts.reader_bucket import index_path
from scripts.reader_gc_graph import is_manifest, is_root


def page(number, text="t"):
    return {"p": number, "source": "ocr", "o": f"objects/aa/{'a' * 64}/{'b' * 16}/ocr/page-{number:06d}.json.gz",
            "os": "c" * 64, "ob": 10, "text": text}


class MemoryBucket:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.publishes = []

    def read(self, path):
        if path not in self.objects:
            raise FileNotFoundError(path)
        return self.objects[path]

    def publish(self, payloads):
        self.publishes.append(sorted(payloads))
        self.objects.update(payloads)

    def store(self):
        return progress.ProgressStore(read=self.read, publish=self.publish)


class OcrProgressStoreTests(unittest.TestCase):
    def legacy(self, files):
        return json.dumps({"files": files, "version": 1}, ensure_ascii=False, sort_keys=True, indent=2).encode()

    def migrate(self, bucket, legacy_bytes):
        def download(target, token=None):
            if legacy_bytes is None:
                return None
            target.write_bytes(legacy_bytes)
            return "d" * 64, len(legacy_bytes)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(progress, "download_legacy", side_effect=download):
            return progress.migrate_legacy(bucket.store(), Path(directory))

    def test_migration_splits_books_writes_index_last_and_keeps_gc_roots(self):
        files = {"甲/书.pdf": {"generation": "g1", "pages": {"1": page(1, "古史"), "3": page(3)}},
                 "乙.pdf": {"generation": "g2", "pages": {}}}
        bucket = MemoryBucket()
        with patch.object(progress, "MIGRATION_BATCH", 1):
            result = self.migrate(bucket, self.legacy(files))
        self.assertEqual(result["books"], 2)
        self.assertEqual(result["pages"], 2)
        self.assertEqual(bucket.publishes[-1], [index_path(progress.INDEX)])
        self.assertEqual(len(bucket.publishes), 3)
        store = bucket.store()
        self.assertEqual(store.get("甲/书.pdf"), files["甲/书.pdf"])
        self.assertEqual(store.get("missing", {}), {})
        self.assertEqual(store.summaries()["甲/书.pdf"], {"generation": "g1", "pages": [1, 3]})
        for path in bucket.objects:
            self.assertTrue(is_root(path))
            self.assertFalse(is_manifest(path))
        shard = json.loads(gzip.decompress(bucket.objects[index_path(progress.shard_name("甲/书.pdf"))]))
        self.assertEqual(shard["files"]["甲/书.pdf"]["pages"]["1"]["o"], files["甲/书.pdf"]["pages"]["1"]["o"])

    def test_missing_legacy_creates_empty_index_and_rejects_invalid_registry(self):
        bucket = MemoryBucket()
        self.assertEqual(self.migrate(bucket, None), {"books": 0, "pages": 0})
        self.assertEqual(bucket.store().summaries(), {})
        with self.assertRaisesRegex(ValueError, "version 1"):
            self.migrate(MemoryBucket(), json.dumps({"files": {}, "version": 2}).encode())
        with self.assertRaisesRegex(ValueError, "invalid OCR progress page"):
            self.migrate(MemoryBucket(), self.legacy({"x": {"generation": "g", "pages": {"2": page(1)}}}))

    def test_save_rewrites_only_updated_books_with_existing_merge_semantics(self):
        bucket = MemoryBucket()
        self.migrate(bucket, self.legacy({"a": {"generation": "g", "pages": {"1": page(1)}},
                                          "b": {"generation": "g", "pages": {"1": page(1)}}}))
        untouched = bucket.objects[index_path(progress.shard_name("b"))]
        store = bucket.store()
        store.save({"a": {"generation": "g", "pages": {"2": page(2)}}}, stages.merge_progress)
        self.assertEqual(bucket.publishes[-1], sorted([index_path(progress.shard_name("a")),
                                                      index_path(progress.INDEX)]))
        self.assertEqual(sorted(bucket.store().get("a")["pages"]), ["1", "2"])
        self.assertEqual(bucket.objects[index_path(progress.shard_name("b"))], untouched)
        bucket.store().save({"a": {"generation": "new", "pages": {"5": page(5)}}}, stages.merge_progress)
        self.assertEqual(bucket.store().summaries()["a"], {"generation": "new", "pages": [5]})
        with self.assertRaisesRegex(RuntimeError, "migrate"):
            MemoryBucket().store().save({"a": {"generation": "g", "pages": {}}}, stages.merge_progress)

    def test_readback_mismatch_and_wrong_shard_key_fail_closed(self):
        bucket = MemoryBucket()
        self.migrate(bucket, self.legacy({"a": {"generation": "g", "pages": {}}}))
        bucket.objects[index_path(progress.shard_name("a"))] = progress.encode_shard("other", {
            "generation": "g", "pages": {}})
        with self.assertRaisesRegex(ValueError, "invalid OCR progress shard"):
            bucket.store().get("a")
        bucket.objects[index_path(progress.shard_name("a"))] = progress.encode_shard("a", {
            "generation": "g", "pages": {}})
        lossy = progress.ProgressStore(read=bucket.read, publish=lambda payloads: None)
        with patch.object(progress.time, "sleep"), self.assertRaisesRegex(ValueError, "readback mismatch"):
            lossy.save({"a": {"generation": "g2", "pages": {}}}, stages.merge_progress)

    def test_planner_resumes_from_lazy_store(self):
        bucket = MemoryBucket()
        self.migrate(bucket, self.legacy({"book": {"generation": "g", "pages": {"1": page(1)}}}))
        store = bucket.store()
        self.assertEqual(store.get("book", {}).get("generation"), "g")
        self.assertEqual(store.get("other", {}).get("pages", {}), {})

    def retire(self, bucket, current):
        def download(target, token=None):
            target.write_bytes(current)
            return ("d" * 64 if current == self.original else "e" * 64), len(current)
        with patch.object(progress, "download_legacy", side_effect=download):
            return progress.retire_legacy(bucket.store())

    def test_legacy_root_is_tombstoned_only_when_unchanged_since_migration(self):
        files = {"book": {"generation": "g", "pages": {"1": page(1)}}}
        self.original = self.legacy(files)
        changed = MemoryBucket()
        self.migrate(changed, self.original)
        self.assertEqual(self.retire(changed, self.legacy({**files, "late": files["book"]})), "legacy-changed")
        self.assertNotIn(index_path(progress.LEGACY), changed.objects)
        bucket = MemoryBucket()
        self.migrate(bucket, self.original)
        self.assertEqual(self.retire(bucket, self.original), "retired")
        stone = json.loads(bucket.objects[index_path(progress.LEGACY)])
        self.assertEqual(stone["files"], {})
        self.assertEqual(stone["migrated"]["books"], 1)
        self.assertTrue(is_root(index_path(progress.LEGACY)))
        self.assertTrue(bucket.store().index()["legacy_retired"])
        self.assertEqual(bucket.store().get("book"), files["book"])
        self.assertEqual(self.retire(bucket, self.original), "not-applicable")
        # Tombstone written but index flag lost: recognized, not treated as a change.
        flag_lost = MemoryBucket()
        self.migrate(flag_lost, self.original)
        self.assertEqual(self.retire(flag_lost, progress.encode_json(progress.tombstone(
            flag_lost.store().index()["migrated_from"]))), "retired")
        self.assertTrue(flag_lost.store().index()["legacy_retired"])


if __name__ == "__main__":
    unittest.main()
