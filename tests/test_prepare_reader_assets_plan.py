import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_reader_assets_plan


class PrepareReaderAssetsPlanTests(unittest.TestCase):
    def test_prepare_splits_only_the_selected_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "queue.json"
            output.write_text(json.dumps({
                "items": [
                    {"extension": "pdf", "key": "a"},
                    {"extension": "docx", "key": "b"},
                ],
                "stale_keys": ["old"],
                "authoritative_snapshot": True,
            }))
            with patch.object(prepare_reader_assets_plan, "scan_reader_assets") as scanner:
                scanner.shard_for_key.return_value = 0
                with patch.dict(os.environ, {"FORCE_REBUILD": "true", "INPUT_PATH": ""}, clear=False):
                    extension, count, stale, authoritative = prepare_reader_assets_plan.prepare(output)
            self.assertEqual((extension, count, stale, authoritative), ("pdf", 1, 1, True))
            self.assertEqual(json.loads((root / "queue.json").read_text())["items"], [{"extension": "pdf", "key": "a"}])

    def test_script_imports_when_executed_from_scripts_directory(self):
        self.assertTrue(prepare_reader_assets_plan.scan_reader_assets)

    def test_bucket_migration_keeps_the_mixed_static_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "queue.json"
            output.write_text(json.dumps({
                "items": [
                    {"extension": "docx", "key": "a"},
                    {"extension": "md", "key": "b"},
                    {"extension": "png", "key": "c"},
                ],
                "bucket_migration": True,
                "stale_keys": [],
                "authoritative_snapshot": False,
            }))
            with patch.object(prepare_reader_assets_plan, "scan_reader_assets") as scanner:
                scanner.shard_for_key.return_value = 0
                with patch.dict(os.environ, {"FORCE_REBUILD": "true", "INPUT_PATH": ""}, clear=False):
                    extension, count, _, authoritative = prepare_reader_assets_plan.prepare(output)
            self.assertEqual((extension, count, authoritative), ("static", 3, False))
            self.assertEqual(len(json.loads((root / "queue.json").read_text())["items"]), 3)
            self.assertEqual(len(json.loads((root / "queue-0.json").read_text())["items"]), 3)
            self.assertEqual(json.loads((root / "queue-19.json").read_text())["items"], [])

    def test_bucket_migration_keeps_twenty_shards_for_scoped_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "queue.json"
            output.write_text(json.dumps({
                "items": [{"extension": "chm", "key": str(index)} for index in range(40)],
                "bucket_migration": True,
                "stale_keys": [],
                "authoritative_snapshot": False,
            }))
            with patch.object(prepare_reader_assets_plan, "scan_reader_assets") as scanner:
                scanner.shard_for_key.side_effect = lambda key, count: int(key) % count
                with patch.dict(os.environ, {"FORCE_REBUILD": "true", "INPUT_EXTENSION": "chm",
                                              "INPUT_PATH": "", "INPUT_REPO": ""}, clear=False):
                    prepare_reader_assets_plan.prepare(output)
            self.assertEqual(len(json.loads((root / "queue-0.json").read_text())["items"]), 2)
            self.assertEqual(len(json.loads((root / "queue-19.json").read_text())["items"]), 2)


if __name__ == "__main__":
    unittest.main()
