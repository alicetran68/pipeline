from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class ReaderAssetConcurrencyTests(unittest.TestCase):
    def test_all_reader_asset_publishers_share_one_queue(self):
        for filename in (
            "reader-assets.yml", "prune-reader-assets.yml", "pdf-assets-worker.yml",
            "migrate-pdf-page-manifests.yml",
            "migrate-pdf-range-bucket.yml",
        ):
            workflow = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("group: reader-assets", workflow, filename)
            self.assertNotIn("group: reader-assets-pdf", workflow, filename)
            self.assertNotIn("group: pdf-range-assets", workflow, filename)
            self.assertIn("cancel-in-progress: false", workflow, filename)
            self.assertIn("queue: max", workflow, filename)

    def test_pdf_worker_skips_empty_dynamic_matrix(self):
        import yaml
        workflow = yaml.safe_load((ROOT / "pdf-assets-worker.yml").read_text())
        plan = workflow["jobs"]["plan"]
        build = workflow["jobs"]["build"]
        self.assertIn("shard_count", plan["outputs"])
        self.assertIn("shard_ids", plan["outputs"])
        self.assertEqual(build["if"], "${{ needs.plan.outputs.shard_count != '0' }}")
        self.assertIn("fromJSON(needs.plan.outputs.shard_ids)", build["strategy"]["matrix"]["shard"])


if __name__ == "__main__":
    unittest.main()
