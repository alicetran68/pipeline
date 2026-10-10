import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from scripts import pdf_ocr, pdf_ocr_stages as stages, pdf_worker_lanes as lanes
from scripts.validate_pdf_worker_run import validate_run


class PdfWorkerLaneTests(unittest.TestCase):
    def setUp(self):
        self.config = lanes.load_config()

    def item(self, name="book.pdf", size=1024, **fields):
        return {"key": "repo\0" + name, "repo": "repo", "path": name,
                "source_extension": Path(name).suffix.lstrip("."), "source_kind": "upstream",
                "source_revision": "revision", "source_bytes": size, **fields}

    def test_every_source_has_one_account_at_boundaries_and_after_conversion(self):
        items = [self.item(str(index) + ".pdf", size) for index, size in enumerate(
            [0, None, -1, True, 1.5, "1024", 1, 5242879, 5242880, 10485759, 10485760, 52428799,
             52428800, 104857599, 104857600, 524287999, 524288000, 2**40])]
        items += [self.item("book." + extension, 1, source_kind="generated", extension="pdf")
                  for extension in ["djvu", "caj", "kdh", "doc", "tiff"]]
        selections = [lanes.select_records(self.config, items, lane["name"], lane["owner"])
                      for lane in [*self.config["native_lanes"], self.config["converted_lane"]]]
        keys = [item["key"] for selected in selections for item in selected]
        self.assertCountEqual(keys, [item["key"] for item in items])
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual([item["path"] for item in selections[0]], ["6.pdf", "7.pdf"])
        self.assertEqual([item["path"] for item in selections[1]], ["8.pdf", "9.pdf"])
        self.assertEqual([item["path"] for item in selections[2]], ["10.pdf", "11.pdf"])
        self.assertEqual([item["path"] for item in selections[3]], ["12.pdf", "13.pdf"])
        self.assertEqual([item["path"] for item in selections[4]], ["14.pdf"])
        self.assertEqual([item["path"] for item in selections[5]],
                         ["15.pdf"])
        self.assertEqual([item["path"] for item in selections[6]],
                          ["0.pdf", "1.pdf", "2.pdf", "3.pdf", "4.pdf", "5.pdf", "16.pdf", "17.pdf"])
        self.assertEqual(len(selections[-1]), 5)
        self.assertEqual(lanes.lane_for_owner(self.config, "rioholland79"), "native-small-upper")
        self.assertEqual(lanes.lane_for_owner(self.config, "devondunn7"), "native-small-mid")
        self.assertEqual(lanes.lane_for_owner(self.config, "dellamcastillo"), "native-medium-lower")

    def test_source_probe_does_not_change_account_ownership(self):
        source = self.item(size=0)
        selected = lanes.select_records(self.config, [source], "native-large", "alicetran68")[0]
        self.assertEqual(lanes.lane_for(self.config, {**selected, "source_bytes": 1024}), "native-large")
        repaired = self.item(size=1024, source_kind="generated", original_source_bytes=104857600)
        self.assertEqual(lanes.lane_for(self.config, repaired), "native-medium")
        with self.assertRaises(ValueError):
            lanes.select_records(self.config, [source], "native-large", "vomebook")

    def test_repository_override_keeps_routing_fingerprint_and_enforces_account(self):
        config = copy.deepcopy(self.config)
        original = lanes.config_identity(config)
        config["native_lanes"][0]["repository"] = "vomebook/other-worker"
        self.assertEqual(lanes.config_identity(config), original)
        self.assertEqual(lanes.repository_for_owner(config, "vomebook"), "vomebook/other-worker")
        run = {"repository": {"full_name": "vomebook/pipeline"}, "head_branch": "main",
               "path": ".github/workflows/pdf-account-worker.yml", "event": "workflow_dispatch",
               "status": "completed", "conclusion": "success"}
        with self.assertRaises(ValueError):
            validate_run(config, "vomebook/pipeline", run)
        run["repository"]["full_name"] = "vomebook/other-worker"
        validate_run(config, "vomebook/other-worker", run)

    def test_generated_stream_keeps_primary_pdf_through_sidecar_rebuild(self):
        from scripts import shared
        from scripts.build_reader_assets_v2_sidecar import add_pdf_streams
        path = "derived/test/" + "a" * 32 + "/document.pdf"
        stream = "objects/aa/" + "a" * 64 + "/" + "b" * 16 + "/page-manifest.json"
        result = {"status": "ready", "reader_assets_path": path,
                  "reader_assets_bucket": shared.PDF_PAGES_BUCKET, "page_manifest": {"path": stream}}
        entry = shared.pdf_pages_sidecar_entry(stream, result)
        self.assertEqual((entry["pd"], entry["pdb"]), (path, shared.PDF_PAGES_BUCKET))
        files = {}
        add_pdf_streams(files, {"files": {"key": result}}, shared.PDF_PAGES_BUCKET)
        self.assertEqual(files["key"], entry)
        original = {"s": 2, "m": "p", "p": path, "b": shared.PDF_PAGES_BUCKET}
        rebuilt = shared.pdf_pages_sidecar_entry(stream, {}, original)
        self.assertEqual((rebuilt["pd"], rebuilt["pdb"]), (path, shared.PDF_PAGES_BUCKET))
        rebuilt_again = shared.pdf_pages_sidecar_entry(stream, {}, rebuilt)
        self.assertEqual(rebuilt_again, rebuilt)

    def test_current_pdf_source_indexes_feed_the_converted_lane(self):
        from huggingface_hub import HfApi
        from scripts import shared
        def load(path, token, bucket=shared.READER_ASSETS_BUCKET):
            if path == "reader-index/manifest.json":
                return {"version": 1, "files": {}}
            if path == "documents/pdf/ppt/index.json":
                return {"files": [{"key": "repo\0slides.ppt", "object": "documents/pdf/ppt/hash/document.pdf",
                                   "bytes": 50, "source_revision": "revision"}]}
            if path == "reader-index/derived_pdf_manifest.json":
                self.assertEqual(bucket, shared.PDF_PAGES_BUCKET)
                return {"files": [{"key": "repo\0scan.djvu", "new_path": "derived/weekly-djvu/hash/document.pdf",
                                   "derived_bytes": 100, "source_revision": "revision"}]}
            raise FileNotFoundError(path)
        with patch.object(stages, "read_bucket_json", side_effect=load):
            sources = stages.load_pdf_sources(HfApi(), "unused")
        with patch.object(pdf_ocr.pdf_assets, "load_records", return_value=[]):
            records = pdf_ocr.source_records(Path("unused"), Path("unused"), sources)
        self.assertEqual({item["source_extension"] for item in records}, {"ppt", "djvu"})
        self.assertTrue(all(lanes.lane_for(self.config, item) == "converted" for item in records))
        self.assertEqual({item["reader_assets_bucket"] for item in records},
                         {shared.READER_ASSETS_BUCKET, shared.PDF_PAGES_BUCKET})

    def test_generated_pdf_download_uses_its_explicit_current_bucket(self):
        from scripts import plan_pdf_ocr, run_pdf_ocr, shared
        item = self.item("source.djvu", source_kind="generated",
                         reader_assets_bucket=shared.PDF_PAGES_BUCKET,
                         reader_assets_path="derived/weekly-djvu/hash/document.pdf")
        for module, operation in ((plan_pdf_ocr, plan_pdf_ocr.download_source), (run_pdf_ocr, run_pdf_ocr.source_path)):
            with patch.object(module, "materialize_bucket", return_value=Path("document.pdf")) as download:
                operation(item)
            self.assertEqual(download.call_args.kwargs["bucket"], shared.PDF_PAGES_BUCKET)

    def test_config_rejects_gaps_overlaps_duplicate_accounts_and_bounded_tail(self):
        variants = []
        for field, value in [("min_bytes", 104857601), ("min_bytes", 104857599), ("owner", "VOMEBOOK")]:
            config = copy.deepcopy(self.config)
            config["native_lanes"][1][field] = value
            variants.append(config)
        config = copy.deepcopy(self.config)
        config["native_lanes"][-1]["max_bytes"] = 2**40
        variants.append(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lanes.json"
            for config in variants:
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    lanes.load_config(path)

    def test_native_ranges_can_be_split_for_more_accounts(self):
        config = copy.deepcopy(self.config)
        config["native_lanes"] = [
            {"name": f"size-{index}", "owner": f"account-{index}", "min_bytes": index * 10,
             "max_bytes": (index + 1) * 10 if index < 6 else None} for index in range(7)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lanes.json"
            path.write_text(json.dumps(config))
            loaded = lanes.load_config(path)
        for size in range(1, 100):
            self.assertEqual(lanes.lane_for(loaded, self.item(size=size)), f"size-{min(size // 10, 6)}")

    def test_central_publisher_rejects_foreign_sources_and_config_drift(self):
        selected = lanes.select_records(self.config, [self.item()], "native-small", "vomebook")
        queue = {"version": 1, "kind": "pdf-render-queue", "worker_owner": "vomebook",
                 "worker_lane": "native-small", "worker_config": lanes.config_identity(self.config),
                 "books": selected, "shards": [{"records": selected}]}
        lanes.validate_queue(self.config, queue, "vomebook", "render")
        variants = [dict(queue, worker_config="old"), dict(queue, worker_owner="anftm"),
                    dict(queue, books=[self.item(size=2**40, worker_lane="native-small")])]
        for variant in variants:
            with self.assertRaises(ValueError):
                lanes.validate_queue(self.config, variant, "vomebook", "render")

    def test_compact_time_weighted_tasks_retain_book_ownership(self):
        item = self.item(page_count=2, source_sha256="a" * 64, profile="profile")
        selected = lanes.select_records(self.config, [item], "native-small", "vomebook")
        queue = stages.plan_render_ranges({"version": 1, "kind": "pdf-render-queue",
            "shards": [{"records": selected}]}, {}, 1500)
        queue.update(worker_owner="vomebook", worker_lane="native-small",
                     worker_config=lanes.config_identity(self.config))
        self.assertNotIn("worker_lane", queue["shards"][0]["records"][0])
        lanes.validate_queue(self.config, queue, "vomebook", "render")

    def test_repaired_native_pdf_retains_original_size_before_source_exclusion(self):
        upstream = self.item(size=104857600)
        generated = self.item(size=1024, source_kind="generated")
        with patch.object(pdf_ocr.pdf_assets, "load_records", return_value=[upstream]), \
                patch.object(pdf_ocr.pdf_assets, "load_generated_records", return_value=[generated]), \
                patch.object(pdf_ocr.reader_assets, "known_gbk_pdf", return_value=True):
            records = pdf_ocr.source_records(Path("search.json"), Path("revisions.json"), {"files": {"x": {}}})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["original_source_bytes"], 104857600)
        self.assertEqual(lanes.lane_for(self.config, records[0]), "native-medium")

    def test_both_planners_filter_before_expensive_work(self):
        records = [self.item(), self.item("large.pdf", 2**40),
                   self.item("scan.caj", 1024, source_kind="generated")]
        with tempfile.TemporaryDirectory() as temporary:
            for stage in ("plan-render", "plan-ocr"):
                output = Path(temporary) / stage
                queue_path = output / "queue.json"
                plan = Mock(return_value={"version": 1, "kind": "pdf-image-ocr-queue", "shards": [],
                                          "shard_count": 0, "shard_ids": [], "books": []})
                def registry(_api, _repo, name, *args):
                    return {"files": {item["key"]: item for item in records}
                            if name == stages.RENDER_REGISTRY else {}}
                argv = ["pdf_ocr_stages.py", stage, "--worker-lane", "native-small",
                        "--worker-owner", "vomebook", "--queue", str(queue_path), "--output", str(output)]
                with patch.object(sys, "argv", argv), patch.object(stages, "HfApi"), \
                        patch.object(stages, "load_registry", side_effect=registry), \
                        patch.object(pdf_ocr, "source_records", return_value=records), \
                        patch.object(stages.plan_pdf_ocr, "plan", plan), \
                        patch.object(stages, "plan_render_ranges", side_effect=lambda queue, *args: queue), \
                        patch.object(stages, "plan_images", plan):
                    self.assertEqual(stages.main(), 0)
                selected = plan.call_args.args[0]
                if isinstance(selected, dict):
                    selected = list(selected.values())
                self.assertEqual([item["key"] for item in selected], [records[0]["key"]])
                self.assertEqual(selected[0]["worker_lane"], "native-small")
                queue = json.loads(queue_path.read_text())
                lanes.validate_queue(self.config, queue, "vomebook", stage.removeprefix("plan-"))

    def test_run_validation_rejects_other_workflows_branches_and_unfinished_jobs(self):
        repository = lanes.repository_for_owner(self.config, "vomebook")
        run = {"repository": {"full_name": repository}, "head_branch": "main",
               "path": ".github/workflows/pdf-account-worker.yml", "event": "workflow_dispatch",
               "status": "completed", "conclusion": "failure"}
        validate_run(self.config, repository, run)
        for field, value in [("path", ".github/workflows/other.yml"), ("head_branch", "topic"),
                             ("event", "pull_request"), ("status", "in_progress"), ("conclusion", "cancelled")]:
            with self.assertRaises(ValueError):
                validate_run(self.config, repository, {**run, field: value})
        with self.assertRaises(ValueError):
            validate_run(self.config, "unconfigured/pipeline", run)

    def test_worker_cannot_publish_registries_and_publisher_keeps_shared_lock(self):
        root = Path(__file__).resolve().parents[1]
        worker = yaml.safe_load((root / ".github/workflows/pdf-account-worker.yml").read_text())
        publisher = yaml.safe_load((root / ".github/workflows/pdf-account-publish.yml").read_text())
        worker_commands = "\n".join(step.get("run", "") for job in worker["jobs"].values() for step in job["steps"])
        self.assertNotIn("publish-render", worker_commands)
        self.assertNotIn("publish-ocr", worker_commands)
        self.assertNotIn("publish_search_reader_index.py", worker_commands)
        self.assertEqual(worker["jobs"]["build"]["strategy"]["max-parallel"], 4)
        self.assertEqual(publisher["concurrency"]["group"], "reader-sidecar")
        for workflow in (worker, publisher):
            for job in workflow["jobs"].values():
                for step in job["steps"]:
                    if "run" in step:
                        subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
