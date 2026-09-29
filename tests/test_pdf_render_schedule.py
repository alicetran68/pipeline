import copy
import math
import unittest

from scripts import pdf_render_schedule as schedule
from scripts import pdf_ocr_stages as stages


class RenderScheduleTests(unittest.TestCase):
    def book(self, name, pages, seconds):
        return {"key": "repo\0" + name, "source_kind": "upstream", "source_revision": "rev",
                "source_sha256": "a" * 64, "page_count": pages,
                "_render_cost": {"seconds_per_page": seconds, "setup_seconds": 0}}

    def test_time_weights_reduce_slow_book_tail_without_losing_pages(self):
        books = [self.book("slow.pdf", 610, 220 * 60 / 610)]
        books += [self.book(f"fast-{i}.pdf", 500, 14.7 * 60 / 500) for i in range(9)]
        source = {"shards": [{"records": books}]}
        old = stages.plan_render_ranges(source, {})
        new = stages.plan_render_ranges(source, {}, 1500)
        old_loads = [sum(schedule.task_seconds(t) for t in s["records"]) for s in old["shards"]]
        new_loads = [sum(schedule.task_seconds(t) for t in s["records"]) for s in new["shards"]]
        self.assertLess(max(new_loads), max(old_loads) / 3)
        self.assertLess(max(new_loads), 1500 * 1.5)
        self.assertEqual(new, stages.plan_render_ranges(source, {}, 1500))
        for book in books:
            intervals = sorted((t["start"], t["end"]) for shard in new["shards"] for t in shard["records"]
                               if t["key"] == book["key"])
            numbers = [n for first, last in intervals for n in range(first, last + 1)]
            self.assertEqual(numbers, list(range(1, book["page_count"] + 1)))
        self.assertTrue(all(t["end"] - t["start"] + 1 < 250 for shard in new["shards"]
                            for t in shard["records"] if t["key"] == books[0]["key"]))

    def test_matrix_cap_keeps_all_work_and_reports_over_target_load(self):
        book = self.book("huge.pdf", 5000, 100)
        queue = stages.plan_render_ranges({"shards": [{"records": [book]}]}, {}, 100)
        self.assertEqual(queue["shard_count"], 256)
        self.assertEqual(sum(s["page_count"] for s in queue["shards"]), 5000)
        self.assertGreater(max(s["estimated_seconds"] for s in queue["shards"]), 100)

    def test_history_requires_matching_identity_valid_timings_and_saved_range(self):
        book = {**self.book("history.pdf", 1000, 1), "render_profile": stages.render_profile()}
        identity = stages.range_identity(book)
        prior = {**identity, "ranges": {"000001-000250": {}}, "range_timings": {
            "000001-000250": {"page_count": 250, "page_seconds": 5000, "setup_seconds": 10},
            "000251-000500": {"page_count": 250, "page_seconds": 1, "setup_seconds": 0},
            "bad": {"page_count": 1, "page_seconds": 1, "setup_seconds": 0}}}
        cost = schedule.history_cost(identity, prior)
        self.assertEqual((cost["seconds_per_page"], cost["setup_seconds"], cost["samples"]), (20, 10, 1))
        for field in identity:
            changed = {**prior, field: "changed"}
            self.assertIsNone(schedule.history_cost(identity, changed))
        for value in (0, -1, True, "20", math.inf, math.nan, 10 ** 400):
            bad = copy.deepcopy(prior)
            bad["range_timings"]["000001-000250"]["page_seconds"] = value
            self.assertIsNone(schedule.history_cost(identity, bad))
        self.assertIsNone(schedule.history_cost(identity, {**prior, "ranges": None}))
        self.assertIsNone(schedule.history_cost(identity, prior, image_rendered=True))
        prior["range_timings"]["000001-000250"]["image_rendered"] = False
        self.assertIsNotNone(schedule.history_cost(identity, prior, image_rendered=False))

    def test_gap_splitting_preserves_old_and_overlapping_completed_ranges(self):
        self.assertEqual(schedule.missing_ranges(1000, [(1, 250), (200, 500), (751, 1000)], 40),
                         [(501, 540), (541, 580), (581, 620), (621, 660), (661, 700), (701, 740), (741, 750)])
        self.assertEqual(schedule.missing_ranges(500, [(1, 500)], 10), [])
        for key in ("1-250", "000000-000250", "000251-000250", "000001-001001", "x-y", None):
            self.assertIsNone(schedule.parse_range(key, 1000))

    def test_invalid_target_cannot_produce_a_partial_queue(self):
        for value in (0, -1, True, math.inf, math.nan):
            with self.assertRaises(ValueError):
                stages.plan_render_ranges({"shards": []}, {}, value)


if __name__ == "__main__":
    unittest.main()
