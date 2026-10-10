"""Text contracts exercise real offsets/geometry, not recognition accuracy."""

import copy
import unittest

from scripts import ocr_layout, pdf_text_layer as layer


def raw_page(text="English", confidence=.99, mode="horizontal-ltr"):
    block = {"t": text, "b": [.1, .1, .8, .2], "c": confidence, "s": "ocr"}
    return {"version": 1, "kind": "pdf-ocr-page", "page": 1, "source": "ocr",
            "width": 100, "height": 200,
            **ocr_layout.arrange([block] if text else [], 100, 200, {"writing_mode": mode},
                                  include_writing_modes=True)}


class TextLayerTests(unittest.TestCase):
    def test_explicit_order_acceptance_preserves_regions_and_rejects_unpositioned_text(self):
        blocks = [{"t": "First", "b": [.1, .1, .4, .2], "c": .8, "s": "ocr"},
                  {"t": "Second", "b": [.6, .1, .9, .2], "c": .8, "s": "ocr"}]
        raw = {**raw_page(), **ocr_layout.arrange(blocks, 100, 200, include_writing_modes=True)}
        baseline = layer.from_page(raw, "a" * 64)
        order = [r["id"] for r in reversed(baseline["regions"])]
        proposal = {"version": 1, "kind": "pdf-text-correction", "base_generation": baseline["generation"],
                    "raw_sha256": baseline["raw_sha256"], "page_identity": baseline["page_identity"],
                    "replacements": [], "region_order": order}
        accepted = layer.accept_proposal(baseline, proposal, actor="reviewer")
        self.assertEqual(accepted["text"], "Second\nFirst")
        self.assertEqual([r["id"] for r in accepted["regions"]], order)
        self.assertEqual(accepted["regions"][0]["quad"], baseline["regions"][1]["quad"])
        self.assertEqual(baseline["revision"], "raw")
        for bad in ([order[0]], [order[0]] * 2, ["b999", order[0]]):
            with self.assertRaises(ValueError):
                layer.accept_proposal(baseline, {**proposal, "region_order": bad}, actor="reviewer")
        unmapped = copy.deepcopy(baseline)
        unmapped["text"] += "unpositioned"
        unmapped["generation"] = layer.digest({k: v for k, v in unmapped.items() if k != "generation"})
        with self.assertRaisesRegex(ValueError, "unpositioned"):
            layer.accept_proposal(unmapped, {**proposal, "base_generation": unmapped["generation"]}, actor="reviewer")

    def test_legacy_page_is_deterministic_and_does_not_mutate_raw(self):
        raw = raw_page()
        before = copy.deepcopy(raw)
        first = layer.from_page(raw, "a" * 64, "en")
        self.assertEqual(first, layer.from_page(raw, "a" * 64, "en"))
        self.assertEqual(raw, before)
        self.assertEqual(first["quality"], "unreviewed")
        self.assertEqual(first["regions"][0]["quad"], [[.1, .1], [.8, .1], [.8, .2], [.1, .2]])

    def test_arabic_hebrew_and_mixed_text_keep_logical_string(self):
        for text, expected in (("كتاب", "rtl"), ("שלום", "rtl"), ("كتاب test 123", "mixed")):
            with self.subTest(text=text):
                result = layer.from_page(raw_page(text, mode="horizontal-rtl"), "a" * 64)
                self.assertEqual(result["text"], text)
                self.assertEqual(result["regions"][0]["direction"], expected)
                self.assertIn("bidi-order-needs-review", result["review_flags"])

    def test_unicode_codepoint_offsets_include_non_bmp_and_combining_text(self):
        text = "A\U00020000e\u0301"
        result = layer.from_page(raw_page(text), "a" * 64)
        self.assertEqual(result["regions"][0]["end"], 4)
        self.assertEqual(result["text"], text)

    def test_mixed_region_writing_modes_survive_page_default(self):
        blocks = [{"t": "Header", "b": [.1, .01, .9, .06], "c": 1, "s": "ocr"},
                  {"t": "竖排正文", "b": [.7, .2, .75, .8], "c": 1, "s": "ocr"}]
        raw = {**raw_page(), **ocr_layout.arrange(blocks, 100, 200, {"regions": [
            {"box": [0, 0, 1, .1], "writing_mode": "horizontal-ltr"},
            {"box": [0, .1, 1, 1], "writing_mode": "vertical-rl"}]}, include_writing_modes=True)}
        result = layer.from_page(raw, "a" * 64)
        self.assertEqual([r["writing_mode"] for r in result["regions"]], ["horizontal-ltr", "vertical-rl"])

    def test_degenerate_or_out_of_bounds_quads_are_rejected(self):
        baseline = layer.from_page(raw_page(), "a" * 64)
        for quad in ([[.1, .1], [.2, .2], [.3, .3], [.4, .4]],
                     [[-.1, .1], [.8, .1], [.8, .2], [.1, .2]]):
            bad = copy.deepcopy(baseline)
            bad["regions"][0]["quad"] = quad
            bad["generation"] = layer.digest({k: v for k, v in bad.items() if k != "generation"})
            with self.assertRaises(ValueError):
                layer.validate(bad)

    def test_legacy_spans_get_region_modes_without_changing_raw_object(self):
        raw = raw_page("Text")
        raw["text_spans"][0].pop("writing_mode")
        before = copy.deepcopy(raw)
        result = layer.from_page(raw, "a" * 64, layout_options={"regions": [
            {"box": [0, 0, 1, 1], "writing_mode": "vertical-rl"}]})
        self.assertEqual(result["regions"][0]["writing_mode"], "vertical-rl")
        self.assertEqual(raw, before)

    def test_empty_recognition_is_not_claimed_to_be_blank(self):
        result = layer.from_page(raw_page(""), "a" * 64)
        self.assertEqual(result["processing"], "processed-empty")
        self.assertIn("empty-recognition-unverified", result["review_flags"])
        self.assertEqual(result["quality"], "unreviewed")

    def test_mismatched_native_text_does_not_get_false_offsets(self):
        raw = raw_page("native words")
        raw.update(source="native", text="different extractor order")
        result = layer.from_page(raw, "a" * 64)
        self.assertEqual(result["regions"], [])
        self.assertIn("text-span-mismatch", result["review_flags"])
        self.assertEqual(result["text"], "different extractor order")

    def test_low_confidence_and_bad_geometry_are_reviewed(self):
        raw = raw_page("Text", .4)
        self.assertIn("low-recognition-confidence", layer.from_page(raw, "a" * 64)["review_flags"])
        raw["text_spans"][0]["box"] = [0, 0, float("nan"), 1]
        # Nonfinite evidence is not allowed into content-addressed output.
        with self.assertRaises(ValueError):
            layer.from_page(raw, "a" * 64)

    def test_generation_and_geometry_tampering_are_rejected(self):
        raw = layer.from_page(raw_page(), "a" * 64)
        for change in ({"text": "altered"}, {"page": 2}, {"source_sha256": "b" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                layer.validate({**raw, **change})

    def test_explicit_acceptance_updates_offsets_without_rewriting_raw(self):
        blocks = [{"t": "First", "b": [.1, .1, .5, .2], "c": 1, "s": "ocr"},
                  {"t": "Second", "b": [.1, .3, .5, .4], "c": 1, "s": "ocr"}]
        raw = {**raw_page(), **ocr_layout.arrange(blocks, 100, 200, {"writing_mode": "horizontal-ltr"})}
        original = layer.from_page(raw, "a" * 64)
        proposal = {"kind": "pdf-text-correction", "version": 1,
                    "base_generation": original["generation"], "raw_sha256": original["raw_sha256"],
                    "page_identity": original["page_identity"],
                    "replacements": [{"region_id": "b0", "before": "First", "after": "First corrected"}]}
        accepted = layer.accept_proposal(original, proposal, actor="reviewer")
        self.assertEqual(original["text"], "First\n\nSecond")
        self.assertEqual(accepted["text"], "First corrected\n\nSecond")
        self.assertEqual(accepted["raw_sha256"], original["raw_sha256"])
        self.assertEqual(accepted["quality"], "partially-reviewed")
        self.assertEqual(accepted["regions"][0]["mapping_precision"], "corrected-block")
        for region in accepted["regions"]:
            self.assertEqual(accepted["text"][region["start"]:region["end"]], region["text"])
        with self.assertRaises(ValueError):
            layer.accept_proposal(accepted, proposal, actor="reviewer")

    def test_unknown_duplicate_and_stale_correction_regions_are_rejected(self):
        original = layer.from_page(raw_page(), "a" * 64)
        proposal = {"kind": "pdf-text-correction", "version": 1,
                    "base_generation": original["generation"], "raw_sha256": original["raw_sha256"],
                    "page_identity": original["page_identity"],
                    "replacements": [{"region_id": "b0", "before": "English", "after": "Correct"}]}
        for replacement in ([{"region_id": "missing", "before": "English", "after": "Correct"}],
                            proposal["replacements"] * 2,
                            [{"region_id": "b0", "before": "wrong", "after": "Correct"}]):
            with self.assertRaises(ValueError):
                layer.accept_proposal(original, {**proposal, "replacements": replacement}, actor="reviewer")


if __name__ == "__main__":
    unittest.main()
