"""Geometry fixtures exercise ordering, not claims about model OCR accuracy."""

import unittest

from scripts import ocr_layout as layout
from scripts import pdf_ocr


def block(text, box):
    return {"t": text, "b": box, "c": .99, "s": "ocr"}


class OcrLayoutTests(unittest.TestCase):
    def test_vertical_columns_are_right_to_left_and_strings_are_not_reversed(self):
        blocks = [block("左列文字", [.1, .1, .15, .7]), block("右列文字", [.7, .1, .75, .7])]
        result = layout.arrange(blocks, 1000, 1000)
        self.assertEqual(result["text"], "右列文字\n\n左列文字")
        self.assertEqual(result["layout"]["writing_mode"], "vertical-rl")
        self.assertEqual(result["raw_blocks"], blocks)
        self.assertEqual(result["blocks"][0]["id"], 1)

    def test_vertical_individual_glyphs_group_by_column(self):
        blocks = [block("甲", [.7, .1, .74, .14]), block("乙", [.7, .15, .74, .19]),
                  block("丙", [.1, .1, .14, .14]), block("丁", [.1, .15, .14, .19])]
        result = layout.arrange(blocks, 1000, 1000, {"writing_mode": "vertical-rl"})
        self.assertEqual(result["text"], "甲乙\n\n丙丁")
        self.assertEqual([x["precision"] for x in result["text_spans"]], ["block"] * 4)

    def test_rtl_heading_reorders_glyph_blocks_but_never_reverses_line_string(self):
        glyphs = [block("民", [.1, .1, .15, .15]), block("人", [.2, .1, .25, .15])]
        result = layout.arrange(glyphs, 1000, 1000, {"writing_mode": "horizontal-rtl"})
        self.assertEqual([b["t"] for b in result["blocks"]], ["人", "民"])
        self.assertEqual(result["text"], "人民")
        result = layout.arrange([block("人民", [.1, .1, .4, .15])], 1000, 1000,
                                {"writing_mode": "horizontal-rtl"})
        self.assertEqual(result["text"], "人民")
        self.assertIn("within-block-character-order-unverified", result["layout"]["review"])

    def test_two_columns_do_not_interleave_rows(self):
        blocks = [block("左栏第一行", [.1, .1, .4, .14]), block("右栏第一行", [.6, .1, .9, .14]),
                  block("左栏第二行", [.1, .16, .4, .2]), block("右栏第二行", [.6, .16, .9, .2])]
        result = layout.arrange(blocks, 1000, 1000)
        self.assertEqual([b["t"] for b in result["blocks"]], ["左栏第一行", "左栏第二行", "右栏第一行", "右栏第二行"])
        self.assertIn("\n\n", result["text"])

    def test_spanning_title_then_two_columns(self):
        blocks = [block("通栏标题", [.1, .02, .9, .06]), block("左栏上", [.1, .15, .4, .19]),
                  block("右栏上", [.6, .15, .9, .19]), block("左栏下", [.1, .22, .4, .26]),
                  block("右栏下", [.6, .22, .9, .26])]
        result = layout.arrange(blocks, 1000, 1000)
        self.assertEqual([b["t"] for b in result["blocks"]], ["通栏标题", "左栏上", "左栏下", "右栏上", "右栏下"])

    def test_mixed_regions_use_explicit_order_and_direction(self):
        blocks = [block("横排标题", [.1, .01, .9, .06]), block("竖排左列", [.2, .2, .25, .8]),
                  block("竖排右列", [.7, .2, .75, .8]), block("横排页注", [.1, .9, .9, .95])]
        result = layout.arrange(blocks, 1000, 1000, {"regions": [
            {"box": [0, 0, 1, .1], "writing_mode": "horizontal-ltr"},
            {"box": [0, .1, 1, .85], "writing_mode": "vertical-rl"},
            {"box": [0, .85, 1, 1], "writing_mode": "horizontal-ltr"},
        ]})
        self.assertEqual([b["t"] for b in result["blocks"]], ["横排标题", "竖排右列", "竖排左列", "横排页注"])

    def test_only_geometric_soft_wrap_joins_and_keeps_offset_mapping(self):
        blocks = [block("这里使用手", [.1, .1, .9, .15]), block("机进行阅读", [.1, .16, .9, .21])]
        result = layout.arrange(blocks, 1000, 1000)
        self.assertIn("手机", result["text"])
        for span in result["text_spans"]:
            self.assertEqual(result["text"][span["start"]:span["end"]], blocks[span["block"]]["t"])
        self.assertEqual(result["layout"]["offset_unit"], "unicode-codepoint")
        self.assertIn("soft-line", [b["kind"] for b in result["layout"]["boundaries"]])
        blocks[0]["t"] += "。"
        result = layout.arrange(blocks, 1000, 1000)
        self.assertNotIn("手机", result["text"])
        self.assertIn("手。", result["text"])

    def test_paragraph_indent_and_separate_regions_keep_boundaries(self):
        blocks = [block("这里使用手", [.1, .1, .9, .15]), block("机进行阅读", [.2, .16, .9, .21])]
        self.assertNotIn("手机", layout.arrange(blocks, 1000, 1000)["text"])
        blocks[1]["b"] = [.1, .5, .9, .55]
        self.assertNotIn("手机", layout.arrange(blocks, 1000, 1000)["text"])

    def test_rotation_maps_all_polygon_corners_and_spans_to_original(self):
        original = [.1, .2, .3, .4]
        b = block("旋转文字", original)
        b["q"] = [[.1, .2], [.3, .2], [.3, .4], [.1, .4]]
        arranged = layout.arrange([b], 800, 600)
        restored = layout.restore_coordinates(arranged, 90)
        expected = [.2, .7, .4, .9]
        for x, y in zip(restored["blocks"][0]["b"], expected):
            self.assertAlmostEqual(x, y)
        self.assertEqual(restored["text_spans"][0]["box"], restored["blocks"][0]["b"])
        self.assertEqual(restored["raw_blocks"][0]["q"][0], [.2, .9])

    def test_punctuation_is_never_erased(self):
        result = layout.arrange([block("手。机", [.1, .1, .5, .2])], 1000, 1000)
        self.assertEqual(result["text"], "手。机")

    def test_native_words_on_same_row_preserve_phrases_and_latin_spaces(self):
        blocks = [block("手", [.1, .1, .14, .14]), block("机", [.145, .1, .185, .14]),
                  block("Hello", [.2, .1, .3, .14]), block("world", [.31, .1, .42, .14])]
        result = layout.arrange(blocks, 1000, 1000)
        self.assertEqual(result["text"], "手机 Hello world")
        self.assertEqual([result["text"][s["start"]:s["end"]] for s in result["text_spans"]],
                         [b["t"] for b in blocks])

    def test_punctuation_between_native_blocks_is_never_dropped(self):
        blocks = [block("手。", [.1, .1, .17, .14]), block("机", [.175, .1, .215, .14])]
        text = layout.arrange(blocks, 1000, 1000)["text"]
        self.assertIn("手。机", text)
        self.assertNotIn("手机", text)

    def test_rec_boxes_coordinates_are_not_replaced_by_zero_boxes(self):
        blocks = pdf_ocr.normalize_ocr_result({"rec_texts": ["正文"], "rec_scores": [.9],
                                             "rec_boxes": [[10, 20, 30, 40]]}, 100, 100)
        self.assertEqual(blocks[0]["b"], [.1, .2, .3, .4])
        self.assertEqual(blocks[0]["q"], [[.1, .2], [.3, .2], [.3, .4], [.1, .4]])

    def test_rtl_text_layer_metadata_keeps_unverified_recognizer_order(self):
        from scripts import pdf_text_layer
        for word in ("كتاب", "שלום"):
            result = layout.arrange([block(word, [.1, .1, .4, .2])], 1000, 1000)
            self.assertEqual(result["text"], word)
            page = {"kind": "pdf-ocr-page", "page": 1, "width": 1000, "height": 1000,
                    "source": "ocr", **result}
            layer = pdf_text_layer.from_page(page, "a" * 64)
            self.assertEqual(layer["regions"][0]["direction"], "rtl")
            self.assertIn("bidi-order-needs-review", layer["review_flags"])


if __name__ == "__main__":
    unittest.main()
