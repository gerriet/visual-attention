"""Target diagnostics of the VLM front-end harness (delivered / crop-hit /
fixation rank) and the V*Bench sidecar boxes they read."""

import json
import tempfile
import unittest
from pathlib import Path

from datasets import vstar
import vlm_frontend
from vlm_frontend import (make_view, random_coverage, random_fixations, target_fixation_rank,
                          target_visible, tile_boxes, tiled_fixations)


def view(source_box, scale=1.0):
    return make_view(None, source_box, scale)


class TestTargetVisible(unittest.TestCase):
    def test_every_box_must_be_delivered(self):
        boxes = [(10, 10, 40, 40), (500, 500, 530, 530)]
        self.assertTrue(target_visible([view((0, 0, 100, 100)), view((480, 480, 600, 600))], boxes, 24))
        self.assertFalse(target_visible([view((0, 0, 100, 100))], boxes, 24))

    def test_downsampled_target_is_illegible(self):
        box = [(10, 10, 40, 40)]  # 30 px native
        self.assertTrue(target_visible([view((0, 0, 1000, 1000), 1.0)], box, 24))
        self.assertFalse(target_visible([view((0, 0, 1000, 1000), 0.5)], box, 24))

    def test_no_boxes_is_unknown(self):
        self.assertIsNone(target_visible([view((0, 0, 10, 10))], None, 24))


class TestTargetFixationRank(unittest.TestCase):
    SIZE = (2000, 1500)

    def test_rank_of_first_covering_fixation(self):
        fixations = [(1800, 1300, 1.0), (110, 110, 0.9)]
        self.assertEqual(target_fixation_rank(fixations, [(100, 100, 130, 130)], 224, self.SIZE), 2)

    def test_all_boxes_needed(self):
        fixations = [(110, 110, 1.0), (900, 900, 0.9), (1500, 200, 0.8)]
        boxes = [(100, 100, 130, 130), (1490, 190, 1510, 210)]
        self.assertEqual(target_fixation_rank(fixations, boxes, 224, self.SIZE), 3)

    def test_never_covered(self):
        fixations = [(1800, 1300, 1.0)]
        self.assertIsNone(target_fixation_rank(fixations, [(100, 100, 130, 130)], 224, self.SIZE))


class TestRandomBaseline(unittest.TestCase):
    SIZE = (2000, 1500)

    def test_fixations_are_seeded_by_the_question(self):
        a = random_fixations(self.SIZE, 10, seed=17)
        self.assertEqual(a, random_fixations(self.SIZE, 10, seed=17))
        self.assertNotEqual(a, random_fixations(self.SIZE, 10, seed=18))
        self.assertTrue(all(0 <= x <= 2000 and 0 <= y <= 1500 for x, y, _ in a))

    def test_coverage_is_the_base_rate_of_a_window(self):
        # A 36-px target in a 2000x1500 image: a 336-px window covers it for
        # roughly 300^2 of 3e6 centre positions -> ~3% per fixation.
        small = [(1000, 700, 1036, 736)]
        rates = random_coverage(self.SIZE, small, 336, seed=1)
        self.assertLess(rates["1"], 0.08)
        self.assertLess(rates["1"], rates["3"])
        self.assertLess(rates["3"], rates["10"])
        self.assertLess(rates["10"], 0.45)

    def test_no_boxes_is_unknown(self):
        self.assertIsNone(random_coverage(self.SIZE, None, 336, seed=1))


class TestTiling(unittest.TestCase):
    def test_tiles_cover_the_image_and_overlap(self):
        boxes = tile_boxes((2000, 1500), 2)
        self.assertEqual(len(boxes), 4)
        self.assertEqual((boxes[0][0], boxes[0][1]), (0, 0))
        self.assertEqual((boxes[-1][2], boxes[-1][3]), (2000, 1500))
        self.assertGreater(boxes[0][2], boxes[1][0])  # horizontal neighbours overlap
        self.assertGreater(boxes[0][3], boxes[2][1])  # vertical neighbours overlap

    def test_fixations_merge_by_rank_in_native_coordinates(self):
        class Img:  # the only parts of PIL.Image the merge touches
            size = (2000, 1500)

            def __init__(self, box=None):
                self.box = box

            def crop(self, box):
                return Img(box)

        def fake_emit(binary, image, proc_max_side, config=None, top_down=None):
            if image.box is None:  # the whole image: one fixation
                return [(1000.0, 750.0, 1.0)]
            strong = 0.9 if image.box[0] == 0 and image.box[1] == 0 else 0.5
            # local (10, 10) in every tile, plus a duplicate of the global one
            return [(10.0, 10.0, strong), (1000.0 - image.box[0], 750.0 - image.box[1], 0.4)]

        original = vlm_frontend.emit_fixations
        vlm_frontend.emit_fixations = fake_emit
        try:
            merged = tiled_fixations("bin", Img(), 1024, tiles=2)
        finally:
            vlm_frontend.emit_fixations = original
        self.assertEqual(merged[0][:2], (1000.0, 750.0))   # the gist comes first
        self.assertEqual(merged[1][:2], (10.0, 10.0))      # then the strongest tile
        self.assertEqual(len(merged), 5)                   # 1 global + 4 tile tops; duplicates dropped
        self.assertIn((867.0, 653.0), [m[:2] for m in merged])  # tile offsets applied


class TestVStarBoxes(unittest.TestCase):
    def test_sidecar_bbox_becomes_corner_boxes(self):
        record = {"image": "relative_position/x.jpg", "category": "relative_position",
                  "question_id": 7, "label": "A",
                  "text": "Is the cup on the left or right of the dog?\n(A) left\n(B) right\n"
                          "Answer with the option's letter from the given choices directly."}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relative_position").mkdir()
            (root / "relative_position" / "x.json").write_text(
                json.dumps({"bbox": [[10, 20, 30, 40], [100, 100, 5, 5]]}))
            (root / "test_questions.jsonl").write_text(json.dumps(record) + "\n")
            item = next(vstar.iter_items(root))
        self.assertEqual(item["target_boxes"], [(10, 20, 40, 60), (100, 100, 105, 105)])
        self.assertEqual(item["answer"], "left")


if __name__ == "__main__":
    unittest.main()
