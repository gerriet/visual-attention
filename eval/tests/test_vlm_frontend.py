"""Target diagnostics of the VLM front-end harness (delivered / crop-hit /
fixation rank) and the V*Bench sidecar boxes they read."""

import json
import tempfile
import unittest
from pathlib import Path

from datasets import vstar
from vlm_frontend import make_view, target_fixation_rank, target_visible


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
