"""M19 on DAVIS: mask-based attribution and delivery, object crops, the
category questions — against a small fake sequence (no dataset needed)."""

import unittest

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None

from vlm_frontend import make_view
import vlm_video_davis as davis


class FakeSequence:
    """Two frames, 200x100: object 1 a 20x20 square, object 2 a small 6x6 one."""

    categories = {1: "dog", 2: "phone"}
    n_frames = 2
    size = (200, 100)

    def objects(self):
        return [1, 2]

    def frame(self, f):
        return Image.new("RGB", self.size)

    def masks(self, f):
        big = np.zeros((100, 200), bool)
        big[40:60, 40 + 10 * f:60 + 10 * f] = True
        small = np.zeros((100, 200), bool)
        small[10:16, 150:156] = True
        return {1: big, 2: small}


def view(box, scale, frame=0):
    return dict(make_view(None, box, scale), frame=frame)


@unittest.skipIf(np is None, "needs numpy + PIL (eval venv)")
class TestMasks(unittest.TestCase):
    def setUp(self):
        self.seq = FakeSequence()

    def test_attribution_inside_and_near_a_mask(self):
        self.assertEqual(davis.attributed_object(self.seq, 0, 50, 50, 5), 1)
        self.assertEqual(davis.attributed_object(self.seq, 0, 63, 50, 5), 1)  # 4 px off the edge
        self.assertIsNone(davis.attributed_object(self.seq, 0, 100, 90, 5))

    def test_delivery_needs_most_of_the_mask_at_a_readable_size(self):
        self.assertTrue(davis.view_shows(self.seq, view((30, 30, 80, 80), 1.0), 1, 12))
        self.assertFalse(davis.view_shows(self.seq, view((30, 30, 80, 80), 0.5), 1, 12))  # 10 px on screen
        self.assertFalse(davis.view_shows(self.seq, view((50, 30, 80, 80), 1.0), 1, 12))  # half the mask
        self.assertFalse(davis.view_shows(self.seq, view((0, 0, 200, 100), 0.5), 2, 12))  # small object shrunk

    def test_perfect_tracker_dedup_and_identity_stats(self):
        path = [{"frame": 0, "label": 1, "x": 50, "y": 50},
                {"frame": 1, "label": 7, "x": 60, "y": 50},    # same dog, new label
                {"frame": 1, "label": 8, "x": 152, "y": 12},   # the phone
                {"frame": 1, "label": 9, "x": 100, "y": 90}]   # nothing
        self.assertEqual([p["label"] for p in davis.first_per_object(self.seq, path, 5, 5)], [1, 8])
        stats = davis.identity_stats(self.seq, path, 5)
        self.assertEqual(stats["coverage"], 1.0)
        self.assertEqual(stats["labels_per_object"], 1.5)
        self.assertEqual(stats["off_object"], 0.25)


@unittest.skipIf(np is None, "needs numpy + PIL (eval venv)")
class TestCropsAndQuestions(unittest.TestCase):
    def test_box_crop_covers_the_box_and_caps_the_size(self):
        image = Image.new("RGB", (800, 480))
        small = davis.box_crop_view(image, 0, (100, 100, 140, 140), side=224)
        self.assertEqual(small["image"].size, (224, 224))
        self.assertEqual(small["scale"], 1.0)
        big = davis.box_crop_view(image, 0, (100, 50, 500, 450), side=224)
        self.assertLessEqual(max(big["image"].size), 224)
        x0, y0, x1, y1 = big["source_box"]
        self.assertTrue(x0 <= 100 and x1 >= 500 and y0 <= 50 and y1 >= 450)

    def test_one_question_per_present_category_with_absent_distractors(self):
        vocab = ["dog", "phone", "car", "cow", "pig", "kite"]
        questions = davis.make_questions({1: "dog", 2: "dog", 3: "phone"}, vocab, seed=1)
        self.assertEqual([q["category"] for q in questions], ["dog", "phone"])
        self.assertEqual(questions[0]["objects"], [1, 2])
        for q in questions:
            self.assertEqual(len(set(q["choices"])), 4)
            self.assertEqual(sum(c in ("dog", "phone") for c in q["choices"]), 1)


class TestResolution(unittest.TestCase):
    def test_paths_follow_the_resolution(self):
        import tempfile
        from pathlib import Path

        from datasets import davis2017
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part in ("JPEGImages", "Annotations"):
                (root / part / "Full-Resolution" / "judo").mkdir(parents=True)
            self.assertTrue(davis2017.available(root, "Full-Resolution"))
            self.assertFalse(davis2017.available(root))  # 480p not unpacked here
            self.assertEqual(davis2017.frames_dir("judo", root, "Full-Resolution").name, "judo")
            self.assertIn("Full-Resolution", str(davis2017.frames_dir("judo", root, "Full-Resolution")))
            self.assertIn("480p", str(davis2017.frames_dir("judo", root)))


if __name__ == "__main__":
    unittest.main()
