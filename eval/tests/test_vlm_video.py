"""M19 video front-end: identity- vs location-keyed crop selection, the
question set, the legibility oracle, and the generator's --tags / --late."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from vlm_frontend import make_view
from vlm_video import delivered, identity_picks, identity_stats, location_picks, make_questions, object_picks

GENERATOR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "tools", "make_dynamic_scene.py")

# Object 1 is fixated, then moves 300 px; object 2 then arrives where object 1
# was first seen.
SCANPATH = [
    {"frame": 0, "label": 1, "x": 100, "y": 100},
    {"frame": 5, "label": 1, "x": 400, "y": 100},
    {"frame": 10, "label": 2, "x": 110, "y": 105},
]


class TestPicks(unittest.TestCase):
    def test_identity_memory_sends_each_object_once(self):
        self.assertEqual(object_picks(SCANPATH, 5), [(0, 100, 100), (10, 110, 105)])

    def test_location_memory_resends_the_mover_and_skips_the_newcomer(self):
        self.assertEqual(location_picks(SCANPATH, 5, radius=96), [(0, 100, 100), (5, 400, 100)])

    def test_budget_caps_the_crops(self):
        self.assertEqual(len(object_picks(SCANPATH, 1)), 1)

    def test_label_switch_costs_object_files_a_crop_not_a_perfect_tracker(self):
        # One object seen twice under two labels (a label switch), then a
        # fixation on empty background.
        gt = {"objects": [{"id": 0, "positions": [{"frame": f, "x": 100 + 30 * f, "y": 100, "visible": True}
                                                  for f in range(3)]}]}
        path = [{"frame": 0, "label": 1, "x": 100, "y": 100},
                {"frame": 1, "label": 3, "x": 130, "y": 100},
                {"frame": 2, "label": 4, "x": 900, "y": 900}]
        self.assertEqual(len(object_picks(path, 5)), 3)
        self.assertEqual(identity_picks(path, gt, 5, radius=40), [(0, 100, 100)])
        stats = identity_stats(path, gt, radius=40)
        self.assertEqual(stats["labels_per_object"], 2.0)
        self.assertAlmostEqual(stats["off_object"], 1 / 3)


def gt_with(n):
    return {"objects": [{"id": i, "tag": tag, "color": color, "tag_box": [30, 16],
                         "positions": [{"frame": 0, "x": 500, "y": 400, "visible": True}]}
                        for i, (tag, color) in enumerate(zip(["K7", "A3", "M9", "P4", "T6"][:n],
                                                             ["red", "green", "blue", "cyan", "pink"]))]}


class TestQuestions(unittest.TestCase):
    def test_one_question_per_object_four_distinct_choices(self):
        for n in (1, 5):
            questions = make_questions(gt_with(n), seed=3)
            self.assertEqual(len(questions), n)
            for q in questions:
                self.assertEqual(len(set(q["choices"])), 4)
                self.assertIn(q["answer"], q["choices"])


class TestDelivered(unittest.TestCase):
    def test_native_crop_reads_the_code_downsampled_frame_does_not(self):
        obj = gt_with(1)["objects"][0]
        crop = dict(make_view(None, (400, 300, 592, 492), 1.0), frame=0)
        frame = dict(make_view(None, (0, 0, 1280, 960), 0.3), frame=0)
        clipped = dict(make_view(None, (505, 300, 700, 500), 1.0), frame=0)
        self.assertTrue(delivered([crop], obj, 10))
        self.assertFalse(delivered([frame], obj, 10))
        self.assertFalse(delivered([clipped], obj, 10))


class TestGeneratorFlags(unittest.TestCase):
    def generate(self, tmp, *flags):
        subprocess.run([sys.executable, GENERATOR, "--out", tmp, "--frames", "6", "--objects", "3",
                        "--radius", "30", "--tag-size", "14", *flags], check=True, stdout=subprocess.DEVNULL)
        with open(os.path.join(tmp, "gt.json")) as fh:
            return json.load(fh)

    def test_tags_and_late_arrivals(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = self.generate(tmp, "--tags", "--late", "1")
        tags = [o["tag"] for o in gt["objects"]]
        self.assertEqual(len(set(tags)), 3)
        self.assertTrue(all(o["tag_box"][0] > 0 and o["tag_box"][1] > 0 for o in gt["objects"]))
        late = gt["objects"][-1]
        self.assertGreater(late["onset"], 0)
        self.assertFalse(late["positions"][0]["visible"])

    def test_default_scene_unchanged_by_the_new_flags(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain, tagged = self.generate(a), self.generate(b, "--tags")
        self.assertNotIn("tag", plain["objects"][0])
        # Same trajectories with or without tags (a separate random stream).
        self.assertEqual([o["positions"] for o in plain["objects"]],
                         [o["positions"] for o in tagged["objects"]])


if __name__ == "__main__":
    unittest.main()
