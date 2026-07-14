"""Unit tests for the dynamic-IOR scorer (eval/dynamic_ior.py), M12."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dynamic_ior as dio


def gt_two_static(frames=6):
    """Two objects, visible every frame, at fixed well-separated positions."""
    def positions(x, y):
        return [{"frame": f, "x": x, "y": y, "w": 20, "h": 20, "visible": True} for f in range(frames)]
    return {
        "schema": "dynamic-scene-gt/v1", "width": 200, "height": 200, "frames": frames,
        "objects": [{"id": 0, "positions": positions(50, 50)},
                    {"id": 1, "positions": positions(150, 150)}],
    }


class TestScorer(unittest.TestCase):
    def test_perfect_alternation_covers_all_without_waste(self):
        gt = gt_two_static()
        scan = [{"frame": f, "x": (50 if f % 2 == 0 else 150), "y": (50 if f % 2 == 0 else 150)}
                for f in range(6)]
        m = dio.score(gt, scan, 28.0)
        self.assertEqual(m["coverage"], 1.0)
        self.assertEqual(m["revisit_waste"], 0.0)      # nothing uncovered while revisiting
        self.assertEqual(m["perseveration"], 0.0)      # never the same object twice in a row

    def test_perseveration_wastes_revisits_and_penalizes_latency(self):
        gt = gt_two_static()
        scan = [{"frame": f, "x": 50, "y": 50} for f in range(6)]  # always object 0
        m = dio.score(gt, scan, 28.0)
        self.assertEqual(m["coverage"], 0.5)                  # object 1 never attended
        self.assertAlmostEqual(m["revisit_waste"], 5 / 6, places=3)
        self.assertAlmostEqual(m["perseveration"], 5 / 6, places=3)
        self.assertAlmostEqual(m["mean_latency"], 3.0, places=3)  # (0 + full-penalty 6) / 2

    def test_focus_far_from_any_object_is_not_counted(self):
        gt = gt_two_static()
        m = dio.score(gt, [{"frame": 0, "x": 5, "y": 5}], 28.0)
        self.assertEqual(m["coverage"], 0.0)

    def test_occluded_object_is_not_matchable(self):
        gt = gt_two_static()
        gt["objects"][1]["positions"][0]["visible"] = False
        # focus on object 1's location while it is occluded -> no match
        m = dio.score(gt, [{"frame": 0, "x": 150, "y": 150}], 28.0)
        self.assertEqual(m["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
