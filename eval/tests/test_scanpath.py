import math
import unittest

from attention_eval import scanpath


class TestMatchStats(unittest.TestCase):
    def test_identical_paths(self):
        path = [(10, 10), (50, 50), (90, 20)]
        stats = scanpath.match_stats(path, path)
        self.assertEqual(stats["matched_fraction"], 1.0)
        self.assertEqual(stats["mean_distance"], 0.0)
        self.assertEqual(stats["mean_rank_shift"], 0.0)

    def test_shifted_beyond_tolerance(self):
        ref = [(10, 10), (50, 50)]
        other = [(200, 200), (250, 250)]
        stats = scanpath.match_stats(ref, other, pos_tol=20)
        self.assertEqual(stats["matched_fraction"], 0.0)

    def test_order_swap_detected_as_rank_shift(self):
        ref = [(10, 10), (90, 90)]
        other = [(90, 90), (10, 10)]
        stats = scanpath.match_stats(ref, other, pos_tol=5)
        self.assertEqual(stats["matched_fraction"], 1.0)
        self.assertEqual(stats["mean_rank_shift"], 1.0)


class TestLevenshtein(unittest.TestCase):
    def test_equal_sequences(self):
        self.assertEqual(scanpath.levenshtein([1, 2, 3], [1, 2, 3]), 0)

    def test_single_substitution(self):
        self.assertEqual(scanpath.levenshtein([1, 2, 3], [1, 9, 3]), 1)

    def test_length_difference(self):
        self.assertEqual(scanpath.levenshtein([1, 2], [1, 2, 3, 4]), 2)

    def test_gridded_identical(self):
        path = [(10, 10), (90, 90)]
        self.assertEqual(scanpath.gridded_levenshtein(path, path, (100, 100)), 0.0)

    def test_gridded_opposite_corners(self):
        a = [(5, 5)]
        b = [(95, 95)]
        self.assertEqual(scanpath.gridded_levenshtein(a, b, (100, 100)), 1.0)


class TestMultiMatch(unittest.TestCase):
    size = (100, 100)

    def test_identical_is_one(self):
        path = [(10, 10), (30, 30), (50, 20), (70, 60)]
        mm = scanpath.multimatch(path, path, self.size)
        for dim in ("shape", "direction", "length", "position"):
            self.assertAlmostEqual(mm[dim], 1.0, places=6)

    def test_different_below_identical(self):
        a = [(10, 10), (30, 30), (50, 20), (70, 60)]
        b = [(90, 90), (10, 80), (60, 10), (20, 50)]
        mm = scanpath.multimatch(a, b, self.size)
        self.assertLess(mm["position"], 1.0)
        self.assertLess(mm["direction"], 1.0)

    def test_short_paths_position_only(self):
        mm = scanpath.multimatch([(5, 5)], [(5, 5)], self.size)
        self.assertTrue(math.isnan(mm["shape"]))
        self.assertAlmostEqual(mm["position"], 1.0, places=6)

    def test_empty_is_nan(self):
        mm = scanpath.multimatch([], [(5, 5)], self.size)
        self.assertTrue(math.isnan(mm["position"]))


class TestScanMatch(unittest.TestCase):
    size = (100, 100)

    def test_identical_is_one(self):
        path = [(10, 10), (30, 30), (50, 20), (70, 60)]
        self.assertAlmostEqual(scanpath.scanmatch(path, path, self.size), 1.0, places=6)

    def test_different_below_identical(self):
        a = [(10, 10), (30, 30), (50, 20), (70, 60)]
        b = [(90, 90), (10, 80), (60, 10), (20, 50)]
        self.assertLess(scanpath.scanmatch(a, b, self.size), scanpath.scanmatch(a, a, self.size))

    def test_signed_substitution_penalizes_distance(self):
        # A path that stays in the opposite corner scores below 0.5 (net-negative
        # per-element alignment), unlike a lenient non-negative substitution.
        a = [(5, 5), (10, 10)]
        b = [(95, 95), (90, 90)]
        self.assertLess(scanpath.scanmatch(a, b, self.size), 0.5)

    def test_empty_is_nan(self):
        self.assertTrue(math.isnan(scanpath.scanmatch([], [(1, 1)], self.size)))


if __name__ == "__main__":
    unittest.main()
