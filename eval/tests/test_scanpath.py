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


if __name__ == "__main__":
    unittest.main()
