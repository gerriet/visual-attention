"""The shared statistics helpers — the one place a silent bug would poison
every table."""

import unittest

from study_common import bootstrap_ci, paired_bootstrap


class TestBootstrap(unittest.TestCase):
    def test_ci_brackets_the_mean_and_is_deterministic(self):
        values = [0, 1] * 50
        lo, hi = bootstrap_ci(values)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual((lo, hi), bootstrap_ci(values))
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0))

    def test_pairing_finds_a_small_consistent_difference(self):
        # Units differ a lot from each other, arm a is always 0.1 better: two
        # unpaired intervals overlap almost completely, the paired one excludes 0.
        b = [float(i % 10) for i in range(60)]
        a = [v + 0.1 for v in b]
        mean, lo, hi = paired_bootstrap(a, b)
        self.assertAlmostEqual(mean, 0.1)
        self.assertGreater(lo, 0.0)
        a_lo, a_hi = bootstrap_ci(a)
        b_lo, b_hi = bootstrap_ci(b)
        self.assertLess(a_lo, b_hi)  # the unpaired reading would call it a tie

    def test_no_difference_includes_zero(self):
        a = [0, 1, 1, 0, 1, 0, 0, 1] * 10
        b = [1, 0, 1, 0, 0, 1, 1, 0] * 10
        mean, lo, hi = paired_bootstrap(a, b)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
