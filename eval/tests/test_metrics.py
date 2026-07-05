import unittest

import numpy as np

from attention_eval import metrics


def blob_map(shape=(100, 100), center=(30, 40), sigma=5.0):
    height, width = shape
    y, x = np.mgrid[0:height, 0:width]
    cx, cy = center
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))


class TestPointMetrics(unittest.TestCase):
    def test_nss_high_at_blob_center(self):
        sal = blob_map()
        self.assertGreater(metrics.nss(sal, [(30, 40)]), 3.0)

    def test_nss_negative_far_from_blob(self):
        sal = blob_map()
        self.assertLess(metrics.nss(sal, [(90, 90)]), 0.0)

    def test_nss_flat_map_is_zero(self):
        self.assertEqual(metrics.nss(np.ones((50, 50)), [(10, 10)]), 0.0)

    def test_nss_out_of_bounds_fixations_ignored(self):
        self.assertTrue(np.isnan(metrics.nss(blob_map(), [(-5, 10), (200, 10)])))

    def test_auc_judd_perfect_for_blob_fixation(self):
        sal = blob_map()
        self.assertGreater(metrics.auc_judd(sal, [(30, 40)]), 0.95)

    def test_auc_judd_chance_for_flat_map_multiple_fixations(self):
        sal = np.linspace(0, 1, 100 * 100).reshape(100, 100)
        rng = np.random.default_rng(42)
        fixations = [(int(x), int(y)) for x, y in rng.uniform(0, 99, size=(200, 2))]
        auc = metrics.auc_judd(sal, fixations)
        self.assertAlmostEqual(auc, 0.5, delta=0.1)


class TestMapMetrics(unittest.TestCase):
    def test_cc_identity(self):
        sal = blob_map()
        self.assertAlmostEqual(metrics.cc(sal, sal), 1.0, places=5)

    def test_cc_anticorrelated(self):
        sal = blob_map()
        self.assertLess(metrics.cc(sal, 1.0 - sal), -0.99)

    def test_sim_identity_and_disjoint(self):
        a = blob_map(center=(20, 20))
        b = blob_map(center=(80, 80))
        self.assertAlmostEqual(metrics.sim(a, a), 1.0, places=5)
        self.assertLess(metrics.sim(a, b), 0.05)

    def test_kl_zero_for_identical(self):
        sal = blob_map()
        self.assertAlmostEqual(metrics.kl_div(sal, sal), 0.0, places=3)

    def test_kl_positive_for_different(self):
        a = blob_map(center=(20, 20))
        b = blob_map(center=(80, 80))
        self.assertGreater(metrics.kl_div(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
