"""Unit tests for the saliency->scanpath readouts (eval/attention_eval/readout.py), M11."""

import unittest

import numpy as np

from attention_eval import readout


def _two_peak_map(h=60, w=80):
    m = np.zeros((h, w))
    for (r, c, a) in [(10, 20, 1.0), (45, 60, 0.8)]:
        yy, xx = np.mgrid[0:h, 0:w]
        m += a * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 16))
    return m


class TestWtaIor(unittest.TestCase):
    def test_visits_peaks_in_order(self):
        m = _two_peak_map()
        fx = readout.wta_ior(m, size=(160, 120), n=2, ior_frac=0.1)
        self.assertEqual(len(fx), 2)
        # peak1 (col20,row10)->(41,21), peak2 (col60,row45)->(121,91) at 2x scale
        self.assertLess(abs(fx[0][0] - 41), 15)
        self.assertLess(abs(fx[1][0] - 121), 15)

    def test_ior_moves_focus_away(self):
        m = _two_peak_map()
        fx = readout.wta_ior(m, n=3, ior_frac=0.1)
        # No two fixations coincide (IOR suppressed the previous peak).
        self.assertNotEqual(fx[0], fx[1])

    def test_flat_map_stops_early(self):
        self.assertEqual(readout.wta_ior(np.zeros((10, 10)), n=5), [])


class TestStochastic(unittest.TestCase):
    def test_reproducible_under_seed(self):
        m = _two_peak_map()
        a = readout.sample_paths(m, seed=7, count=3, n=3)
        b = readout.sample_paths(m, seed=7, count=3, n=3)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        m = _two_peak_map()
        a = readout.sample_paths(m, seed=1, count=1, n=4)
        b = readout.sample_paths(m, seed=2, count=1, n=4)
        self.assertNotEqual(a, b)

    def test_flat_map_samples_uniformly(self):
        # Proportional-to-saliency sampling on a flat map is ~uniform
        # (regression: a min-subtraction would zero a flat map and break this).
        flat = np.ones((20, 20))
        xs = [p[0][0] for p in readout.sample_paths(flat, seed=3, count=300, size=(20, 20), n=1)]
        self.assertTrue(6 < float(np.mean(xs)) < 14)


if __name__ == "__main__":
    unittest.main()
