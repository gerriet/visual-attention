"""Unit tests for the MIT1003 I-DT fixation extraction (datasets/mit1003.py), M11.

Tests the fixation-detection logic directly (no scipy, no dataset). The .mat
parser is exercised separately where scipy is available."""

import unittest

import numpy as np

from datasets import mit1003


def _cluster(cx, cy, n=60, jitter=2.0, seed=0):
    rng = np.random.RandomState(seed)
    return np.tile([cx, cy], (n, 1)) + rng.randn(n, 2) * jitter


class TestIDT(unittest.TestCase):
    size = (1024, 768)

    def test_two_dwells_with_saccade(self):
        samples = np.vstack([_cluster(300, 200, seed=1),
                             np.linspace([300, 200], [700, 550], 12),
                             _cluster(700, 550, seed=2)])
        fx = mit1003.idt_fixations(samples, self.size)
        self.assertEqual(len(fx), 2)
        self.assertLess(abs(fx[0][0] - 300), 8)
        self.assertLess(abs(fx[1][0] - 700), 8)

    def test_blink_separates_nearby_fixations(self):
        # Two nearby dwells split by a blink (NaN) must NOT merge — deleting the
        # blink samples and treating survivors as contiguous once fused them.
        samples = np.vstack([_cluster(300, 200, seed=1),
                             np.full((15, 2), np.nan),
                             _cluster(320, 210, seed=2)])
        self.assertEqual(len(mit1003.idt_fixations(samples, self.size)), 2)

    def test_offscreen_excursion_separates(self):
        samples = np.vstack([_cluster(300, 200, seed=1),
                             np.tile([5000, 5000], (15, 1)),  # off-screen
                             _cluster(320, 210, seed=2)])
        self.assertEqual(len(mit1003.idt_fixations(samples, self.size)), 2)

    def test_pure_saccade_yields_nothing(self):
        samples = np.linspace([0, 0], [1000, 700], 40)
        self.assertEqual(mit1003.idt_fixations(samples, self.size), [])


if __name__ == "__main__":
    unittest.main()
