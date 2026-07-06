"""Gaussian centre-bias baseline.

Human fixations cluster near the image centre, so a static central Gaussian is
a deceptively strong fixation-prediction baseline — a standard sanity floor in
saliency benchmarks (a model that cannot beat it has learned nothing spatial).
"""

import numpy as np

from .base import SaliencyModel, normalize01


class CenterBias(SaliencyModel):
    name = "center-bias"

    def __init__(self, sigma_frac=0.35):
        self.sigma_frac = sigma_frac

    def compute(self, image):
        h, w = image.shape[:2]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        sy, sx = self.sigma_frac * h, self.sigma_frac * w
        yy, xx = np.mgrid[0:h, 0:w]
        gauss = np.exp(-(((yy - cy) ** 2) / (2 * sy * sy) + ((xx - cx) ** 2) / (2 * sx * sx)))
        return normalize01(gauss)
