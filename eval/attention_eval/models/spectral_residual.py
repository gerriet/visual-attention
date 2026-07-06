"""Spectral Residual saliency (Hou & Zhang, CVPR 2007).

The canonical modern-classical baseline: saliency is the part of the log
amplitude spectrum that departs from its local average, reconstructed back to
the spatial domain and smoothed. numpy-only (FFT + small separable filters).
"""

import numpy as np
from PIL import Image

from .base import SaliencyModel, normalize01


def _to_gray(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    rgb = image.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _resize(img, size_wh):
    im = Image.fromarray(img.astype(np.float32), mode="F").resize(size_wh, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _separable(img, kernel):
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, img)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)


def _box(img, size):
    return _separable(img, np.ones(size, dtype=np.float32) / size)


def _gaussian(img, sigma):
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    return _separable(img, k)


class SpectralResidual(SaliencyModel):
    name = "spectral-residual"

    def __init__(self, work_size=64, blur_sigma=3.0):
        self.work_size = work_size
        self.blur_sigma = blur_sigma

    def compute(self, image):
        gray = _to_gray(image)
        h, w = gray.shape

        # Classic SR operates on a small fixed scale; resize the longer side.
        longer = max(h, w)
        scale = self.work_size / longer
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = _resize(gray, (sw, sh))

        spectrum = np.fft.fft2(small)
        log_amp = np.log(np.abs(spectrum) + 1e-8)
        phase = np.angle(spectrum)
        residual = log_amp - _box(log_amp, 3)

        recon = np.fft.ifft2(np.exp(residual + 1j * phase))
        saliency = np.abs(recon) ** 2
        saliency = _gaussian(saliency, self.blur_sigma)

        saliency = _resize(saliency, (w, h))
        return normalize01(saliency)
