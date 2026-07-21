"""Turn any saliency map into a fixation sequence (roadmap M11, H4).

A generic winner-take-all + inhibition-of-return readout makes every saliency
map — the thesis map, the six alternative operators, the spectral/center
baselines, DeepGaze — a *scanpath* model and therefore a fair peer against
human fixation sequences. It also isolates H4's stage-2 ordering effect: the
same saliency map read two ways (this WTA readout vs the object-file
readout of `--attend`) shows how much of any scanpath agreement comes from the
map versus from the object-based second stage.

  wta_ior     deterministic: repeatedly take the argmax, then suppress a
              Gaussian neighbourhood (inhibition of return).
  stochastic  sample a fixation proportional to saliency (with the same IOR),
              seeded — the generative variability peer, so the deterministic
              path is scored as one draw of a distribution (best-of-N).

Coordinates come out in the requested (w, h) image space, so a map computed at
any resolution scores directly against image-pixel human fixations.
"""

import numpy as np


def _prepare(saliency):
    """Non-negative float saliency, so argmax/sampling are well defined.
    Clamps negatives to 0 (NOT a min-subtraction, which would distort the
    sampling distribution) and scales by the peak — a constant factor that
    leaves the sampling *distribution* proportional to the saliency itself.
    Returns a fresh array (never a view of the caller's map)."""
    m = np.maximum(np.asarray(saliency, dtype=np.float64), 0.0)
    peak = m.max()
    return m / peak if peak > 0 else m


def _inhibit(m, row, col, radius, strength):
    """Subtract a Gaussian bump centred at (row, col) — inhibition of return."""
    h, w = m.shape
    r = int(radius * 3) + 1
    r0, r1 = max(0, row - r), min(h, row + r + 1)
    c0, c1 = max(0, col - r), min(w, col + r + 1)
    ys = np.arange(r0, r1)[:, None]
    xs = np.arange(c0, c1)[None, :]
    bump = strength * np.exp(-((ys - row) ** 2 + (xs - col) ** 2) / (2.0 * radius * radius))
    m[r0:r1, c0:c1] = np.maximum(0.0, m[r0:r1, c0:c1] - bump)


def _scale(row, col, shape, size):
    """Map (row, col) in array space to (x, y) in the requested image size."""
    h, w = shape
    out_w, out_h = size if size else (w, h)
    x = (col + 0.5) * out_w / w
    y = (row + 0.5) * out_h / h
    return (x, y)


def wta_ior(saliency, size=None, n=10, ior_frac=0.08, ior_strength=1.0):
    """Deterministic winner-take-all + IOR scanpath: the top `n` peaks, each
    followed by Gaussian inhibition. `ior_frac` is the IOR radius as a fraction
    of the map's shorter side. Returns [(x, y), ...] in `size` coords."""
    m = _prepare(saliency)  # already a fresh array — safe to mutate for IOR
    radius = max(1.0, ior_frac * min(m.shape))
    fixations = []
    for _ in range(n):
        if m.max() <= 0:
            break
        row, col = np.unravel_index(int(np.argmax(m)), m.shape)
        fixations.append(_scale(row, col, m.shape, size))
        _inhibit(m, row, col, radius, ior_strength)
    return fixations


def stochastic(saliency, rng, size=None, n=10, ior_frac=0.08, ior_strength=1.0):
    """Stochastic readout: sample each fixation proportional to the (IOR-
    suppressed) saliency, using the given numpy RandomState. One draw of the
    generative distribution. Inverse-CDF sampling (cumsum + searchsorted)
    avoids rng.choice's per-call probability re-validation over a full-res map."""
    m = _prepare(saliency)
    radius = max(1.0, ior_frac * min(m.shape))
    w = m.shape[1]
    fixations = []
    for _ in range(n):
        flat = m.ravel()
        total = flat.sum()
        if total <= 0:
            break
        idx = int(np.searchsorted(np.cumsum(flat), rng.random() * total))
        idx = min(idx, flat.size - 1)
        row, col = divmod(idx, w)
        fixations.append(_scale(row, col, m.shape, size))
        _inhibit(m, row, col, radius, ior_strength)
    return fixations


def sample_paths(saliency, seed, count, size=None, n=10, ior_frac=0.08, ior_strength=1.0):
    """`count` stochastic scanpaths from one map, for best-of-N / distribution
    scoring. Deterministic given `seed`."""
    rng = np.random.RandomState(seed)
    return [stochastic(saliency, rng, size, n, ior_frac, ior_strength) for _ in range(count)]
