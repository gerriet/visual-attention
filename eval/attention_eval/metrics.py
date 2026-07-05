"""Standard saliency evaluation metrics.

Conventions follow the MIT/Tuebingen saliency benchmark definitions:
NSS and AUC-Judd score a saliency map against discrete fixation points;
CC, SIM and KL compare two (saliency or fixation) maps.
"""

import numpy as np

_EPS = 1e-12


def _as_map(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("expected a 2D map")
    return a


def _fixation_indices(fixations, shape):
    """Valid integer (row, col) indices for (x, y) fixations."""
    rows, cols = [], []
    height, width = shape
    for x, y in fixations:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            rows.append(yi)
            cols.append(xi)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def nss(saliency, fixations):
    """Normalized Scanpath Saliency: mean z-scored saliency at fixations."""
    saliency = _as_map(saliency)
    rows, cols = _fixation_indices(fixations, saliency.shape)
    if rows.size == 0:
        return float("nan")
    std = saliency.std()
    if std < _EPS:
        return 0.0
    z = (saliency - saliency.mean()) / std
    return float(z[rows, cols].mean())


def auc_judd(saliency, fixations):
    """AUC-Judd: ROC area with fixated-pixel saliency values as thresholds."""
    saliency = _as_map(saliency)
    rows, cols = _fixation_indices(fixations, saliency.shape)
    if rows.size == 0:
        return float("nan")

    fixated = saliency[rows, cols]
    n_fix = fixated.size
    n_pixels = saliency.size

    # For each threshold (fixated values, descending): TP rate over fixations,
    # FP rate over all pixels above threshold
    thresholds = np.sort(fixated)[::-1]
    above = np.sort(saliency.ravel())
    tp = np.arange(1, n_fix + 1) / n_fix
    # number of pixels with value >= threshold, via searchsorted on sorted array
    n_above = n_pixels - np.searchsorted(above, thresholds, side="left")
    fp = (n_above - np.arange(1, n_fix + 1)) / max(n_pixels - n_fix, 1)
    fp = np.clip(fp, 0.0, 1.0)

    # Prepend (0,0), append (1,1) and integrate
    tp = np.concatenate(([0.0], tp, [1.0]))
    fp = np.concatenate(([0.0], fp, [1.0]))
    return float(np.trapezoid(tp, fp))


def cc(map_a, map_b):
    """Pearson correlation coefficient between two maps."""
    a, b = _as_map(map_a).ravel(), _as_map(map_b).ravel()
    if a.size != b.size:
        raise ValueError("maps must have the same size")
    if a.std() < _EPS or b.std() < _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sim(map_a, map_b):
    """Similarity: histogram intersection of the two maps as distributions."""
    a, b = _as_map(map_a), _as_map(map_b)
    a = a / max(a.sum(), _EPS)
    b = b / max(b.sum(), _EPS)
    return float(np.minimum(a, b).sum())


def kl_div(target, prediction):
    """KL divergence KL(target || prediction), maps as distributions."""
    t, p = _as_map(target), _as_map(prediction)
    t = t / max(t.sum(), _EPS)
    p = p / max(p.sum(), _EPS)
    return float(np.sum(t * np.log(t / (p + _EPS) + _EPS)))
