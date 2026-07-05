"""Scanpath (fixation sequence) comparison metrics."""

import math


def match_stats(reference, other, pos_tol=20.0):
    """Greedy nearest-match statistics between two fixation sequences.

    Returns a dict with:
      matched_fraction  fraction of reference fixations with a counterpart
                        within pos_tol
      mean_distance     mean distance of matched pairs (px)
      mean_rank_shift   mean |rank difference| of matched pairs
    """
    if not reference:
        return {"matched_fraction": float("nan"), "mean_distance": float("nan"),
                "mean_rank_shift": float("nan")}

    distances, rank_shifts, matched = [], [], 0
    for i, (rx, ry) in enumerate(reference):
        best = None
        for j, (ox, oy) in enumerate(other):
            d = math.hypot(rx - ox, ry - oy)
            if d <= pos_tol and (best is None or d < best[1]):
                best = (j, d)
        if best is not None:
            matched += 1
            distances.append(best[1])
            rank_shifts.append(abs(best[0] - i))

    return {
        "matched_fraction": matched / len(reference),
        "mean_distance": sum(distances) / len(distances) if distances else float("nan"),
        "mean_rank_shift": sum(rank_shifts) / len(rank_shifts) if rank_shifts else float("nan"),
    }


def grid_string(fixations, size, grid=5):
    """Quantize fixations into a grid-cell sequence (ScanMatch-style)."""
    width, height = size
    cells = []
    for x, y in fixations:
        col = min(int(x * grid / max(width, 1)), grid - 1)
        row = min(int(y * grid / max(height, 1)), grid - 1)
        cells.append(row * grid + col)
    return cells


def levenshtein(seq_a, seq_b):
    """Edit distance between two sequences."""
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    previous = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        current = [i]
        for j, b in enumerate(seq_b, start=1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (a != b)))  # substitution
        previous = current
    return previous[-1]


def gridded_levenshtein(reference, other, size, grid=5):
    """Normalized edit distance between grid-quantized scanpaths (0 = equal,
    1 = maximally different)."""
    seq_a = grid_string(reference, size, grid)
    seq_b = grid_string(other, size, grid)
    longest = max(len(seq_a), len(seq_b))
    if longest == 0:
        return 0.0
    return levenshtein(seq_a, seq_b) / longest
