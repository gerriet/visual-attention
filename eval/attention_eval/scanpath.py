"""Scanpath (fixation sequence) comparison metrics.

Beyond the loose greedy match (match_stats) and gridded Levenshtein used by the
C++ behavioural goldens, this module implements the two standard scanpath
metrics for the human-comparison study (roadmap M11, H4), hand-rolled to keep
the eval layer dependency-light (math only):

  multimatch  the Dewhurst et al. (2012) dimensions — shape, direction, length,
              position — over DTW-aligned saccade-vector sequences. The
              duration dimension is omitted: model scanpaths carry no fixation
              durations (standard for saliency-model evaluation). Each
              dimension is a similarity in [0, 1], higher = more alike.
  scanmatch   Cristino et al. (2010): grid-quantize the fixations, then a
              Needleman-Wunsch global alignment with a substitution score that
              falls off with grid distance. Normalized to [0, 1].
"""

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


# --- MultiMatch (Dewhurst et al. 2012), spatial dimensions -------------------

def _saccades(fixations):
    """Saccade vectors between consecutive fixations: [(dx, dy), ...]."""
    return [(fixations[k + 1][0] - fixations[k][0], fixations[k + 1][1] - fixations[k][1])
            for k in range(len(fixations) - 1)]


def _dtw_path(cost):
    """Monotonic alignment (i, j) pairs minimizing summed cost through an
    M x N cost matrix — dynamic time warping. Returns the aligned index pairs."""
    m, n = len(cost), len(cost[0])
    acc = [[0.0] * n for _ in range(m)]
    acc[0][0] = cost[0][0]
    for i in range(1, m):
        acc[i][0] = acc[i - 1][0] + cost[i][0]
    for j in range(1, n):
        acc[0][j] = acc[0][j - 1] + cost[0][j]
    for i in range(1, m):
        for j in range(1, n):
            acc[i][j] = cost[i][j] + min(acc[i - 1][j], acc[i][j - 1], acc[i - 1][j - 1])
    # Backtrack.
    i, j, pairs = m - 1, n - 1, []
    while i > 0 or j > 0:
        pairs.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = min(acc[i - 1][j], acc[i][j - 1], acc[i - 1][j - 1])
            if step == acc[i - 1][j - 1]:
                i, j = i - 1, j - 1
            elif step == acc[i - 1][j]:
                i -= 1
            else:
                j -= 1
    pairs.append((0, 0))
    pairs.reverse()
    return pairs


def multimatch(path_a, path_b, size):
    """MultiMatch spatial similarities between two fixation sequences, each a
    list of (x, y). Returns {shape, direction, length, position} in [0, 1]
    (1 = identical); NaN dimensions when a path is too short to define them."""
    nan = float("nan")
    diag = math.hypot(size[0], size[1]) or 1.0
    if len(path_a) < 2 or len(path_b) < 2:
        # Position can still be compared if both have >=1 fixation.
        if path_a and path_b:
            cost = [[math.hypot(ax - bx, ay - by) for (bx, by) in path_b] for (ax, ay) in path_a]
            pairs = _dtw_path(cost)
            pos = sum(1.0 - cost[i][j] / diag for i, j in pairs) / len(pairs)
            return {"shape": nan, "direction": nan, "length": nan, "position": max(0.0, pos)}
        return {"shape": nan, "direction": nan, "length": nan, "position": nan}

    sacc_a, sacc_b = _saccades(path_a), _saccades(path_b)
    # Shape/direction/length: align the saccade-vector sequences.
    cost = [[math.hypot(ax - bx, ay - by) for (bx, by) in sacc_b] for (ax, ay) in sacc_a]
    pairs = _dtw_path(cost)

    shape, direction, length = [], [], []
    for i, j in pairs:
        ax, ay = sacc_a[i]
        bx, by = sacc_b[j]
        shape.append(1.0 - math.hypot(ax - bx, ay - by) / (2.0 * diag))
        amp_a, amp_b = math.hypot(ax, ay), math.hypot(bx, by)
        length.append(1.0 - abs(amp_a - amp_b) / diag)
        if amp_a > 1e-9 and amp_b > 1e-9:
            cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (amp_a * amp_b)))
            direction.append(1.0 - math.acos(cos) / math.pi)

    # Position: align the full *fixation* sequences (so both endpoints, including
    # the terminal fixation, count) by Euclidean cost.
    pos_cost = [[math.hypot(ax - bx, ay - by) for (bx, by) in path_b] for (ax, ay) in path_a]
    position = [1.0 - pos_cost[i][j] / diag for i, j in _dtw_path(pos_cost)]

    def clamp_mean(values):
        return max(0.0, sum(values) / len(values)) if values else nan

    return {
        "shape": clamp_mean(shape),
        "direction": clamp_mean(direction),
        "length": clamp_mean(length),
        "position": clamp_mean(position),
    }


# --- ScanMatch (Cristino et al. 2010) ----------------------------------------

def scanmatch(path_a, path_b, size, grid=8, gap=0.2):
    """ScanMatch similarity in [0, 1]: grid-quantize both fixation sequences,
    then a Needleman-Wunsch global alignment whose substitution score is +1 for
    the same cell and falls *through zero to -1* at opposite corners, so distant
    substitutions are penalized and a gap can be preferable (the standard signed
    ScanMatch substitution matrix). Normalized so identical scanpaths score 1.0,
    a maximally-distant alignment 0.0.

    Two things about `gap` and length, because both are easy to get wrong:

    - Cristino et al. set the gap value to **0** in their own experiments,
      relying on the signed substitution matrix alone; 0.2 here is this
      implementation's choice. `eval/scanpath_vs_human.py --scanmatch-gap`
      exists so that a conclusion can be checked against both.
    - The score's dependence on path length comes mainly from the
      normalization below, not from the gap: dividing by the *longer* sequence
      caps a k-fixation path scored against an m-fixation one at roughly k/m,
      however well its k fixations match. Comparing paths of different lengths
      under a string-edit measure therefore confounds similarity with length
      (Jarodzka, Holmqvist & Nystrom, ETRA 2010) — see the matched-length
      control in docs/SCANPATH_VS_HUMAN.md.
    """
    seq_a = grid_string(path_a, size, grid)
    seq_b = grid_string(path_b, size, grid)
    if not seq_a or not seq_b:
        return float("nan")

    max_cell_dist = math.hypot(grid - 1, grid - 1) or 1.0

    def sub(a, b):
        ra, ca = divmod(a, grid)
        rb, cb = divmod(b, grid)
        return 1.0 - 2.0 * math.hypot(ra - rb, ca - cb) / max_cell_dist  # +1 near .. -1 far

    m, n = len(seq_a), len(seq_b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = -gap * i
    for j in range(1, n + 1):
        dp[0][j] = -gap * j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = max(dp[i - 1][j - 1] + sub(seq_a[i - 1], seq_b[j - 1]),
                           dp[i - 1][j] - gap,
                           dp[i][j - 1] - gap)
    # Normalize by the *longer* sequence (Cristino's convention), which is what
    # makes the score length-dependent: a short path cannot reach 1 against a
    # long one however well it matches. Per-element score in [-1, 1], mapped to
    # [0, 1].
    per_element = dp[m][n] / max(m, n)
    return max(0.0, min(1.0, (per_element + 1.0) / 2.0))
