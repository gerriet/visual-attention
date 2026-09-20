"""Small helpers shared by the study scorers (eval/priority_search.py,
eval/coco_search.py). Stdlib only, so the non-dataset arms need no venv."""

import random


def bootstrap_ci(values, iterations=2000, seed=0):
    """Bootstrap 95% CI of the mean of `values` (deterministic given `seed`)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def paired_bootstrap(a, b, iterations=4000, seed=0):
    """Mean of the paired differences a[i] - b[i] with its bootstrap 95% CI:
    (mean, lo, hi). For arms scored on the *same* units (scenes, images) — the
    pairing removes the between-unit variance an unpaired comparison of two CIs
    has to swallow. Resample the independent unit: pass one value per scene or
    image, never one per question within a scene."""
    if len(a) != len(b):
        raise ValueError("paired_bootstrap needs equally long sequences (%d vs %d)" % (len(a), len(b)))
    if not a:
        return (0.0, 0.0, 0.0)
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, iterations, seed)
    return (sum(diffs) / len(diffs), lo, hi)
