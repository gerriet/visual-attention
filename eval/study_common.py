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
