"""Plots for the benchmark: a saliency montage and a metric bar chart.

matplotlib is an optional dependency (imported lazily, headless Agg backend),
so the core metrics/report/benchmark run without it.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def saliency_montage(image_path, model_results, out_png, reference=None):
    """One row: the original image, then each model's saliency heatmap.

    model_results: {display_name: Result}. reference (if given) is drawn first."""
    plt = _plt()
    names = list(model_results)
    if reference and reference in names:
        names = [reference] + [n for n in names if n != reference]

    image = np.asarray(Image.open(image_path).convert("RGB"))
    n = len(names) + 1
    # squeeze=False so `axes` is always a 2D array, even for a single panel
    # (zero models) — indexing then never hits a bare Axes.
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), squeeze=False)
    axes = axes[0]
    axes[0].imshow(image)
    axes[0].set_title("input", fontsize=10)
    axes[0].axis("off")
    for ax, name in zip(axes[1:], names):
        sal = model_results[name].saliency
        if sal is not None:
            ax.imshow(sal, cmap="inferno")
        ax.set_title(name + (" (ref)" if name == reference else ""), fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def metric_bars(rows, metric, out_png, reference=None, title=None):
    """Bar chart of one metric across models. rows: {model: {metric: value}}."""
    plt = _plt()
    names = [n for n in rows if not (n == reference)]  # reference is trivially 1.0/0.0
    values = [rows[n].get(metric, float("nan")) for n in names]

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(names) + 2), 3.5))
    ax.bar(names, values, color="#4a7fb5")
    ax.set_ylabel(metric)
    ax.set_title(title or (f"{metric} vs {reference}" if reference else metric))
    ax.axhline(0, color="#888", linewidth=0.8)
    for i, v in enumerate(values):
        if v == v:
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
