#!/usr/bin/env python3
"""Render the priority-map study (H5) figure: the synthetic ablation and the
COCO-Search18 external-validity arm, side by side. Reads the summary.json files
written by eval/priority_search.py and eval/coco_search.py. Needs matplotlib.

  eval/plot_priority_search.py results/priority_search/summary.json \\
      results/coco_search/summary.json \\
      --out docs/images/priority_map_h5.png
"""

import argparse
import json

# Validated categorical slots (reference palette, light surface). Contrast WARN
# on aqua/yellow relieved by direct value labels on every bar.
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

SYNTH_ARMS = [("bottom-up", BLUE), ("feature-td", AQUA), ("target-td", YELLOW),
              ("full-priority", GREEN)]
COCO_ARMS = [("human", "#8a8a8a"), ("bottom-up", BLUE), ("prior", YELLOW)]


def load(paths):
    synth, coco = None, None
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        if "summary" in data and isinstance(data["summary"], dict):
            synth = data
        if "rows" in data:
            coco = data
    return synth, coco


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def bar_with_ci(ax, i, value, ci, color, label):
    ax.bar(i, value, width=0.66, color=color, zorder=2)
    top = value
    if ci:
        ax.plot([i, i], ci, color=TEXT_PRIMARY, linewidth=1.2, zorder=3)
        top = max(value, ci[1])
    ax.annotate("%.1f" % value, (i, top), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=8, color=TEXT_PRIMARY)


def plot_synth(ax, synth):
    s = synth["summary"]
    for i, (arm, color) in enumerate(SYNTH_ARMS):
        bar_with_ci(ax, i, s[arm]["time_to_target"], s[arm].get("time_ci"), color, arm)
    ax.set_xticks(range(len(SYNTH_ARMS)))
    ax.set_xticklabels([a for a, _ in SYNTH_ARMS], fontsize=8, rotation=12)
    style(ax, "Synthetic search — time-to-target (frames, lower better)", "frames")


def plot_coco(ax, coco):
    rows = {r["name"]: r for r in coco["rows"]}
    for i, (arm, color) in enumerate(COCO_ARMS):
        r = rows[arm]
        bar_with_ci(ax, i, r["mean"], r.get("ci"), color, arm)
    ax.set_xticks(range(len(COCO_ARMS)))
    ax.set_xticklabels([a for a, _ in COCO_ARMS], fontsize=8)
    style(ax, "COCO-Search18 — fixations-to-target (lower better)", "fixations")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", default="docs/images/priority_map_h5.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synth, coco = load(args.summaries)
    panels = int(synth is not None) + int(coco is not None)
    if panels == 0:
        raise SystemExit("no priority_search or coco_search summary found")

    fig, axes = plt.subplots(1, panels, figsize=(5.6 * panels, 4.0), facecolor="#fcfcfb")
    axes = [axes] if panels == 1 else list(axes)
    if synth is not None:
        plot_synth(axes.pop(0), synth)
    if coco is not None:
        plot_coco(axes.pop(0), coco)
    fig.suptitle("Priority map (H5): each task-relevance term, ablated",
                 fontsize=12, color=TEXT_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
