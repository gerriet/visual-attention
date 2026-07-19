#!/usr/bin/env python3
"""Render the gated-recognition (H2) accuracy-vs-compute figure.

Reads the summary.json files that eval/gated_recognition.py wrote (the vtest
self-relative arm and/or the DAVIS ground-truth arm) and draws recall against
the fraction of recognition compute (pixels through the detector), one panel
per ground-truth regime. Needs matplotlib (the eval venv).

  eval/plot_gated_recognition.py results/gated_recognition/summary.json \\
      results/gated_recognition_davis/summary.json \\
      --out docs/images/gated_recognition_tradeoff.png
"""

import argparse
import json

# Categorical slots from the validated reference palette (docs figure, light
# surface). Contrast WARN on aqua/yellow is relieved by direct labels and
# distinct marker shapes.
ARM_STYLE = {
    "full-frame": {"color": "#2a78d6", "marker": "o", "label": "full-frame"},
    "frame": {"color": "#1baf7a", "marker": "s", "label": "gated (every frame)"},
    "dwell": {"color": "#eda100", "marker": "^", "label": "gated (per dwell)"},
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"


def load_summaries(paths):
    video, davis = None, None
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        video = data.get("video") or video
        davis = data.get("davis") or davis
    return video, davis


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 1.6)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.set_xticks([0.03, 0.1, 0.3, 1.0])
    ax.set_xticklabels(["3%", "10%", "30%", "100%"])


def plot_video(ax, video):
    points = [("full-frame", 1.0, 1.0)]
    for cadence, m in video["arms"].items():
        points.append((cadence, m["pixels_fraction"], m["windowed"]))
    points.sort(key=lambda p: p[1])
    ax.plot([p[1] for p in points], [p[2] for p in points],
            color=GRID, linewidth=1.2, zorder=1)
    for arm, x, y in points:
        s = ARM_STYLE[arm]
        ax.scatter([x], [y], s=55, color=s["color"], marker=s["marker"], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(s["label"], (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=TEXT_PRIMARY)
    style_axis(ax, "vtest — recovery of full-frame detections (±15 frames)",
               "recognition compute (fraction of pixels, log)", "windowed recovery")


def plot_davis(ax, davis):
    ends = []
    for sequence, rows in davis.items():
        trace = sorted(((arm, m["pixels_fraction"], m["windowed"])
                        for arm, m in rows.items()), key=lambda p: p[1])
        ax.plot([p[1] for p in trace], [p[2] for p in trace],
                color=GRID, linewidth=1.0, zorder=1)
        for arm, x, y in trace:
            s = ARM_STYLE[arm]
            ax.scatter([x], [y], s=40, color=s["color"], marker=s["marker"], zorder=3,
                       edgecolors="white", linewidths=1.2)
        ends.append((sequence, trace[-1][2]))

    # Sequence names at the full-frame end (x = 1.0), vertically dodged so
    # close endpoints don't collide.
    ends.sort(key=lambda e: e[1], reverse=True)
    slot_y = None
    for sequence, y in ends:
        slot_y = y if slot_y is None else min(y, slot_y - 0.062)
        ax.annotate(sequence, (1.0, y), textcoords="offset points",
                    xytext=(9, (slot_y - y) * 205), fontsize=7.5,
                    color=TEXT_SECONDARY, va="center")
    from matplotlib.lines import Line2D  # matplotlib is imported by main()

    handles = [Line2D([], [], color=s["color"], marker=s["marker"], linestyle="none",
                      markersize=7, markeredgecolor="white", label=s["label"])
               for s in ARM_STYLE.values()]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
              labelcolor=TEXT_PRIMARY)
    style_axis(ax, "DAVIS person sequences — recall vs mask ground truth (±15)",
               "recognition compute (fraction of pixels, log)", "windowed person-frame recall")
    ax.set_xlim(0.02, 2.8)  # room for the sequence labels right of x = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+", help="summary.json files from eval/gated_recognition.py")
    ap.add_argument("--out", default="docs/images/gated_recognition_tradeoff.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video, davis = load_summaries(args.summaries)
    panels = int(video is not None) + int(davis is not None)
    if panels == 0:
        raise SystemExit("no 'video' or 'davis' section found in the given summaries")

    fig, axes = plt.subplots(1, panels, figsize=(5.4 * panels, 4.2), facecolor="#fcfcfb")
    axes = [axes] if panels == 1 else list(axes)
    if video is not None:
        plot_video(axes.pop(0), video)
    if davis is not None:
        plot_davis(axes.pop(0), davis)
    fig.suptitle("Gated recognition (H2): accuracy vs recognition compute",
                 fontsize=12, color=TEXT_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
