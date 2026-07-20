#!/usr/bin/env python3
"""Render the VLM front-end (H6) accuracy-vs-token-budget figure from the
summary.json that eval/vlm_frontend.py writes. Needs matplotlib (the eval venv).

  eval/plot_vlm_frontend.py results/vlm_frontend/summary.json \\
      --out docs/images/vlm_frontend_tradeoff.png
"""

import argparse
import json

# Validated categorical slots (reference palette, light surface).
STYLE = {
    "full-res": {"color": "#2a78d6", "marker": "o", "label": "full-res (ceiling)"},
    "uniform": {"color": "#e34948", "marker": "s", "label": "uniform downsample"},
    "fovea": {"color": "#1baf7a", "marker": "^", "label": "attention fovea (ours)"},
}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="summary.json from eval/vlm_frontend.py")
    ap.add_argument("--out", default="docs/images/vlm_frontend_tradeoff.png")
    ap.add_argument("--title", default=None, help="override the panel title")
    args = ap.parse_args()

    with open(args.summary) as fh:
        data = json.load(fh)
    summary = data["summary"]
    backend = data.get("backend", "?")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor="#fcfcfb")
    for arm in ("full-res", "uniform", "fovea"):
        s = summary[arm]
        x = max(s["mean_token_fraction"], 0.01)
        y = s["accuracy"]
        lo, hi = s["accuracy_ci"]
        st = STYLE[arm]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=GRID, elinewidth=1.2, zorder=1)
        ax.scatter([x], [y], s=90, color=st["color"], marker=st["marker"], zorder=3,
                   edgecolors="white", linewidths=1.5, label=st["label"])
        ax.annotate("%.0f%%" % (y * 100), (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=TEXT_PRIMARY)

    ax.set_xscale("log")
    ax.set_xlim(0.02, 1.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0.03, 0.1, 0.3, 1.0])
    ax.set_xticklabels(["3%", "10%", "30%", "100%"])
    ax.set_xlabel("visual-token budget (fraction of full-res, log)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("accuracy", fontsize=9, color=TEXT_SECONDARY)
    title = args.title or ("Attention as a VLM token budget (H6) — backend: %s" % backend)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
