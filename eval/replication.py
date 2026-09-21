#!/usr/bin/env python3
"""Replication dossier (roadmap M10): the dissertation's own feature findings,
re-run on the reimplementation.

Each experiment rebuilds the *kind* of stimulus a thesis figure used (the lab
images themselves are not available), runs the C++ pipeline on it, and measures
the claim the thesis makes about that figure — so every verdict in
docs/replication/REPLICATION_DOSSIER.md rests on a number this script prints.
Replication track (docs/adr/0005): only dissertation components are exercised.

  shapes        Abb. 5.10  eccentricity of simple shapes: 0 for round, high for elongated
  ecc-variation Abb. 5.13  stretch an object: eccentricity response rises, symmetry falls
  gray-noise    Abb. 5.14  eccentricity / symmetry maxima stay put under added noise
  grow-threshold  Abb. 5.15  eccentricity vs the region-growing threshold (plausible 0.5-0.75)
  merge-threshold Abb. 5.16  eccentricity vs the merge threshold (plausible 12-36)
  colour-variation Abb. 5.20 colour-contrast response rises with the colour difference
  colour-noise  Abb. 5.21  colour-contrast maximum stays put under added noise
  colour-thresholds Abb. 5.22 colour contrast vs cc_add / cc_mult (broad stable range)
  exclusivity   Abb. 5.34  the odd orientation / colour gains against the common one

Feature responses are read from `attention --emit-features` (16-bit maps on a
fixed [0, 1] scale — absolute values, never stretched). Parameter sweeps on a
natural image use the thesis's running example, data/test_images/inputc.png.

  eval/replication.py --all                 # every experiment, plots into docs/replication/figures
  eval/replication.py --only shapes,exclusivity --json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNING_EXAMPLE = os.path.join(REPO, "data", "test_images", "inputc.png")
W, H = 400, 300
GRAY = (128, 128, 128)

# One feature at a time, thesis parameters, no exclusivity unless an experiment
# asks for it (exclusivity rescales by the number of segments, which would
# confound a variation curve).
FEATURES = ("color-munsell", "eccentricity", "symmetry", "color", "intensity", "orientation")


def config_text(enabled, params=None):
    lines = ["pipeline:", "  fusion: weighted-sum", "  selection: nms", "features:"]
    for name in FEATURES:
        lines.append("  %s:" % name)
        if name not in enabled:
            lines.append("    enabled: false")
            continue
        lines.append("    weight: 1.0")
        if params and params.get(name):
            lines.append("    params:")
            lines += ["      %s: %s" % (k, v) for k, v in params[name].items()]
    return "\n".join(lines) + "\n"


def feature_maps(binary, image, enabled, params=None):
    """{feature: float array in [0, 1]} for a PIL image or an image path."""
    with tempfile.TemporaryDirectory() as tmp:
        path = image
        if not isinstance(image, str):
            path = os.path.join(tmp, "stimulus.png")
            image.save(path)
        config = os.path.join(tmp, "config.yaml")
        with open(config, "w") as fh:
            fh.write(config_text(enabled, params))
        out = os.path.join(tmp, "features")
        subprocess.run([binary, "--config", config, path, "--no-display", "--emit-features", out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {name: np.asarray(Image.open(os.path.join(out, "feature_%s.png" % name))).astype(np.float64) / 65535.0
                for name in enabled}


def canvas(color=GRAY):
    image = Image.new("RGB", (W, H), color)
    return image, ImageDraw.Draw(image)


def ellipse(draw, cx, cy, rx, ry, fill):
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill)


def add_noise(image, level, seed):
    """Normally distributed noise, sigma = level * 128 grey levels, clipped."""
    rng = np.random.RandomState(seed)
    data = np.asarray(image).astype(np.float64)
    data += rng.normal(0.0, level * 128.0, data.shape)
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))


def region_mean(array, box):
    x0, y0, x1, y1 = box
    return float(array[y0:y1, x0:x1].mean())


def argmax_xy(array):
    y, x = np.unravel_index(int(np.argmax(array)), array.shape)
    return int(x), int(y)


def inside(point, box, margin=0):
    x0, y0, x1, y1 = box
    return x0 - margin <= point[0] <= x1 + margin and y0 - margin <= point[1] <= y1 + margin


def correlation(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def monotone_fraction(values, rising=True):
    """Share of consecutive steps that go the predicted way (ties count)."""
    steps = np.diff(values)
    good = steps >= -1e-9 if rising else steps <= 1e-9
    return float(good.mean()) if len(steps) else 1.0


# --- experiments --------------------------------------------------------------

def exp_shapes(binary):
    image, d = canvas()
    ellipse(d, 70, 70, 40, 40, (200,) * 3)       # disk
    ellipse(d, 230, 80, 80, 40, (60,) * 3)       # ellipse, axes 2:1
    d.rectangle([40, 180, 200, 200], fill=(220,) * 3)   # bar, 8:1
    d.rectangle([260, 170, 340, 250], fill=(40,) * 3)   # square
    ecc = feature_maps(binary, image, ["eccentricity"])["eccentricity"]
    values = {"disk": ecc[70, 70], "square": ecc[210, 300], "ellipse_2to1": ecc[80, 230], "bar_8to1": ecc[190, 120]}
    return {"values": {k: float(v) for k, v in values.items()},
            "expected": {"disk": 0.0, "square": 0.0, "ellipse_2to1": 0.36, "bar_8to1": (63.0 / 65.0) ** 2},
            "stimulus": image}


def exp_ecc_variation(binary):
    """A reference disk on the left; on the right an object of constant area
    stretched from round to 6:1."""
    aspects = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    area_radius = 34.0
    ecc_values, sym_values = [], []
    for aspect in aspects:
        image, d = canvas()
        ellipse(d, 90, 150, 34, 34, (210,) * 3)
        rx, ry = area_radius * np.sqrt(aspect), area_radius / np.sqrt(aspect)
        ellipse(d, 280, 150, rx, ry, (210,) * 3)
        maps = feature_maps(binary, image, ["eccentricity", "symmetry"])
        ecc_values.append(float(maps["eccentricity"][150, 280]))
        box = (int(280 - rx), int(150 - ry), int(280 + rx), int(150 + ry))
        sym_values.append(float(maps["symmetry"][box[1]:box[3], box[0]:box[2]].max()))
    predicted = [((a * a - 1) / (a * a + 1)) ** 2 for a in aspects]
    return {"aspects": aspects, "eccentricity": ecc_values, "eccentricity_predicted": predicted,
            "symmetry_max": sym_values,
            "eccentricity_monotone": monotone_fraction(ecc_values, rising=True),
            "symmetry_falls": monotone_fraction(sym_values, rising=False),
            "symmetry_first_to_last": [sym_values[0], sym_values[-1]]}


def exp_gray_noise(binary, seeds=5):
    """A bar (eccentricity target) and a disk (symmetry target); does each
    feature's maximum stay on its target as noise grows?"""
    bar, disk = (60, 60, 260, 84), (270, 170, 350, 250)
    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for level in levels:
        hits = {"eccentricity": 0, "symmetry": 0}
        for seed in range(seeds):
            # A dark bar: the feature histogram-equalises its input (as the
            # original does), and on a three-level image that maps a *bright*
            # bar and the dominant ground to nearly the same grey — they merge.
            image, d = canvas()
            d.rectangle(list(bar), fill=(20,) * 3)
            ellipse(d, 310, 210, 40, 40, (235,) * 3)
            maps = feature_maps(binary, add_noise(image, level, seed), ["eccentricity", "symmetry"])
            hits["eccentricity"] += inside(argmax_xy(maps["eccentricity"]), bar, 4)
            hits["symmetry"] += inside(argmax_xy(maps["symmetry"]), disk, 4)
        rows.append({"level": level, "eccentricity_on_bar": hits["eccentricity"] / seeds,
                     "symmetry_on_disk": hits["symmetry"] / seeds})
    # The same for symmetry with the disk alone: with two objects its maximum
    # lies *between* them already without noise (docs/VLM_VIDEO.md saw the same).
    alone = []
    for level in levels:
        hits = 0
        for seed in range(seeds):
            image, d = canvas()
            ellipse(d, 200, 150, 40, 40, (235,) * 3)
            m = feature_maps(binary, add_noise(image, level, seed), ["symmetry"])["symmetry"]
            hits += inside(argmax_xy(m), (160, 110, 240, 190), 4)
        alone.append({"level": level, "symmetry_on_disk": hits / seeds})
    return {"levels": rows, "symmetry_disk_alone": alone, "sigma_gray_levels": [l * 128 for l in levels]}


def sweep_on_running_example(binary, feature, param, values, default):
    reference = feature_maps(binary, RUNNING_EXAMPLE, [feature], {feature: {param: default}})[feature]
    rows = []
    for value in values:
        m = feature_maps(binary, RUNNING_EXAMPLE, [feature], {feature: {param: value}})[feature]
        rows.append({"value": value, "correlation_with_default": correlation(m, reference),
                     "salient_share": float((m > 0.25).mean())})
    return rows


def exp_grow_threshold(binary):
    values = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    return {"rows": sweep_on_running_example(binary, "eccentricity", "edge_threshold", values, 0.65),
            "thesis_plausible_range": [0.5, 0.75], "default": 0.65}


def exp_merge_threshold(binary):
    values = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    return {"rows": sweep_on_running_example(binary, "eccentricity", "merge_mean_difference", values, 20),
            "thesis_plausible_range": [12, 36], "default": 20}


def exp_colour_variation(binary):
    """A blob whose colour moves from the ground colour to its opponent."""
    ground, far = np.array([70, 150, 60]), np.array([215, 40, 40])  # green -> red (RGB)
    steps = np.linspace(0.0, 1.0, 9)
    values = []
    for t in steps:
        image, d = canvas(tuple(int(v) for v in ground))
        ellipse(d, 200, 150, 45, 45, tuple(int(v) for v in ground + t * (far - ground)))
        values.append(float(feature_maps(binary, image, ["color-munsell"])["color-munsell"][150, 200]))
    return {"mix": [float(t) for t in steps], "saliency": values,
            "monotone": monotone_fraction(values, rising=True)}


def exp_colour_noise(binary, seeds=5):
    blob = (155, 105, 245, 195)
    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for level in levels:
        hits, contrast = 0, []
        for seed in range(seeds):
            image, d = canvas((70, 150, 60))
            ellipse(d, 200, 150, 45, 45, (215, 40, 40))
            m = feature_maps(binary, add_noise(image, level, seed), ["color-munsell"])["color-munsell"]
            hits += inside(argmax_xy(m), blob, 4)
            contrast.append(region_mean(m, (170, 120, 230, 180)) - region_mean(m, (0, 0, 120, 80)))
        rows.append({"level": level, "maximum_on_blob": hits / seeds, "blob_minus_ground": float(np.mean(contrast))})
    return {"levels": rows}


def exp_colour_thresholds(binary):
    ball, picture = (30, 155, 75, 200), (5, 45, 75, 100)  # the two regions the thesis names (256 x 256)
    reference = feature_maps(binary, RUNNING_EXAMPLE, ["color-munsell"])["color-munsell"]
    rows = []
    for name, values, default in (("threshold", [2, 4, 8, 12, 16, 24, 32], 8),
                                  ("threshold_sigma", [0, 2, 5, 8, 12, 20], 5)):
        for value in values:
            m = feature_maps(binary, RUNNING_EXAMPLE, ["color-munsell"], {"color-munsell": {name: value}})["color-munsell"]
            rows.append({"parameter": name, "value": value, "is_default": value == default,
                         "correlation_with_default": correlation(m, reference),
                         "ball": region_mean(m, ball), "picture": region_mean(m, picture),
                         "rest": float(np.median(m))})
    return {"rows": rows, "thesis_defaults": {"cc_add": 8, "cc_mult": 5}}


def exp_exclusivity(binary):
    out = {}
    # five vertical bars and one horizontal one
    image, d = canvas()
    for i in range(5):
        d.rectangle([30 + i * 60, 30, 44 + i * 60, 140], fill=(230,) * 3)
    d.rectangle([100, 220, 260, 236], fill=(230,) * 3)
    for label, strength in (("plain", 1.0), ("weighted", 1.1)):
        m = feature_maps(binary, image, ["eccentricity"], {"eccentricity": {"exclusivity": strength}})["eccentricity"]
        out["orientation_" + label] = {"common": float(m[80, 37]), "odd": float(m[228, 180])}
    # five muted green blobs and one muted red one
    image, d = canvas((120, 120, 120))
    for x, y in ((230, 70), (330, 90), (210, 210), (320, 230), (90, 220)):
        ellipse(d, x, y, 30, 30, (100, 140, 95))
    ellipse(d, 80, 80, 30, 30, (150, 100, 100))
    for label, strength in (("plain", 1.0), ("weighted", 1.1)):
        m = feature_maps(binary, image, ["color-munsell"],
                         {"color-munsell": {"exclusivity": strength, "attribute_threshold": 2.0}})["color-munsell"]
        out["colour_" + label] = {"common": float(m[70, 230]), "odd": float(m[80, 80])}
    for kind in ("orientation", "colour"):
        out[kind + "_odd_over_common"] = {k: out["%s_%s" % (kind, k)]["odd"] / max(out["%s_%s" % (kind, k)]["common"], 1e-9)
                                          for k in ("plain", "weighted")}
    out["thesis_factor_for_5_vs_1"] = 1.1 ** 4
    return out


EXPERIMENTS = {
    "shapes": exp_shapes,
    "ecc-variation": exp_ecc_variation,
    "gray-noise": exp_gray_noise,
    "grow-threshold": exp_grow_threshold,
    "merge-threshold": exp_merge_threshold,
    "colour-variation": exp_colour_variation,
    "colour-noise": exp_colour_noise,
    "colour-thresholds": exp_colour_thresholds,
    "exclusivity": exp_exclusivity,
}


# --- plots ----------------------------------------------------------------------

def plot_all(results, directory):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots", file=sys.stderr)
        return
    os.makedirs(directory, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(directory, name), dpi=130)
        plt.close(fig)

    if "ecc-variation" in results:
        r = results["ecc-variation"]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(r["aspects"], r["eccentricity"], "o-", color="#2a78d6", label="eccentricity (measured)")
        ax.plot(r["aspects"], r["eccentricity_predicted"], "--", color="#2a78d6", alpha=0.5, label="eq. 5.6 for an ellipse")
        ax.plot(r["aspects"], r["symmetry_max"], "s-", color="#e34948", label="symmetry (max on the object)")
        ax.set_xlabel("aspect ratio of the right object")
        ax.set_ylabel("feature response")
        ax.set_title("Abb. 5.13: stretching an object")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "ecc_variation.png")
    if "colour-variation" in results:
        r = results["colour-variation"]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(r["mix"], r["saliency"], "o-", color="#1baf7a")
        ax.set_xlabel("blob colour: ground (0) to opponent colour (1)")
        ax.set_ylabel("colour-contrast response")
        ax.set_title("Abb. 5.20: increasing colour contrast")
        save(fig, "colour_variation.png")
    for key, title, xlabel in (("grow-threshold", "Abb. 5.15: region-growing threshold", "share of pixels below the threshold"),
                               ("merge-threshold", "Abb. 5.16: merge threshold", "max. mean-grey difference")):
        if key in results:
            r = results[key]
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            ax.plot([x["value"] for x in r["rows"]], [x["correlation_with_default"] for x in r["rows"]], "o-", color="#4a3aa7")
            ax.axvspan(*r["thesis_plausible_range"], color="#4a3aa7", alpha=0.08, label="thesis: plausible range")
            ax.axvline(r["default"], color="#52514e", lw=0.8, ls=":")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("correlation with the default map")
            ax.set_title(title)
            ax.legend(frameon=False, fontsize=8)
            save(fig, key.replace("-", "_") + ".png")
    if "gray-noise" in results and "colour-noise" in results:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        g, c = results["gray-noise"]["levels"], results["colour-noise"]["levels"]
        ax.plot([x["level"] for x in g], [x["eccentricity_on_bar"] for x in g], "o-", color="#2a78d6", label="eccentricity max on the bar")
        ax.plot([x["level"] for x in g], [x["symmetry_on_disk"] for x in g], "s-", color="#e34948", label="symmetry max on the disk")
        ax.plot([x["level"] for x in c], [x["maximum_on_blob"] for x in c], "^-", color="#1baf7a", label="colour max on the blob")
        ax.set_xlabel("noise level (sigma = level x 128 grey levels)")
        ax.set_ylabel("share of noise seeds")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Abb. 5.14 / 5.21: robustness to noise")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "noise.png")
    if "shapes" in results and results["shapes"].get("stimulus") is not None:
        results["shapes"]["stimulus"].save(os.path.join(directory, "shapes_stimulus.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=os.path.join(REPO, "build", "attention"))
    ap.add_argument("--all", action="store_true", help="run every experiment")
    ap.add_argument("--only", default="", help="comma-separated experiment names")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "replication"))
    ap.add_argument("--figures", default=os.path.join(REPO, "docs", "replication", "figures"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    names = list(EXPERIMENTS) if args.all else [n.strip() for n in args.only.split(",") if n.strip()]
    if not names:
        sys.exit("nothing to do: pass --all or --only NAME[,NAME] (available: %s)" % ", ".join(EXPERIMENTS))
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        sys.exit("unknown experiment(s): %s (available: %s)" % (", ".join(unknown), ", ".join(EXPERIMENTS)))
    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)

    results = {}
    for name in names:
        print("running %s ..." % name, file=sys.stderr)
        results[name] = EXPERIMENTS[name](args.binary)
    if not args.no_plots:
        plot_all(results, args.figures)

    serializable = {name: {k: v for k, v in r.items() if k != "stimulus"} for name, r in results.items()}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "replication.json"), "w") as fh:
        json.dump(serializable, fh, indent=2)
    if args.json:
        print(json.dumps(serializable, indent=2))
    print("results: %s" % os.path.join(args.out, "replication.json"))


if __name__ == "__main__":
    main()
