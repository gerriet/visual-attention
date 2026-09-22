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
  stereo-distance Abb. 5.28 the depth response follows an object's disparity
  stereo-rds    Abb. 5.29  a random-dot stereogram: depth where no monocular cue exists
  stereo-noise  Abb. 5.30  disparity estimates under independent noise in both images
  stereo-orientations Abb. 5.31 one vs several near-vertical Gabor orientations
  stereo-variance Abb. 5.32 the variance threshold: high drops correct pixels, low admits wrong ones
  stereo-real   (calibration) the variance threshold on real pairs with ground truth (Middlebury
                           2001: Tsukuba, Venus, Sawtooth; data/Middlebury, see eval/datasets/middlebury.py)
  text-vs-code  (no figure) the feature experiments under the thesis text's parameters and under
                           the ones the dissertation system set (esab2.C): do both reproduce them?
  field         Abb. 6.4-6.10 the neural field driven with synthetic activation (build/field_dynamics):
                           hysteresis, bifurcation, noise, convergence, tracking, two approaching
                           maxima — under the dissertation system's field parameters and the port's
  symmetry-sides  finding A, quantified: the response of an edge on one side alone against a
                           closed contour's, with and without the thesis's clip offset
  world-model   thesis 9.2, Abb. 9.2 (= WAPCV 2003, Fig. 5): the two-stage model against a conventional
                           inhibition-map model on simulated master maps — recognized objects and
                           position error, 50 runs per condition (build/world_model)

Stereo stimuli are rendered pairs with known disparity (textured surfaces, a
random-dot stereogram); the depth response is |disparity| / search range (eq.
5.16), so it reads back as a disparity estimate.

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


# Parameter values the dissertation system set where the thesis text says
# otherwise (esab2.C; configs/thesis/thesis_esab2.yaml). Applied underneath an
# experiment's own parameters while the text-vs-code experiment runs.
CODE_PARAMETERS = {
    "eccentricity": {"edge_threshold": 0.78, "variance_threshold": 1.5, "saliency_offset": 0.2},
    "color-munsell": {"threshold": 12, "threshold_sigma": 6, "max_segment": 0.12, "sigmoid_beta": 4.0},
}
PARAMETER_BASE = {}


def config_text(enabled, params=None):
    merged = {name: dict(PARAMETER_BASE.get(name, {})) for name in enabled}
    for name, values in (params or {}).items():
        merged.setdefault(name, {}).update(values)
    params = merged
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


# --- stereo ---------------------------------------------------------------------

STEREO_SIZE = 256
STEREO_RANGE = 16                      # disparity search range (px), as in configs/thesis/stereo.yaml
FOREGROUND = (80, 64, 176, 176)        # x0, y0, x1, y1 of the near surface in the left image
FLAT = (16, 200, 240, 244)             # a textureless band (stereo-variance)
FAINT = (20, 20, 70, 130)              # a low-contrast surface at disparity 6 (stereo-variance)
FAINT_DISPARITY = 6


def dots(rng, h, w, cell=2):
    """Random dots: no monocular structure, texture at every orientation."""
    coarse = rng.randint(0, 2, size=(h // cell + 1, w // cell + 1)) * 255
    return np.kron(coarse, np.ones((cell, cell)))[:h, :w].astype(np.float64)


def stereo_pair(disparity, seed=0, flat_band=False, faint_surface=False):
    """(left, right) grey arrays: a textured ground at disparity 0 and a textured
    rectangle shifted left by `disparity` px in the right image (a nearer
    surface). Optionally a textureless band in the ground, and a second surface
    whose texture has a tenth of the contrast."""
    rng = np.random.RandomState(1000 + seed)
    ground, surface = dots(rng, STEREO_SIZE, STEREO_SIZE), dots(rng, STEREO_SIZE, STEREO_SIZE)
    if flat_band:
        x0, y0, x1, y1 = FLAT
        ground[y0:y1, x0:x1] = 128.0
    x0, y0, x1, y1 = FOREGROUND
    left, right = ground.copy(), ground.copy()
    left[y0:y1, x0:x1] = surface[y0:y1, x0:x1]
    right[y0:y1, x0 - disparity:x1 - disparity] = surface[y0:y1, x0:x1]
    if faint_surface:
        x0, y0, x1, y1 = FAINT
        faint = 128.0 + (dots(rng, STEREO_SIZE, STEREO_SIZE) - 127.5) * 0.1
        left[y0:y1, x0:x1] = faint[y0:y1, x0:x1]
        right[y0:y1, x0 - FAINT_DISPARITY:x1 - FAINT_DISPARITY] = faint[y0:y1, x0:x1]
    return left, right


def noisy(array, sigma, rng):
    return np.clip(array + rng.normal(0.0, sigma, array.shape), 0, 255) if sigma > 0 else array


def stereo_map(binary, left, right, params=None):
    """The depth feature's response in [0, 1] for a (left, right) pair."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, array in (("left", left), ("right", right)):
            paths.append(os.path.join(tmp, name + ".png"))
            Image.fromarray(np.asarray(array).astype(np.uint8), mode="L").save(paths[-1])
        merged = {"min_disparity": -STEREO_RANGE, "max_disparity": 0}
        merged.update(params or {})
        config = os.path.join(tmp, "config.yaml")
        with open(config, "w") as fh:
            fh.write(config_text([]) + "  stereo:\n    weight: 1.0\n    params:\n"
                     + "".join("      %s: %s\n" % (k, v) for k, v in merged.items()))
        out = os.path.join(tmp, "features")
        # abspath: the CLI runs in the temp directory (it writes a visualization into its cwd)
        subprocess.run([os.path.abspath(binary), "--stereo", paths[0], paths[1], "--config", config,
                        "--emit-features", out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp)
        return np.asarray(Image.open(os.path.join(out, "feature_stereo.png"))).astype(np.float64) / 65535.0


def inner(box, margin=12):
    x0, y0, x1, y1 = box
    return (x0 + margin, y0 + margin, x1 - margin, y1 - margin)


def disparity_stats(depth, box, truth):
    """Mean estimated disparity in `box` and the share of its pixels within one
    pixel of the true disparity."""
    x0, y0, x1, y1 = box
    estimate = depth[y0:y1, x0:x1] * STEREO_RANGE
    return float(estimate.mean()), float((np.abs(estimate - truth) <= 1.0).mean())


def exp_stereo_distance(binary):
    rows = []
    for disparity in (0, 2, 4, 6, 8, 10, 12, 14):
        depth = stereo_map(binary, *stereo_pair(disparity))
        mean, correct = disparity_stats(depth, inner(FOREGROUND), disparity)
        rows.append({"disparity": disparity, "response": mean / STEREO_RANGE, "estimated_disparity": mean,
                     "within_1px": correct, "ground_response": region_mean(depth, (8, 8, 60, 248))})
    return {"rows": rows, "monotone": monotone_fraction([r["response"] for r in rows], rising=True)}


def exp_stereo_rds(binary):
    """The pair is pure random dots; the square exists only as a disparity."""
    left, right = stereo_pair(8, seed=7)
    depth = stereo_map(binary, left, right)
    mean, correct = disparity_stats(depth, inner(FOREGROUND), 8)
    monocular = abs(float(left[FOREGROUND[1]:FOREGROUND[3], FOREGROUND[0]:FOREGROUND[2]].mean()) - float(left.mean()))
    return {"disparity": 8, "square_response": mean / STEREO_RANGE, "square_within_1px": correct,
            "ground_response": region_mean(depth, (8, 8, 60, 248)),
            "monocular_mean_difference_grey_levels": monocular}


def exp_stereo_noise(binary, seeds=5):
    rows = []
    for level in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        means, correct, ground = [], [], []
        for seed in range(seeds):
            rng = np.random.RandomState(seed)
            left, right = stereo_pair(10, seed=seed)
            depth = stereo_map(binary, noisy(left, level * 128, rng), noisy(right, level * 128, rng))  # independent
            mean, share = disparity_stats(depth, inner(FOREGROUND), 10)
            means.append(mean)
            correct.append(share)
            ground.append(region_mean(depth, (8, 8, 60, 248)))
        rows.append({"level": level, "estimated_disparity": float(np.mean(means)),
                     "within_1px": float(np.mean(correct)), "ground_response": float(np.mean(ground))})
    return {"rows": rows, "true_disparity": 10}


def exp_stereo_orientations(binary, seeds=5):
    """Random dots are easy for every orientation set, so the comparison is made
    where it can show: under strong independent noise in the two images."""
    rows = []
    for sigma in (25.0, 90.0, 115.0):
        for count in (1, 3, 5):
            correct, ground = [], []
            for seed in range(seeds):
                rng = np.random.RandomState(seed)
                left, right = stereo_pair(10, seed=seed)
                depth = stereo_map(binary, noisy(left, sigma, rng), noisy(right, sigma, rng),
                                   {"num_orientations": count})
                correct.append(disparity_stats(depth, inner(FOREGROUND), 10)[1])
                ground.append(float((depth[8:248, 8:60] * STEREO_RANGE > 1.0).mean()))
            rows.append({"noise_sigma": sigma, "orientations": count, "within_1px": float(np.mean(correct)),
                         "ground_wrong_share": float(np.mean(ground))})
    return {"rows": rows, "note": "1 = vertical; 3 = +-30 deg added; 5 = +-15 and +-30 deg added"}


def exp_stereo_variance(binary, seeds=3):
    """Textured ground and surface, a second surface with a tenth of the texture
    contrast, and a textureless band — all with a little sensor noise (sigma 3).
    A low threshold lets the band through (wrong disparities); a high one drops
    correct pixels, the faint surface first."""
    rows = []
    for threshold in (0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 160.0, 240.0):
        kept, faint_kept, flat_wrong = [], [], []
        for seed in range(seeds):
            rng = np.random.RandomState(seed)
            left, right = stereo_pair(10, seed=seed, flat_band=True, faint_surface=True)
            depth = stereo_map(binary, noisy(left, 3.0, rng), noisy(right, 3.0, rng), {"variance_threshold": threshold})
            kept.append(disparity_stats(depth, inner(FOREGROUND), 10)[1])
            faint_kept.append(disparity_stats(depth, inner(FAINT), FAINT_DISPARITY)[1])
            x0, y0, x1, y1 = inner(FLAT, 8)
            flat_wrong.append(float((depth[y0:y1, x0:x1] * STEREO_RANGE > 1.0).mean()))
        rows.append({"threshold": threshold, "surface_correct": float(np.mean(kept)),
                     "faint_surface_correct": float(np.mean(faint_kept)),
                     "flat_band_wrong": float(np.mean(flat_wrong))})
    return {"rows": rows, "default": 3.0}


# Middlebury 2001 scenes whose disparities fit the 16-px search range at the
# feature's 256-px working size: (left, right, ground truth for the left view, its scale)
MIDDLEBURY = os.path.join(REPO, "data", "Middlebury")
REAL_PAIRS = {
    "tsukuba": ("scene1.row3.col3.ppm", "scene1.row3.col4.ppm", "truedisp.row3.col3.pgm", 16.0),
    "venus": ("im2.ppm", "im6.ppm", "disp2.pgm", 8.0),
    "sawtooth": ("im2.ppm", "im6.ppm", "disp2.pgm", 8.0),
}


def exp_stereo_real(binary):
    """The variance threshold gates out windows without enough structure. Its
    scale depends on the gain of the Gabor filters, which the original sources
    do not settle, so it is calibrated here against ground truth: per threshold,
    the share of pixels with a correct disparity (within 1 px at the working
    size), a wrong one, or none — overall and on textureless pixels."""
    missing = [n for n, f in REAL_PAIRS.items() if not os.path.exists(os.path.join(MIDDLEBURY, n, f[0]))]
    if missing:
        sys.exit("Middlebury scenes not found (%s) under %s — see eval/datasets/middlebury.py"
                 % (", ".join(missing), MIDDLEBURY))
    thresholds = [0.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0]
    scenes = {}
    for name, (left_file, right_file, truth_file, truth_scale) in REAL_PAIRS.items():
        folder = os.path.join(MIDDLEBURY, name)
        left = np.asarray(Image.open(os.path.join(folder, left_file)).convert("L")).astype(np.float64)
        right = np.asarray(Image.open(os.path.join(folder, right_file)).convert("L")).astype(np.float64)
        truth = np.asarray(Image.open(os.path.join(folder, truth_file))).astype(np.float64) / truth_scale
        work = 256.0 / max(left.shape)          # the feature works on a 256-px image
        valid = truth > 0
        # Textureless: little grey-value variation in a 9 x 9 neighbourhood
        pil = Image.fromarray(left.astype(np.uint8))
        from PIL import ImageFilter
        mean = np.asarray(pil.filter(ImageFilter.BoxBlur(4))).astype(np.float64)
        mean_sq = np.asarray(Image.fromarray(np.clip(left * left / 255.0, 0, 255).astype(np.uint8))
                             .filter(ImageFilter.BoxBlur(4))).astype(np.float64) * 255.0
        flat = valid & (np.sqrt(np.clip(mean_sq - mean * mean, 0, None)) < 4.0)
        rows = []
        for threshold in thresholds:
            depth = stereo_map(binary, left, right, {"variance_threshold": threshold})
            estimate = depth * STEREO_RANGE             # working-size pixels
            accepted = depth > 0
            right_value = np.abs(estimate - truth * work) <= 1.0
            rows.append({"threshold": threshold,
                         "correct": float((valid & accepted & right_value).sum() / valid.sum()),
                         "wrong": float((valid & accepted & ~right_value).sum() / valid.sum()),
                         "rejected": float((valid & ~accepted).sum() / valid.sum()),
                         "flat_wrong": float((flat & accepted & ~right_value).sum() / max(flat.sum(), 1)),
                         "flat_correct": float((flat & accepted & right_value).sum() / max(flat.sum(), 1))})
        scenes[name] = {"rows": rows, "flat_share": float(flat.sum() / valid.sum()),
                        "max_disparity_working_px": float(truth.max() * work)}
    return {"scenes": scenes, "thresholds": thresholds}


def exp_text_vs_code(binary):
    """The feature experiments twice: thesis-text parameters (the defaults) and
    the dissertation system's. Plus how much the two maps differ on the thesis's
    running example."""
    out = {}
    for label, base in (("text", {}), ("code", CODE_PARAMETERS)):
        PARAMETER_BASE.clear()
        PARAMETER_BASE.update(base)
        try:
            shapes = exp_shapes(binary)["values"]
            stretch = exp_ecc_variation(binary)
            gray = exp_gray_noise(binary, seeds=3)
            colour = exp_colour_variation(binary)
            colour_noise = exp_colour_noise(binary, seeds=3)
            maps = feature_maps(binary, RUNNING_EXAMPLE, ["eccentricity", "color-munsell"])
        finally:
            PARAMETER_BASE.clear()
        out[label] = {
            "shapes": shapes,
            "eccentricity_monotone": stretch["eccentricity_monotone"],
            "eccentricity_at_aspect_2_and_6": [stretch["eccentricity"][3], stretch["eccentricity"][-1]],
            "eccentricity_max_on_bar_by_noise": {str(r["level"]): r["eccentricity_on_bar"] for r in gray["levels"]},
            "colour_variation": colour["saliency"], "colour_monotone": colour["monotone"],
            "colour_max_on_blob_by_noise": {str(r["level"]): r["maximum_on_blob"] for r in colour_noise["levels"]},
            "running_example": {"ball": region_mean(maps["color-munsell"], (30, 155, 75, 200)),
                                "picture": region_mean(maps["color-munsell"], (5, 45, 75, 100))},
            "_maps": maps,
        }
    out["running_example_map_correlation"] = {
        name: correlation(out["text"]["_maps"][name], out["code"]["_maps"][name])
        for name in ("eccentricity", "color-munsell")}
    for label in ("text", "code"):
        del out[label]["_maps"]
    return out


# --- neural field ---------------------------------------------------------------

def exp_field(binary):
    """The field experiments run in C++ (examples/field_dynamics.cpp drives the
    field directly); here they are run once per parameter set and collected."""
    harness = os.path.join(os.path.dirname(os.path.abspath(binary)), "field_dynamics")
    if not os.path.exists(harness):
        sys.exit("field harness not found: %s (build first: cmake --build build)" % harness)
    out = {}
    for parameter_set in ("esab2", "port"):
        result = subprocess.run([harness, "--params", parameter_set], check=True, capture_output=True, text=True)
        out[parameter_set] = json.loads(result.stdout)
    return out


# --- the world-model experiment (thesis 9.2, WAPCV 2003 Fig. 5) -----------------

# Confirmatory block: seeds 1000-1049, 50 runs per condition as in the thesis.
# Development used seeds 0-9 (docs/replication/REPLICATION_DOSSIER.md, finding 22).
WORLD_MODEL = {"seed0": 1000, "runs": 50}

# Seeds per noise level (thesis gives no count; 5 was the first version's, 20
# since 2026-09-22 — see the dossier, finding 3).
NOISE = {"seeds": 20}

# Read off the published figure (WAPCV 2003, Fig. 5), to about +-0.1: mean
# recognized objects for 0..5 dynamic objects, and the conventional model's
# position error in px (the two-stage model's is below 0.5 throughout).
THESIS_FIG5 = {
    1: {"two_stage": [0.9, 1.7, 2.45, 3.1, 3.65, 4.1], "conventional": [0.9, 1.35, 1.65, 2.35, 2.85, 3.0],
        "conventional_error": [0.6, 2.1, 3.5, 3.7, 4.2, 4.7]},
    3: {"two_stage": [1.9, 3.1, 3.65, 4.1, 4.4, 4.75], "conventional": [2.3, 2.4, 2.95, 3.3, 3.55, 3.85],
        "conventional_error": [0.6, 1.4, 2.5, 2.8, 3.1, 3.4]},
    5: {"two_stage": [3.4, 4.1, 4.45, 4.75, 4.9, 5.0], "conventional": [3.3, 3.55, 3.95, 3.85, 4.3, 4.05],
        "conventional_error": [0.5, 1.3, 1.7, 2.2, 2.4, 2.8]},
}


def exp_world_model(binary):
    """examples/world_model.cpp runs both attention models on the same simulated
    master maps; here: means, and paired bootstrap intervals over runs."""
    from study_common import paired_bootstrap
    harness = os.path.join(os.path.dirname(os.path.abspath(binary)), "world_model")
    if not os.path.exists(harness):
        sys.exit("world-model harness not found: %s (build first: cmake --build build)" % harness)
    base = [harness, "--runs", str(WORLD_MODEL["runs"]), "--seed0", str(WORLD_MODEL["seed0"])]
    out = summarize_world_model(json.loads(subprocess.run(base, check=True, capture_output=True, text=True).stdout),
                                paired_bootstrap)
    # Two choices the sources leave open, declared before the confirmatory run
    # (dossier, finding 22): when the conventional model's recognition looks at
    # the focus, and whether the noise has zero mean.
    out["variants"] = {}
    for name, flags in (("late-identity", ["--late-identity"]), ("positive-noise", ["--positive-noise"])):
        raw = json.loads(subprocess.run(base + flags, check=True, capture_output=True, text=True).stdout)
        out["variants"][name] = summarize_world_model(raw, paired_bootstrap)
    out["thesis_fig5"] = THESIS_FIG5
    return out


def summarize_world_model(raw, paired_bootstrap):
    settings = {k: v for k, v in raw.items() if k != "conditions"}
    rows = []
    for condition in raw["conditions"]:
        runs = condition["runs"]
        two = [r["two_stage"][0] for r in runs]
        conventional = [r["conventional"][0] for r in runs]
        two_error = [r["two_stage"][1] for r in runs]
        conventional_error = [r["conventional"][1] for r in runs]
        rows.append({
            "static": condition["static"], "dynamic": condition["dynamic"],
            "two_stage": float(np.mean(two)), "conventional": float(np.mean(conventional)),
            "difference": paired_bootstrap(two, conventional),  # (mean, lo, hi), two-stage minus conventional
            "two_stage_error": float(np.mean(two_error)), "conventional_error": float(np.mean(conventional_error)),
            "error_difference": paired_bootstrap(two_error, conventional_error),
            "clusters": float(np.mean([r["clusters"] for r in runs])),
        })
    # Pooled over the static counts, per number of dynamic objects: run i of each
    # block is an independent scene, so the pooled unit is still the run
    pooled = []
    for dynamic in sorted({c["dynamic"] for c in raw["conditions"]}):
        blocks = [c["runs"] for c in raw["conditions"] if c["dynamic"] == dynamic]
        two = [r["two_stage"][0] for block in blocks for r in block]
        conventional = [r["conventional"][0] for block in blocks for r in block]
        pooled.append({"dynamic": dynamic, "difference": paired_bootstrap(two, conventional)})
    return {"settings": settings, "rows": rows, "pooled": pooled}


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


def exp_gray_noise(binary, seeds=None):
    seeds = seeds or NOISE["seeds"]
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


def exp_symmetry_sides(binary):
    """Finding A's quantitative core: how much of a closed contour's symmetry
    response does an edge on *one* side alone produce? The summation of eq.
    5.2/5.3 is additive over the two opposite boxes, so a lone edge scores; the
    original removes those responses with a fixed offset on an absolute scale.
    The raw ratio is measured with the additive per-band bonus off and at gains
    low enough that neither response reaches the clip at 1 — it is then
    invariant to the gain and to the disk's radius, which the rows check; the
    last rows add the bonus back and then the thesis's offset at gain 1."""
    def responses(radius, offset, gain, bonus):
        params = {"symmetry": {"clip_offset": offset, "gabor_gain": gain, "band_bonus": bonus}}
        disk_img, d = canvas()
        ellipse(d, 200, 150, radius, radius, (235,) * 3)
        disk = feature_maps(binary, disk_img, ["symmetry"], params)["symmetry"]
        edge_img, d = canvas()
        d.rectangle([200, 0, 400, 300], fill=(235,) * 3)  # one straight edge at x = 200
        edge = feature_maps(binary, edge_img, ["symmetry"], params)["symmetry"]
        # The disk's response at its centre; the edge's maximum away from the
        # image border (a lone edge has no centre of its own)
        inner = (slice(40, 260), slice(40, 360))
        centre = float(disk[150, 200])
        return {"radius": radius, "clip_offset": offset, "gabor_gain": gain, "band_bonus": bonus,
                "disk_centre": centre, "edge_max": float(edge[inner].max()), "saturated": centre >= 0.999,
                "ratio_edge_to_disk": float(edge[inner].max() / centre) if centre > 0 else None}

    rows = [responses(radius, 0.0, gain, 0.0) for radius in (12, 24, 40) for gain in (0.4, 0.25)]
    rows.append(responses(24, 0.0, 0.4, 1.0 / 255))          # the bonus back
    rows.append(responses(24, 60.0 / 255, 1.0, 1.0 / 255))   # the thesis's profile
    # Per radius, the raw ratio (the two gains agree to 5 decimals)
    raw = {}
    for r in rows:
        if r["clip_offset"] == 0 and r["band_bonus"] == 0 and not r["saturated"]:
            raw.setdefault(r["radius"], []).append(r["ratio_edge_to_disk"])
    return {"rows": rows,
            "raw_ratio_by_radius": {radius: float(np.mean(v)) for radius, v in sorted(raw.items())},
            "gain_invariant": all(max(v) - min(v) < 1e-3 for v in raw.values()),
            "with_thesis_offset": rows[-1]["ratio_edge_to_disk"]}


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


def exp_colour_noise(binary, seeds=None):
    seeds = seeds or NOISE["seeds"]
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
    "stereo-distance": exp_stereo_distance,
    "stereo-rds": exp_stereo_rds,
    "stereo-noise": exp_stereo_noise,
    "stereo-orientations": exp_stereo_orientations,
    "stereo-variance": exp_stereo_variance,
    "stereo-real": exp_stereo_real,
    "text-vs-code": exp_text_vs_code,
    "field": exp_field,
    "world-model": exp_world_model,
    "symmetry-sides": exp_symmetry_sides,
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
    if "stereo-distance" in results:
        rows = results["stereo-distance"]["rows"]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot([r["disparity"] for r in rows], [r["response"] for r in rows], "o-", color="#2a78d6",
                label="depth response on the surface")
        ax.plot([r["disparity"] for r in rows], [r["disparity"] / STEREO_RANGE for r in rows], "--", color="#2a78d6",
                alpha=0.5, label="eq. 5.16: |d| / search range")
        ax.plot([r["disparity"] for r in rows], [r["ground_response"] for r in rows], "s-", color="#8a8985",
                label="ground (disparity 0)")
        ax.set_xlabel("disparity of the surface (px) - nearer to the right")
        ax.set_ylabel("depth response")
        ax.set_title("Abb. 5.28: varying an object's distance")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "stereo_distance.png")
    if "stereo-noise" in results and "stereo-orientations" in results:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        rows = results["stereo-noise"]["rows"]
        ax.plot([r["level"] * 128 for r in rows], [r["within_1px"] for r in rows], "o-", color="#2a78d6",
                label="3 orientations (noise sweep)")
        for count, colour, marker in ((1, "#e34948", "s"), (3, "#2a78d6", "o"), (5, "#1baf7a", "^")):
            sel = [r for r in results["stereo-orientations"]["rows"] if r["orientations"] == count]
            ax.plot([r["noise_sigma"] for r in sel], [r["within_1px"] for r in sel], marker, color=colour, ms=8,
                    mfc="none", label="%d orientation%s" % (count, "" if count == 1 else "s"))
        ax.set_xlabel("independent noise in both images (sigma, grey levels)")
        ax.set_ylabel("surface pixels within 1 px of the true disparity")
        ax.set_ylim(0.6, 1.02)
        ax.set_title("Abb. 5.30 / 5.31: noise and orientations")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "stereo_noise_orientations.png")
    if "stereo-variance" in results:
        r = results["stereo-variance"]
        x = [max(row["threshold"], 0.5) for row in r["rows"]]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(x, [row["surface_correct"] for row in r["rows"]], "o-", color="#2a78d6", label="surface: correct")
        ax.plot(x, [row["faint_surface_correct"] for row in r["rows"]], "^-", color="#1baf7a",
                label="faint surface (1/10 contrast): correct")
        ax.plot(x, [row["flat_band_wrong"] for row in r["rows"]], "s-", color="#e34948",
                label="textureless band: wrong disparity")
        ax.axvline(r["default"], color="#52514e", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("variance threshold (log; dotted: the port's default)")
        ax.set_ylabel("share of pixels")
        ax.set_title("Abb. 5.32: the variance threshold")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "stereo_variance.png")
    if "field" in results:
        sets = (("esab2", "dissertation system (esab2.C)", "#2a78d6"), ("port", "port defaults (until 2026-09)", "#e34948"))
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
        for name, label, colour in sets:
            a, b = results["field"][name]["approach"], results["field"][name]["bifurcation"]
            axes[0].plot(a["distance"], a["cluster_separation"], "o-", ms=3, color=colour, label=label + ": approaching")
            axes[0].plot(b["distance"], b["cluster_separation"], "--", color=colour, alpha=0.6, label=label + ": separating")
        axes[0].plot([0, 30], [0, 30], ":", color="#52514e", lw=0.8)
        axes[0].set_xlabel("distance of the two input maxima (px)")
        axes[0].set_ylabel("distance of the two clusters (0 = one cluster)")
        axes[0].set_title("Abb. 6.5 / 6.10: two maxima")
        axes[0].legend(frameon=False, fontsize=7)
        for name, label, colour in sets:
            n = results["field"][name]["noise"]
            axes[1].plot(n["noise_amplitude"], n["active_outside_zero_mean"], "o-", ms=3, color=colour, label=label)
            axes[1].plot(n["noise_amplitude"], n["active_outside"], "--", color=colour, alpha=0.6,
                         label=label + ", noise added on top")
        axes[1].axvline(1.2, color="#52514e", lw=0.8, ls=":")
        axes[1].set_xlabel("noise amplitude (pulse amplitude 1; dotted: the thesis's 1.2)")
        axes[1].set_ylabel("share of active neurons outside the pulse")
        axes[1].set_title("Abb. 6.6: noise suppression")
        axes[1].legend(frameon=False, fontsize=7)
        save(fig, "field_two_maxima_noise.png")

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
        h = results["field"]["esab2"]["hysteresis"]
        axes[0].plot(h["alpha"], h["position_alpha_rising"], "o-", ms=3, color="#2a78d6", label="alpha rising")
        axes[0].plot(h["alpha"], h["position_alpha_falling"], "s--", ms=3, color="#1baf7a", label="alpha falling")
        axes[0].set_xlabel("alpha (peak at 16: alpha, peak at 48: 1 - alpha)")
        axes[0].set_ylabel("position of the activation cluster")
        axes[0].set_title("Abb. 6.4: hysteresis (dissertation parameters)")
        axes[0].legend(frameon=False, fontsize=8)
        t = results["field"]["esab2"]["tracking"]
        for row in t["rows"]:
            needed = [c if c > 0 else float("nan") for c in row["cycles_needed"]]
            axes[1].plot(t["speed_px_per_frame"], needed, "o-", ms=3, label="amplitude %.1f" % row["amplitude"])
        axes[1].set_xlabel("target speed (px per frame)")
        axes[1].set_ylabel("update cycles per frame needed")
        axes[1].set_title("Abb. 6.9: tracking (dissertation parameters)")
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
        save(fig, "field_hysteresis_tracking.png")
    if "world-model" in results:
        r = results["world-model"]
        statics = sorted({row["static"] for row in r["rows"]})
        fig, axes = plt.subplots(1, len(statics), figsize=(4.0 * len(statics), 3.6), squeeze=False)
        for ax, n_static in zip(axes[0], statics):
            rows = [row for row in r["rows"] if row["static"] == n_static]
            x = [row["dynamic"] for row in rows]
            ax.plot(x, [row["two_stage"] for row in rows], "o-", color="#d1343a", label="two-stage: objects")
            ax.plot(x, [row["conventional"] for row in rows], "s-", color="#1baf7a", label="conventional: objects")
            thesis = r["thesis_fig5"].get(n_static) or r["thesis_fig5"].get(str(n_static))
            if thesis and len(thesis["two_stage"]) == len(x):
                ax.plot(x, thesis["two_stage"], ":", color="#d1343a", lw=1, label="thesis (read off Fig. 5)")
                ax.plot(x, thesis["conventional"], ":", color="#1baf7a", lw=1)
            ax.set_ylim(0, 6.5)
            ax.set_xlabel("dynamic objects")
            ax.set_ylabel("recognized objects (mean over frames)")
            ax.set_title("static objects: %d" % n_static)
            right = ax.twinx()
            right.plot(x, [row["two_stage_error"] for row in rows], "o--", ms=3, color="#2a78d6",
                       label="two-stage: position error")
            right.plot(x, [row["conventional_error"] for row in rows], "s--", ms=3, color="#b04ad1",
                       label="conventional: position error")
            right.set_ylim(0, 14)
            right.set_ylabel("position error (px)")
            if ax is axes[0][0]:
                handles = ax.get_legend_handles_labels()
                more = right.get_legend_handles_labels()
                ax.legend(handles[0] + more[0], handles[1] + more[1], frameon=False, fontsize=6.5, loc="upper left")
        save(fig, "world_model.png")
    if "stereo-real" in results:
        r = results["stereo-real"]
        fig, ax = plt.subplots(figsize=(5.4, 3.5))
        colours = {"tsukuba": "#2a78d6", "venus": "#1baf7a", "sawtooth": "#eda100"}
        for name, scene in r["scenes"].items():
            x = [max(row["threshold"], 1.0) for row in scene["rows"]]
            ax.plot(x, [row["correct"] for row in scene["rows"]], "o-", ms=3, color=colours.get(name, "#52514e"),
                    label="%s: correct" % name)
            ax.plot(x, [row["wrong"] for row in scene["rows"]], "s--", ms=3, color=colours.get(name, "#52514e"),
                    alpha=0.7, label="%s: wrong" % name)
        ax.axvline(3.0, color="#52514e", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("variance threshold (log; dotted: the default)")
        ax.set_ylabel("share of pixels with ground truth")
        ax.set_title("The variance threshold on real stereo pairs")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        save(fig, "stereo_real_threshold.png")
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
    ap.add_argument("--noise-seeds", type=int, default=NOISE["seeds"],
                    help="seeds per level in the noise experiments (default %(default)s)")
    ap.add_argument("--world-model-seed0", type=int, default=WORLD_MODEL["seed0"],
                    help="world-model: first seed (default: the confirmatory block, %(default)s)")
    ap.add_argument("--world-model-runs", type=int, default=WORLD_MODEL["runs"],
                    help="world-model: runs per condition (default %(default)s, as in the thesis)")
    args = ap.parse_args()
    NOISE["seeds"] = args.noise_seeds
    WORLD_MODEL["seed0"] = args.world_model_seed0
    WORLD_MODEL["runs"] = args.world_model_runs

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
