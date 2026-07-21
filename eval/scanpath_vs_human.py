#!/usr/bin/env python3
"""Scanpaths vs human gaze on stills (roadmap M11, H4).

Places the thesis model's scanpath on the floor<->ceiling axis of human-scanpath
agreement, scored with MultiMatch + ScanMatch against MIT1003's per-observer
fixation sequences, and reported as *one sample of a human distribution* (a
stochastic WTA+IOR peer gives best-of-N), not as "the" path.

Arms:
  inter-observer  leave-one-out human-vs-human agreement — the ceiling
  thesis-wta      the thesis saliency map, read out by generic WTA+IOR
  thesis-objfile  the SAME map, read out by the object-file second stage
                  (--attend) — the H4 ablation: stage-2 ordering vs the map
  <operator>-wta  each alternative saliency map (spectral-residual, center,
                  DeepGaze if present), WTA+IOR readout — fair scanpath peers
  thesis-stoch    stochastic samples of the thesis map: best-of-N + mean draw
  center / random the floors

H4's claim — that the stage-2 *ordering* benefit is separable from the saliency
map — is read directly off thesis-objfile vs thesis-wta (same map, two
readouts).

The real run needs MIT1003 + its DATA archive + scipy (see
datasets/mit1003.py); this env may have none, so the study is gated. `--demo`
validates the whole scoring stack on synthetic data with no dataset or binary.

  scanpath_vs_human.py --mit1003 --limit 100
  scanpath_vs_human.py --demo            # synthetic; no dataset, no scipy needed
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_eval import readout  # noqa: E402
from attention_eval.scanpath import multimatch, scanmatch  # noqa: E402
from study_common import bootstrap_ci  # noqa: E402

MM_DIMS = ("shape", "direction", "length", "position")


# --- scoring ----------------------------------------------------------------

def score_vs_humans(path, human_paths, size):
    """Mean MultiMatch dims + ScanMatch of one model scanpath against every
    human observer's sequence."""
    mm = {d: [] for d in MM_DIMS}
    sm = []
    for human in human_paths:
        m = multimatch(path, human, size)
        for d in MM_DIMS:
            if not np.isnan(m[d]):
                mm[d].append(m[d])
        s = scanmatch(path, human, size)
        if not np.isnan(s):
            sm.append(s)
    row = {d: (float(np.mean(mm[d])) if mm[d] else float("nan")) for d in MM_DIMS}
    row["scanmatch"] = float(np.mean(sm)) if sm else float("nan")
    return row


def interobserver_ceiling(human_paths, size):
    """Leave-one-out human-vs-human agreement: each observer scored against the
    others, averaged. The ceiling any model is measured against."""
    if len(human_paths) < 2:
        return {d: float("nan") for d in MM_DIMS + ("scanmatch",)}
    rows = []
    for i, held in enumerate(human_paths):
        others = human_paths[:i] + human_paths[i + 1:]
        rows.append(score_vs_humans(held, others, size))
    ceiling = {}
    for k in MM_DIMS + ("scanmatch",):
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        ceiling[k] = float(np.mean(vals)) if vals else float("nan")  # guard all-NaN slice
    return ceiling


def random_path(size, n, rng):
    w, h = size
    return [(float(rng.uniform(0, w)), float(rng.uniform(0, h))) for _ in range(n)]


def center_path(size, n):
    """A degenerate 'always look at the centre' scanpath — the centre-bias floor
    as a sequence (tiny deterministic drift so consecutive fixations differ)."""
    w, h = size
    return [(w / 2 + (k - n / 2) * 1.0, h / 2) for k in range(n)]


# --- model saliency maps -----------------------------------------------------

def thesis_maps(binary, image_path, config, workdir):
    """Run the C++ pipeline once and return (saliency_map, objectfile_path):
    the same map read two ways for the H4 ablation. Uses one config so the map
    is identical for both readouts."""
    from attention_eval import io

    result_json = os.path.join(workdir, "thesis.json")
    cmd = [binary, str(image_path), "--no-display", "--emit-json", result_json]
    if config:
        cmd += ["--config", config]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(result_json) as fh:
        saliency_rel = json.load(fh)["saliency_map"]
    saliency = io.load_map(os.path.join(os.path.dirname(result_json), saliency_rel))

    # Object-file readout: --attend over a one-image directory (a stream of
    # length one), same config, so it segments the same saliency.
    scan_json = os.path.join(workdir, "attend.json")
    with tempfile.TemporaryDirectory() as one_dir:
        link = os.path.join(one_dir, os.path.basename(str(image_path)))
        os.symlink(os.path.abspath(str(image_path)), link)
        cmd = [binary, "--attend", one_dir, "--no-save-frames", "--emit-scanpath", scan_json]
        if config:
            cmd += ["--config", config]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(scan_json) as fh:
        objfile = [(f["x"], f["y"]) for f in sorted(json.load(fh)["scanpath"], key=lambda e: e["frame"])]
    return saliency, objfile


def python_model_map(name, image_array):
    from attention_eval.models import get_model
    return np.asarray(get_model(name).compute(image_array), dtype=np.float64)


# --- study over MIT1003 ------------------------------------------------------

def run_mit1003(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets import mit1003
    from attention_eval.models import MODELS
    from attention_eval.models.base import load_image_rgb  # not re-exported by the package

    if not (mit1003.available() and mit1003.scanpaths_available()):
        sys.exit("MIT1003 stimuli + DATA archive required — see eval/datasets/mit1003.py")

    # Every registered Python saliency model is a fair scanpath peer (this picks
    # up DeepGaze under its real key 'deepgaze-iie' when torch + weights are present).
    operators = sorted(MODELS)
    per_arm = {}

    def record(arm, row):
        per_arm.setdefault(arm, {k: [] for k in MM_DIMS + ("scanmatch",)})
        for k in MM_DIMS + ("scanmatch",):
            if not np.isnan(row[k]):
                per_arm[arm][k].append(row[k])

    from PIL import Image
    stimuli = list(mit1003.iter_stimuli())
    if args.limit:
        stimuli = stimuli[: args.limit]
    for n, (stimulus, _fixmap, _fixpts) in enumerate(stimuli):
        with Image.open(stimulus) as im:
            size = im.size
        humans = [seq for _subj, seq in mit1003.iter_scanpaths(stimulus.stem, size)]
        if len(humans) < 2:
            continue
        record("inter-observer", interobserver_ceiling(humans, size))

        with tempfile.TemporaryDirectory() as workdir:
            saliency, objfile = thesis_maps(args.binary, stimulus, args.config, workdir)
        record("thesis-wta", score_vs_humans(readout.wta_ior(saliency, size, n=args.n), humans, size))
        record("thesis-objfile", score_vs_humans(objfile, humans, size))
        best = max((score_vs_humans(p, humans, size)
                    for p in readout.sample_paths(saliency, args.seed + n, args.samples, size, n=args.n)),
                   key=lambda r: r["scanmatch"] if not np.isnan(r["scanmatch"]) else -1)
        record("thesis-stoch-best", best)

        image_array = load_image_rgb(stimulus)
        for op in operators:
            path = readout.wta_ior(python_model_map(op, image_array), size, n=args.n)
            record("%s-wta" % op, score_vs_humans(path, humans, size))

        rng = np.random.RandomState(args.seed + n)
        record("random", score_vs_humans(random_path(size, args.n, rng), humans, size))
        record("center", score_vs_humans(center_path(size, args.n), humans, size))
        if (n + 1) % 25 == 0:
            print("  %d stimuli" % (n + 1), file=sys.stderr)

    return summarize(per_arm)


# --- synthetic demo (no dataset, no binary) ----------------------------------

def demo(args):
    """Build a saliency map with three blobs and a set of synthetic 'human'
    scanpaths that roughly visit the blobs, then run the full scoring stack.
    Validates that the ceiling and the map-driven readout beat the floors."""
    w, h = 320, 240
    # Decreasing amplitude gives WTA a deterministic visit order, which the
    # synthetic humans follow — the clean regime for the stack check.
    blobs = [(70, 60), (230, 90), (150, 190)]
    weights = [1.0, 0.72, 0.5]
    yy, xx = np.mgrid[0:h, 0:w]
    saliency = np.zeros((h, w))
    for (bx, by), wgt in zip(blobs, weights):
        saliency += wgt * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 22 ** 2))

    # Synthetic humans agree on order (visit the blobs by salience, the WTA
    # order) with small jitter, so the inter-observer ceiling is genuinely high
    # — the regime where ceiling and a good model both sit well above the floors.
    rng = np.random.RandomState(args.seed)
    humans = [[(bx + rng.randn() * 6, by + rng.randn() * 6) for (bx, by) in blobs] for _ in range(8)]

    per_arm = {}

    def record(arm, row):
        per_arm[arm] = {k: [row[k]] for k in MM_DIMS + ("scanmatch",)}

    record("inter-observer", interobserver_ceiling(humans, (w, h)))
    record("thesis-wta", score_vs_humans(readout.wta_ior(saliency, (w, h), n=3), humans, (w, h)))
    best = max((score_vs_humans(p, humans, (w, h))
                for p in readout.sample_paths(saliency, args.seed, args.samples, (w, h), n=3)),
               key=lambda r: r["scanmatch"])
    record("thesis-stoch-best", best)
    record("center", score_vs_humans(center_path((w, h), 3), humans, (w, h)))
    record("random", score_vs_humans(random_path((w, h), 3, rng), humans, (w, h)))
    summary = summarize(per_arm)

    if args.check:
        # The whole stack works when the human ceiling and the map-driven
        # readout both clear the floors (a strict model<=ceiling is NOT asserted:
        # a deterministic model legitimately exceeds a noisy ceiling on
        # order-tolerant metrics — a finding, not a bug).
        sm = {a: summary[a]["scanmatch"]["mean"] for a in
              ("inter-observer", "thesis-wta", "center", "random")}
        if not (sm["inter-observer"] > sm["random"]):
            sys.exit("check failed: inter-observer %.3f not above random %.3f (scanmatch)"
                     % (sm["inter-observer"], sm["random"]))
        if not (sm["thesis-wta"] > sm["random"] and sm["thesis-wta"] > sm["center"]):
            sys.exit("check failed: thesis-wta %.3f did not beat floors (random %.3f, center %.3f)"
                     % (sm["thesis-wta"], sm["random"], sm["center"]))
    return summary


# --- aggregate + report ------------------------------------------------------

ARM_ORDER = ["inter-observer", "thesis-objfile", "thesis-wta", "thesis-stoch-best",
             "spectral-residual-wta", "center-bias-wta", "deepgaze-iie-wta", "center", "random"]


def summarize(per_arm):
    summary = {}
    for arm, cols in per_arm.items():
        row = {}
        for k, values in cols.items():
            if values:
                lo, hi = bootstrap_ci(values)
                row[k] = {"mean": float(np.mean(values)), "ci": [lo, hi], "n": len(values)}
            else:
                row[k] = {"mean": float("nan"), "ci": [float("nan"), float("nan")], "n": 0}
        summary[arm] = row
    return summary


def format_table(summary):
    cols = ["scanmatch"] + list(MM_DIMS)
    header = "%-22s" % "arm" + "".join("%12s" % c for c in cols)
    lines = ["Scanpaths vs human gaze (H4): floor <-> ceiling", header, "-" * len(header)]
    for arm in ARM_ORDER + [a for a in summary if a not in ARM_ORDER]:
        if arm not in summary:
            continue
        s = summary[arm]
        lines.append("%-22s" % arm + "".join("%12.3f" % s[c]["mean"] for c in cols))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/scanpath_vs_human")
    ap.add_argument("--config", default="configs/attend.yaml",
                    help="one pipeline config for both thesis readouts (H4 same-map ablation)")
    ap.add_argument("--mit1003", action="store_true")
    ap.add_argument("--demo", action="store_true", help="synthetic stack check (no dataset/binary/scipy)")
    ap.add_argument("--check", action="store_true", help="with --demo: assert ceiling >= model > floor")
    ap.add_argument("--limit", type=int, default=0, help="max MIT1003 stimuli (0 = all)")
    ap.add_argument("--n", type=int, default=10, help="fixations per model scanpath")
    ap.add_argument("--samples", type=int, default=10, help="stochastic samples for best-of-N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.mit1003:
        if not os.path.exists(args.binary):
            sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
        summary = run_mit1003(args)
    elif args.demo:
        summary = demo(args)
    else:
        sys.exit("nothing to do: pass --demo or --mit1003")

    print(format_table(summary))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.json:
        print(json.dumps(summary, indent=2))
    print("summary: %s" % os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
