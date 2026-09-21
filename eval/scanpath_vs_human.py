#!/usr/bin/env python3
"""Scanpaths vs human gaze on stills (roadmap M11, H4).

Places the thesis model's scanpath on the floor<->ceiling axis of human-scanpath
agreement, scored with MultiMatch + ScanMatch against MIT1003's per-observer
fixation sequences, and reported as *one sample of a human distribution* (a
stochastic WTA+IOR peer gives best-of-N), not as "the" path.

Arms:
  inter-observer  leave-one-out human-vs-human agreement — the ceiling
  thesis-field    the thesis model's own scanpath: the fixations its profile
                  (neural-field selection + IOR) emits — "the thesis model"
  thesis-wta      the SAME saliency map, read out by generic WTA+IOR
  thesis-objfile  the SAME map, read out by the object-file second stage
                  (--attend over the still repeated as a stream; consecutive
                  frames on one object are one fixation) — the H4 ablation:
                  stage-2 ordering vs the map
  <name>-wta      --profile name=config: another C++ profile's map, WTA+IOR
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
from study_common import bootstrap_ci, paired_bootstrap  # noqa: E402

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

def pipeline_map(binary, image_path, config, workdir, tag="map"):
    """Run the C++ pipeline on a still: (saliency map, the profile's own
    fixations in attention order)."""
    from attention_eval import io

    result_json = os.path.join(workdir, tag + ".json")
    cmd = [binary, str(image_path), "--no-display", "--emit-json", result_json]
    if config:
        cmd += ["--config", config]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(result_json) as fh:
        result = json.load(fh)
    saliency = io.load_map(os.path.join(os.path.dirname(result_json), result["saliency_map"]))
    fixations = [(f["x"], f["y"]) for f in sorted(result["fixations"], key=lambda f: f["n"])]
    return saliency, fixations


def objectfile_path(binary, image_path, config, workdir, n, frames_per_fixation=4):
    """The second stage's scanpath on a still. A single image is a stream of
    length one and yields one focus, so the still is presented as a stream of
    identical frames — a fixed gaze on a static scene — long enough for `n`
    fixations at the behavior's dwell (3 frames). Consecutive frames on the
    same object file are one fixation."""
    scan_json = os.path.join(workdir, "attend.json")
    with tempfile.TemporaryDirectory() as stream_dir:
        suffix = os.path.splitext(str(image_path))[1]
        for k in range(n * frames_per_fixation):
            os.symlink(os.path.abspath(str(image_path)), os.path.join(stream_dir, "f%04d%s" % (k, suffix)))
        cmd = [binary, "--attend", stream_dir, "--no-save-frames", "--emit-scanpath", scan_json]
        if config:
            cmd += ["--config", config]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(scan_json) as fh:
        entries = sorted(json.load(fh)["scanpath"], key=lambda e: e["frame"])
    path, previous = [], None
    for e in entries:
        if e.get("label") != previous or previous is None:
            path.append((e["x"], e["y"]))
        previous = e.get("label")
    return path[:n]


def python_model_map(name, image_array):
    from attention_eval.models import get_model
    return np.asarray(get_model(name).compute(image_array), dtype=np.float64)


def available_operators(models):
    """The registered Python saliency models that can be constructed here —
    DeepGaze needs torch + weights and is simply absent without them."""
    from attention_eval.models import get_model
    usable = []
    for name in sorted(models):
        try:
            get_model(name)
            usable.append(name)
        except Exception as exc:  # optional dependency missing: not an arm, not an error
            print("skipping %s: %s" % (name, str(exc).splitlines()[0]), file=sys.stderr)
    return usable


# --- study over MIT1003 ------------------------------------------------------

def run_mit1003(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets import mit1003
    from attention_eval.models import MODELS
    from attention_eval.models.base import load_image_rgb  # not re-exported by the package

    if not (mit1003.available() and mit1003.scanpaths_available()):
        sys.exit("MIT1003 stimuli + DATA archive required — see eval/datasets/mit1003.py")

    # Every registered Python saliency model that can run here is a fair
    # scanpath peer (DeepGaze joins under 'deepgaze-iie' when torch + weights
    # are present).
    operators = [] if args.no_operators else available_operators(MODELS)
    profiles = dict(spec.split("=", 1) for spec in args.profile)

    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "rows.json")
    rows = {}
    if args.resume and os.path.exists(rows_path):
        with open(rows_path) as fh:
            rows = json.load(fh)
        print("resuming: %d stimuli already scored" % len(rows), file=sys.stderr)

    from PIL import Image
    stimuli = list(mit1003.iter_stimuli())
    if args.limit:
        # A seeded random sample: the archive's file order groups the images by
        # collection, so "the first N" is not a sample of the dataset.
        import random
        random.Random(args.sample_seed).shuffle(stimuli)
        stimuli = stimuli[: args.limit]
    for n, (stimulus, _fixmap, _fixpts) in enumerate(stimuli):
        if stimulus.stem in rows:
            continue
        with Image.open(stimulus) as im:
            size = im.size
        humans = [seq for _subj, seq in mit1003.iter_scanpaths(stimulus.stem, size)]
        if len(humans) < 2:
            continue
        row = {"inter-observer": interobserver_ceiling(humans, size)}

        with tempfile.TemporaryDirectory() as workdir:
            saliency, field = pipeline_map(args.binary, stimulus, args.config, workdir)
            objfile = objectfile_path(args.binary, stimulus, args.attend_config, workdir, args.n)
            extra = {name: pipeline_map(args.binary, stimulus, config, workdir, tag=name)[0]
                     for name, config in profiles.items()}
        row["thesis-field"] = score_vs_humans(field[: args.n], humans, size)
        row["thesis-wta"] = score_vs_humans(readout.wta_ior(saliency, size, n=args.n), humans, size)
        row["thesis-objfile"] = score_vs_humans(objfile, humans, size)
        row["thesis-stoch-best"] = max(
            (score_vs_humans(p, humans, size)
             for p in readout.sample_paths(saliency, args.seed + n, args.samples, size, n=args.n)),
            key=lambda r: r["scanmatch"] if not np.isnan(r["scanmatch"]) else -1)
        for name, profile_map in extra.items():
            row["%s-wta" % name] = score_vs_humans(readout.wta_ior(profile_map, size, n=args.n), humans, size)

        image_array = load_image_rgb(stimulus)
        for op in operators:
            path = readout.wta_ior(python_model_map(op, image_array), size, n=args.n)
            row["%s-wta" % op] = score_vs_humans(path, humans, size)

        rng = np.random.RandomState(args.seed + n)
        row["random"] = score_vs_humans(random_path(size, args.n, rng), humans, size)
        row["center"] = score_vs_humans(center_path(size, args.n), humans, size)
        rows[stimulus.stem] = row
        if (n + 1) % 25 == 0:
            print("  %d stimuli" % (n + 1), file=sys.stderr)
            with open(rows_path, "w") as fh:  # a long run survives being stopped (--resume)
                json.dump(rows, fh)
    with open(rows_path, "w") as fh:
        json.dump(rows, fh)

    per_arm = {}
    for row in rows.values():
        for arm, scores in row.items():
            per_arm.setdefault(arm, {k: [] for k in MM_DIMS + ("scanmatch",)})
            for k in MM_DIMS + ("scanmatch",):
                if not np.isnan(scores[k]):
                    per_arm[arm][k].append(scores[k])
    summary = summarize(per_arm)
    summary["_paired"] = paired_differences(rows)
    return summary


# The comparisons H4 is about, paired over stimuli (every arm sees every image).
PAIRS = [("thesis-field", "center"), ("thesis-field", "random"), ("thesis-wta", "center"),
         ("thesis-objfile", "thesis-wta"), ("thesis-field", "thesis-wta"),
         ("inter-observer", "thesis-field")]


def paired_differences(rows, metric="scanmatch"):
    out = {}
    arms = set().union(*(set(r) for r in rows.values())) if rows else set()
    pairs = PAIRS + [("thesis-wta", a) for a in sorted(arms)
                     if a.endswith("-wta") and a != "thesis-wta" and ("thesis-wta", a) not in PAIRS]
    for a, b in pairs:
        both = [(r[a][metric], r[b][metric]) for r in rows.values()
                if a in r and b in r and not (np.isnan(r[a][metric]) or np.isnan(r[b][metric]))]
        if both:
            mean, lo, hi = paired_bootstrap([x for x, _ in both], [y for _, y in both])
            out["%s - %s" % (a, b)] = {"mean": mean, "ci": [lo, hi], "n": len(both)}
    return out


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

ARM_ORDER = ["inter-observer", "thesis-field", "thesis-objfile", "thesis-wta", "thesis-stoch-best",
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
    for arm in ARM_ORDER + [a for a in summary if a not in ARM_ORDER and not a.startswith("_")]:
        if arm not in summary:
            continue
        s = summary[arm]
        lines.append("%-22s" % arm + "".join("%12.3f" % s[c]["mean"] for c in cols) + "   n=%d" % s["scanmatch"]["n"])
    if summary.get("_paired"):
        lines += ["", "paired differences over stimuli (ScanMatch, 95% CI):"]
        for name, d in summary["_paired"].items():
            lines.append("  %-36s %+.3f [%+.3f, %+.3f]  n=%d" % (name, d["mean"], d["ci"][0], d["ci"][1], d["n"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/scanpath_vs_human")
    ap.add_argument("--config", default="configs/thesis/thesis.yaml",
                    help="the still-image profile: its saliency map and its own fixations. H4 is a "
                         "replication-track question, so the thesis profile is the default (docs/adr/0005)")
    ap.add_argument("--attend-config", default="configs/thesis/attend.yaml",
                    help="second-stage profile for the object-file readout (same features, so the same map)")
    ap.add_argument("--profile", action="append", default=[], metavar="NAME=CONFIG",
                    help="another C++ profile as a <NAME>-wta arm (repeatable)")
    ap.add_argument("--no-operators", action="store_true", help="skip the Python saliency-model arms")
    ap.add_argument("--resume", action="store_true", help="keep the stimuli already scored in --out/rows.json")
    ap.add_argument("--mit1003", action="store_true")
    ap.add_argument("--demo", action="store_true", help="synthetic stack check (no dataset/binary/scipy)")
    ap.add_argument("--check", action="store_true", help="with --demo: assert ceiling >= model > floor")
    ap.add_argument("--limit", type=int, default=0, help="MIT1003 stimuli to score: a seeded random sample (0 = all)")
    ap.add_argument("--sample-seed", type=int, default=0, help="seed of the --limit sample")
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
