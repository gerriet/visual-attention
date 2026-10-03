#!/usr/bin/env python3
"""COCO-Search18 arm of the priority-map study (roadmap M17, H5): does a
top-down channel make target-present search on natural images more efficient,
and where does the model land against human searchers?

Two model arms, identical except the priority map:

  bottom-up   pure salience scanpath (selection: ior, up to 10 fixations)
  prior       + a category-level spatial prior in the top-down file channel —
              where targets of this category were found in the *training*
              split (Gaussian-accumulated bbox centres). Model-free
              "selection history at the category level": it never sees the
              test image's content, so it lower-bounds what a semantic
              (CLIP-style) channel in the same slot would buy (M18).

Scored on validation trials as mean fixations-to-target (first fixation inside
the target bbox; cap+1 when never) and found@10 rate, against the human
baseline from the same images (correct trials, display->image transformed).

Needs the eval venv (PIL) and data/COCO-Search18 (adapter docstring).
One command:  eval/coco_search.py --limit 150
"""

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import bootstrap_ci, paired_bootstrap  # noqa: E402

PRIOR_W, PRIOR_H = 336, 210  # prior raster (display aspect); resized by the core

# %(cap)s is the model's fixation budget — the same cap the scores use, so the
# model never emits more fixations than found@N / mean-ftt account for.
BOTTOM_UP_YAML = """\
pipeline:
  fusion: weighted-sum
  selection: ior
peaks:
  max_count: %(cap)s
  threshold: 0.05
output:
  display: false
"""

PRIORITY_BLOCK = """\
priority:
  top_down_weight: %(weight)s
  top_down_map: %(map)s
"""
PRIOR_YAML = BOTTOM_UP_YAML + PRIORITY_BLOCK


def import_adapter():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets import cocosearch18
    return cocosearch18


def image_size(path, cache={}):
    from PIL import Image
    if path not in cache:
        with Image.open(path) as img:
            cache[path] = img.size
    return cache[path]


def build_priors(records, out_dir, adapter, sigma_frac=0.08, pooled=False):
    """Per-category spatial priors from training-split target boxes, in
    normalized image-fraction coordinates, written as grayscale PNGs.

    `pooled=True` builds **one** prior from every category's boxes together and
    returns it under every category key. That is the control the closure plan
    asks for: targets of all kinds sit away from the edges, so a pooled prior is
    very nearly a centre prior. If it recovers most of the category prior's
    gain, the H5 COCO result is centre bias wearing a category label.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    accum = {}
    for record in records:
        path = adapter.image_path(record)
        if not path.exists():
            continue
        w, h = image_size(str(path))
        bx, by, bw, bh = record["bbox"]
        cx = int((bx + bw / 2.0) / w * (PRIOR_W - 1))
        cy = int((by + bh / 2.0) / h * (PRIOR_H - 1))
        key = "__pooled__" if pooled else record["task"]
        grid = accum.setdefault(key, np.zeros((PRIOR_H, PRIOR_W), dtype=np.float64))
        if 0 <= cx < PRIOR_W and 0 <= cy < PRIOR_H:
            grid[cy, cx] += 1.0

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    radius = sigma_frac * PRIOR_W
    for task, grid in accum.items():
        img = Image.fromarray((grid / grid.max() * 255).astype("uint8") if grid.max() > 0
                              else grid.astype("uint8"))
        img = img.filter(ImageFilter.GaussianBlur(radius))
        arr = np.asarray(img, dtype=np.float64)
        if arr.max() > 0:
            arr = arr / arr.max() * 255.0
        out_path = os.path.join(out_dir, task.replace(" ", "_") + ".png")
        Image.fromarray(arr.astype("uint8")).save(out_path)
        paths[task] = out_path
    if pooled:
        # Same map for every category, so the caller's `priors[task]` lookups
        # work unchanged.
        pooled_path = paths["__pooled__"]
        tasks = set(r["task"] for r in records)
        paths = {task: pooled_path for task in tasks}
    return paths


def run_model(binary, image, config_text, work_dir, tag):
    os.makedirs(work_dir, exist_ok=True)
    config_path = os.path.join(work_dir, tag + ".yaml")
    with open(config_path, "w") as fh:
        fh.write(config_text)
    result_path = os.path.join(work_dir, tag + ".json")
    cmd = [binary, "--config", config_path, str(image), "--no-display",
           "--emit-json", result_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(result_path) as fh:
        return json.load(fh)["fixations"]


def profile_yaml(path):
    """(bottom-up, prior) config templates from a pipeline profile (--config):
    the profile as it is, and the profile with the category prior in its
    top-down slot. The profile brings its own selection and fixation count;
    fixations beyond --cap are ignored by the scores."""
    with open(path) as fh:
        text = fh.read().replace("%", "%%")
    if "\npriority:" in "\n" + text:
        sys.exit("--config %s already has a priority: block; the prior arm adds its own" % path)
    return text, text.rstrip("\n") + "\n" + PRIORITY_BLOCK


def first_hit(fixations, bbox, cap):
    bx, by, bw, bh = bbox
    for i, f in enumerate(fixations):
        if bx <= f["x"] <= bx + bw and by <= f["y"] <= by + bh:
            return i + 1
    return cap + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/coco_search")
    ap.add_argument("--limit", type=int, default=150,
                    help="unique (image, task) validation trials to run (0 = all)")
    ap.add_argument("--top-down-weight", type=float, default=1.5)
    ap.add_argument("--cap", type=int, default=10, help="model fixation budget")
    ap.add_argument("--config", default=None,
                    help="pipeline profile to score instead of the built-in default feature set "
                         "(e.g. configs/thesis/thesis.yaml) — compares stage-1 profiles on search")
    ap.add_argument("--pooled-control", action="store_true",
                    help="also score a pooled, category-agnostic prior (the H5 control): if it "
                         "recovers most of the category prior's gain, the finding is centre bias")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="sample --limit trials at random with this seed instead of taking the "
                         "first ones, so the score is not dominated by whatever sorts first")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    adapter = import_adapter()
    bottom_up_template, prior_template = (profile_yaml(args.config) if args.config
                                          else (BOTTOM_UP_YAML, PRIOR_YAML))
    if not adapter.available():
        sys.exit("COCO-Search18 not found under data/COCO-Search18 — "
                 "see eval/datasets/cocosearch18.py for download steps")

    print("building category priors from the training split...", file=sys.stderr)
    train = adapter.load_fixations("train")
    priors = build_priors(train, os.path.join(args.out, "priors"), adapter)
    pooled = (build_priors(train, os.path.join(args.out, "priors_pooled"), adapter, pooled=True)
              if args.pooled_control else None)

    validation = adapter.load_fixations("validation")
    trials = adapter.unique_trials(validation)
    keys = list(trials)
    if args.sample_seed is not None:
        random.Random(args.sample_seed).shuffle(keys)
    keys = keys[: args.limit] if args.limit else keys

    human, bottom_up, prior, pooled_arm = [], [], [], []
    skipped = 0
    for n, key in enumerate(keys):
        name, task = key
        records = trials[key]
        image = adapter.image_path(records[0])
        if not image.exists() or task not in priors:
            skipped += 1
            continue
        w, h = image_size(str(image))
        bbox = records[0]["bbox"]

        # Human baseline: correct trials of this image, capped at the same
        # budget as the model (cap+1 = "not found within budget") so the two
        # columns are on one scale. The initial central fixation is stripped
        # inside fixations_to_target, matching the model's first-free-saccade.
        for record in records:
            if record["correct"] != 1:
                continue
            hit = adapter.fixations_to_target(record, w, h)
            human.append(min(hit, args.cap + 1) if hit is not None else args.cap + 1)

        work = os.path.join(args.out, "runs", "%s_%s" % (task.replace(" ", "_"),
                                                         os.path.splitext(name)[0]))
        bottom_up_yaml = bottom_up_template % {"cap": args.cap}
        fx = run_model(args.binary, image, bottom_up_yaml, work, "bottom_up")
        bottom_up.append(first_hit(fx[:args.cap], bbox, args.cap))
        prior_yaml = prior_template % {"cap": args.cap, "weight": args.top_down_weight,
                                       "map": priors[task]}
        fx = run_model(args.binary, image, prior_yaml, work, "prior")
        prior.append(first_hit(fx[:args.cap], bbox, args.cap))
        if pooled:
            pooled_yaml = prior_template % {"cap": args.cap, "weight": args.top_down_weight,
                                            "map": pooled[task]}
            fx = run_model(args.binary, image, pooled_yaml, work, "pooled")
            pooled_arm.append(first_hit(fx[:args.cap], bbox, args.cap))
        if (n + 1) % 25 == 0:
            print("  %d/%d trials" % (n + 1, len(keys)), file=sys.stderr)

    def row(name, values):
        lo, hi = bootstrap_ci(values)
        found = sum(1 for v in values if v <= args.cap) / len(values) if values else 0.0
        return {"name": name, "mean": (sum(values) / len(values)) if values else 0.0,
                "ci": [lo, hi], "found_rate": found, "n": len(values)}

    rows = [row("human", human), row("bottom-up", bottom_up), row("prior", prior)]
    if pooled_arm:
        rows.append(row("pooled prior", pooled_arm))
    header = "%-11s %8s %16s %10s %6s" % ("arm", "mean-ftt", "95% CI", "found@%d" % args.cap, "n")
    print("COCO-Search18 target-present search (fixations-to-target; cap+1 = never)")
    if skipped:
        print("  (skipped %d trials with missing images/priors)" % skipped)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("%-11s %8.2f [%6.2f,%6.2f] %10.2f %6d" % (
            r["name"], r["mean"], r["ci"][0], r["ci"][1], r["found_rate"], r["n"]))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"config": vars(args), "rows": rows}, fh, indent=2)
    if pooled_arm:
        # The comparisons the control exists to make, paired over trials: how
        # much of the category prior's gain survives when the prior is told
        # nothing about the category.
        print()
        print("paired differences (negative = fewer fixations = better):")
        for label, a, b in (("category prior - bottom-up", prior, bottom_up),
                            ("pooled prior   - bottom-up", pooled_arm, bottom_up),
                            ("category prior - pooled   ", prior, pooled_arm)):
            mean, lo, hi = paired_bootstrap(a, b)
            flag = "" if lo <= 0 <= hi else "   (excludes zero)"
            print("  %s  %+6.2f  [%+6.2f, %+6.2f]%s" % (label, mean, lo, hi, flag))
            rows.append({"name": label.strip(), "paired_delta": mean, "delta_ci": [lo, hi]})

    print("summary: %s" % os.path.join(args.out, "summary.json"))
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
