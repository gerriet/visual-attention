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

PRIOR_W, PRIOR_H = 336, 210  # prior raster (display aspect); resized by the core

BOTTOM_UP_YAML = """\
pipeline:
  fusion: weighted-sum
  selection: ior
peaks:
  max_count: 10
  threshold: 0.05
output:
  display: false
"""

PRIOR_YAML = BOTTOM_UP_YAML + """\
priority:
  top_down_weight: %(weight)s
  top_down_map: %(map)s
"""


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


def build_priors(records, out_dir, adapter, sigma_frac=0.08):
    """Per-category spatial priors from training-split target boxes, in
    normalized image-fraction coordinates, written as grayscale PNGs."""
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
        grid = accum.setdefault(record["task"], np.zeros((PRIOR_H, PRIOR_W), dtype=np.float64))
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


def first_hit(fixations, bbox, cap):
    bx, by, bw, bh = bbox
    for i, f in enumerate(fixations):
        if bx <= f["x"] <= bx + bw and by <= f["y"] <= by + bh:
            return i + 1
    return cap + 1


def bootstrap_ci(values, iterations=2000, seed=0):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/coco_search")
    ap.add_argument("--limit", type=int, default=150,
                    help="unique (image, task) validation trials to run (0 = all)")
    ap.add_argument("--top-down-weight", type=float, default=1.5)
    ap.add_argument("--cap", type=int, default=10, help="model fixation budget")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    adapter = import_adapter()
    if not adapter.available():
        sys.exit("COCO-Search18 not found under data/COCO-Search18 — "
                 "see eval/datasets/cocosearch18.py for download steps")

    print("building category priors from the training split...", file=sys.stderr)
    train = adapter.load_fixations("train")
    priors = build_priors(train, os.path.join(args.out, "priors"), adapter)

    validation = adapter.load_fixations("validation")
    trials = adapter.unique_trials(validation)
    keys = list(trials)[: args.limit] if args.limit else list(trials)

    human, bottom_up, prior = [], [], []
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

        # Human baseline: correct trials of this image (1-based fixation index).
        for record in records:
            if record["correct"] != 1:
                continue
            hit = adapter.fixations_to_target(record, w, h)
            human.append(hit if hit is not None else args.cap + 1)

        work = os.path.join(args.out, "runs", "%s_%s" % (task.replace(" ", "_"),
                                                         os.path.splitext(name)[0]))
        fx = run_model(args.binary, image, BOTTOM_UP_YAML, work, "bottom_up")
        bottom_up.append(first_hit(fx, bbox, args.cap))
        prior_yaml = PRIOR_YAML % {"weight": args.top_down_weight, "map": priors[task]}
        fx = run_model(args.binary, image, prior_yaml, work, "prior")
        prior.append(first_hit(fx, bbox, args.cap))
        if (n + 1) % 25 == 0:
            print("  %d/%d trials" % (n + 1, len(keys)), file=sys.stderr)

    def row(name, values):
        lo, hi = bootstrap_ci(values)
        found = sum(1 for v in values if v <= args.cap) / len(values) if values else 0.0
        return {"name": name, "mean": (sum(values) / len(values)) if values else 0.0,
                "ci": [lo, hi], "found_rate": found, "n": len(values)}

    rows = [row("human", human), row("bottom-up", bottom_up), row("prior", prior)]
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
    print("summary: %s" % os.path.join(args.out, "summary.json"))
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
