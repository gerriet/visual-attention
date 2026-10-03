#!/usr/bin/env python3
"""Learned top-down feature weights on COCO-Search18 (roadmap M20, H8).

M17 gave the priority map a top-down slot and filled it two ways, both chosen
by hand: a target colour we name, and a category-level spatial prior. This arm
fills the same slot by a *rule* instead — VOCUS's (Frintrop, Backer & Rome,
KI 2005, §2.2):

    w_i = mean of feature map i inside the target region / mean outside it

so a channel counts to the degree it *separates* target from background. The
weights are learned per category from the **training** split and combined by
the geometric mean (they are ratios), then applied as
`S = (1 - t) * S_bu + t * (excitation - inhibition)`.

Why this is a fair fight with M17's prior arm: both are learned from the
training split and both are blind to the test image's content. One says *where*
targets of this category tend to be, the other says *which channels* tell this
category apart. They can also be combined, which is the third arm.

The experimental bar (docs/HYPOTHESIS_CLOSURE_PLAN.md) is respected by
splitting the work: t is chosen on held-out **training** trials (--tune), and
the chosen t is then run once on **validation** (--confirm). Neither phase
sees the other's trials.

    eval/top_down_weights.py --tune                  # pick t on training trials
    eval/top_down_weights.py --confirm --factor 0.5  # score it on validation

Needs the eval venv (PIL) and data/COCO-Search18. Weight learning calls the
binary's --learn-weights mode, so the rule has one implementation, in C++.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import bootstrap_ci, paired_bootstrap  # noqa: E402

# Imported rather than reimplemented: the arms must be scored by exactly the
# code that scores M17's arms, or the comparison is between two harnesses.
import coco_search  # noqa: E402

BASE_YAML = """\
pipeline:
  fusion: weighted-sum
  selection: ior
peaks:
  max_count: %(cap)s
  threshold: 0.05
output:
  display: false
"""

WEIGHTS_BLOCK = """\
priority:
  top_down_factor: %(factor)s
  top_down_weights:
%(weights)s
"""

# The prior arm's block, so "weights + prior" can be scored as one config.
PRIOR_LINES = """\
  top_down_weight: %(prior_weight)s
  top_down_map: %(map)s
"""


def weights_yaml(weights, indent="    "):
    return "".join("%s%s: %.6f\n" % (indent, name, value) for name, value in sorted(weights.items()))


def learn_category_weights(binary, adapter, task, records, examples, config, out_dir, whole_box=False):
    """One weight vector for a category, from `examples` training images.

    Returns (weights, used) or (None, 0) when no example was usable.
    """
    specs = []
    for record in records[:examples]:
        image = adapter.image_path(record)
        if not image.exists():
            continue
        x, y, w, h = [int(round(v)) for v in record["bbox"]]
        if w <= 0 or h <= 0:
            continue
        specs.append("%s:%d,%d,%d,%d" % (image, x, y, w, h))
    if not specs:
        return None, 0

    out_path = os.path.join(out_dir, "weights_%s.yaml" % task.replace(" ", "_"))
    cmd = [binary, "--learn-weights"] + specs + ["--out", out_path]
    if config:
        cmd += ["--config", config]
    if whole_box:
        cmd.append("--whole-box")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    weights = {}
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or ":" not in line or line.endswith(":"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if key in ("top_down_factor", "priority", "top_down_weights"):
                continue
            try:
                weights[key] = float(value)
            except ValueError:
                continue
    return (weights or None), len(specs)


def blind_weights(weights_by_task):
    """The across-category geometric mean: what every category's weight vector
    agrees on, with the category-specific part averaged out.

    This is the control that decides whether the rule learned anything about
    *this target* as opposed to about objects in general. If a category-blind
    vector scores like the matched one, the weights carry no category
    information, however legible they look.
    """
    if not weights_by_task:
        return {}
    names = set()
    for w in weights_by_task.values():
        names.update(w)
    blind = {}
    for name in names:
        logs = [math.log(w[name]) for w in weights_by_task.values() if w.get(name, 0) > 0]
        if logs:
            blind[name] = math.exp(sum(logs) / len(logs))
    return blind


def mismatched_weights(weights_by_task):
    """Each category gets a *different* category's weights — a derangement, so
    no category keeps its own. The sharper half of the same control: if search
    is as good with the wrong category's weights, the matched weights are not
    doing the work."""
    tasks = sorted(weights_by_task)
    if len(tasks) < 2:
        return dict(weights_by_task)
    # Rotate by one: deterministic, and a derangement for any list of >= 2.
    return {task: weights_by_task[tasks[(i + 1) % len(tasks)]] for i, task in enumerate(tasks)}


def base_yaml(profile_path):
    """The config text every arm starts from.

    With --config this must be the *profile*, not the built-in default: the
    weights are learned on the profile's channels, and weights naming channels
    the scoring run does not have would silently match nothing — every arm would
    then be the bottom-up arm and the study would report a tidy null.
    """
    if not profile_path:
        return BASE_YAML
    with open(profile_path) as fh:
        text = fh.read().replace("%", "%%")
    if "\npriority:" in "\n" + text:
        sys.exit("--config %s already has a priority: block; the arms add their own" % profile_path)
    return text.rstrip("\n") + "\n"


def score_arms(binary, adapter, trials, keys, weights_by_task, factors, cap, out_dir,
               priors=None, prior_weight=1.5, controls=False, profile=None):
    """Run every arm over the same trials. Returns {arm: [fixations-to-target]}."""
    arms = {"bottom-up": []}
    for t in factors:
        arms["weights t=%.2f" % t] = []
    blind = mismatched = None
    if controls:
        blind = blind_weights(weights_by_task)
        mismatched = mismatched_weights(weights_by_task)
        for t in factors:
            arms["blind t=%.2f" % t] = []
            arms["mismatched t=%.2f" % t] = []
    if priors:
        arms["prior"] = []
        for t in factors:
            arms["weights+prior t=%.2f" % t] = []

    skipped = 0
    for n, key in enumerate(keys):
        name, task = key
        records = trials[key]
        image = adapter.image_path(records[0])
        if not image.exists() or task not in weights_by_task:
            skipped += 1
            continue
        if priors and task not in priors:
            skipped += 1
            continue
        bbox = records[0]["bbox"]
        work = os.path.join(out_dir, "runs", "%s_%s" % (task.replace(" ", "_"), os.path.splitext(name)[0]))
        wy = weights_yaml(weights_by_task[task])
        base = base_yaml(profile)

        def run(tag, yaml_text):
            fx = coco_search.run_model(binary, image, yaml_text, work, tag)
            return coco_search.first_hit(fx[:cap], bbox, cap)

        arms["bottom-up"].append(run("bottom_up", base % {"cap": cap}))
        if priors:
            arms["prior"].append(run("prior", base % {"cap": cap} + "priority:\n" +
                                     PRIOR_LINES % {"prior_weight": prior_weight, "map": priors[task]}))
        for t in factors:
            cfg = base % {"cap": cap} + WEIGHTS_BLOCK % {"factor": t, "weights": wy}
            arms["weights t=%.2f" % t].append(run("w%.2f" % t, cfg))
            if controls:
                arms["blind t=%.2f" % t].append(run("b%.2f" % t, base % {"cap": cap} +
                                                    WEIGHTS_BLOCK % {"factor": t,
                                                                     "weights": weights_yaml(blind)}))
                arms["mismatched t=%.2f" % t].append(
                    run("m%.2f" % t, base % {"cap": cap} +
                        WEIGHTS_BLOCK % {"factor": t, "weights": weights_yaml(mismatched[task])}))
            if priors:
                combined = cfg.rstrip("\n") + "\n" + PRIOR_LINES % {"prior_weight": prior_weight,
                                                                    "map": priors[task]}
                arms["weights+prior t=%.2f" % t].append(run("wp%.2f" % t, combined))
        if (n + 1) % 25 == 0:
            print("  %d/%d trials" % (n + 1, len(keys)), file=sys.stderr)
    return arms, skipped


def report(arms, cap, baseline="bottom-up"):
    rows = []
    for name, values in arms.items():
        if not values:
            continue
        lo, hi = bootstrap_ci(values)
        row = {"name": name, "mean": sum(values) / len(values), "ci": [lo, hi],
               "found_rate": sum(1 for v in values if v <= cap) / len(values), "n": len(values)}
        if name != baseline and arms.get(baseline) and len(arms[baseline]) == len(values):
            # Paired over trials: every arm saw the same images, so the paired
            # difference is the quantity with the smaller variance and the only
            # one that answers "did this arm help on these trials".
            diff, dlo, dhi = paired_bootstrap(values, arms[baseline])
            row["delta_vs_baseline"] = diff
            row["delta_ci"] = [dlo, dhi]
        rows.append(row)

    header = "%-22s %8s %16s %10s %6s %22s" % ("arm", "mean-ftt", "95% CI", "found@%d" % cap, "n",
                                               "paired delta vs bottom-up")
    print(header)
    print("-" * len(header))
    for r in rows:
        delta = ""
        if "delta_vs_baseline" in r:
            delta = "%+7.2f [%+6.2f,%+6.2f]" % (r["delta_vs_baseline"], r["delta_ci"][0], r["delta_ci"][1])
        print("%-22s %8.2f [%6.2f,%6.2f] %10.2f %6d %22s" % (
            r["name"], r["mean"], r["ci"][0], r["ci"][1], r["found_rate"], r["n"], delta))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/top_down_weights")
    ap.add_argument("--examples", type=int, default=10,
                    help="training images per category used to learn its weights")
    ap.add_argument("--example-seed", type=int, default=None,
                    help="sample the learning examples at random with this seed instead of taking "
                         "the first ones in filename order, which is arbitrary and makes the "
                         "weights depend on how the dataset happens to sort")
    ap.add_argument("--limit", type=int, default=120, help="trials to score (0 = all)")
    ap.add_argument("--cap", type=int, default=10, help="model fixation budget")
    ap.add_argument("--factor", type=float, action="append", default=None,
                    help="top-down factor t; repeatable (default: a sweep)")
    ap.add_argument("--config", default=None,
                    help="pipeline profile — used for BOTH weight learning and scoring, since weights "
                         "name that profile's channels (e.g. configs/split_channels.yaml)")
    ap.add_argument("--whole-box", action="store_true",
                    help="learn from the whole bounding box instead of the salient region in it "
                         "(the ablation of VOCUS's own region step)")
    ap.add_argument("--controls", action="store_true",
                    help="also score the category-blind and mismatched-category weight vectors — "
                         "the test of whether the weights carry anything category-specific")
    ap.add_argument("--with-prior", action="store_true",
                    help="also score M17's category prior, and weights+prior together")
    ap.add_argument("--tune", action="store_true",
                    help="score on held-out TRAINING trials (choose t here)")
    ap.add_argument("--confirm", action="store_true",
                    help="score on VALIDATION trials (report here, once, with t already chosen)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.tune == args.confirm:
        sys.exit("choose exactly one of --tune (training trials) and --confirm (validation trials)")
    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    factors = args.factor if args.factor else [0.25, 0.5, 0.75, 1.0]

    adapter = coco_search.import_adapter()
    if not adapter.available():
        sys.exit("COCO-Search18 not found under data/COCO-Search18 — "
                 "see eval/datasets/cocosearch18.py for download steps")
    os.makedirs(args.out, exist_ok=True)

    train = adapter.load_fixations("train")
    train_trials = adapter.unique_trials(train)

    # Learn on the first `--examples` training trials of each category; hold the
    # rest of the training split out, so --tune never scores an image a weight
    # was learned from.
    by_task = {}
    for key in train_trials:
        by_task.setdefault(key[1], []).append(key)
    if args.example_seed is not None:
        rng = random.Random(args.example_seed)
        for keys in by_task.values():
            rng.shuffle(keys)
    learn_keys = {task: keys[: args.examples] for task, keys in by_task.items()}
    learned_images = set(k for keys in learn_keys.values() for k in keys)

    print("learning weights from %d training image(s) per category..." % args.examples, file=sys.stderr)
    weights_by_task = {}
    for task, keys in sorted(learn_keys.items()):
        records = [train_trials[k][0] for k in keys]
        weights, used = learn_category_weights(args.binary, adapter, task, records, args.examples,
                                               args.config, args.out, args.whole_box)
        if weights:
            weights_by_task[task] = weights
            print("  %-12s %d examples  %s" % (
                task, used, "  ".join("%s %.2f" % (k, v) for k, v in sorted(weights.items()))),
                file=sys.stderr)

    if args.tune:
        pool = [k for k in train_trials if k not in learned_images]
        trials = train_trials
        split_name = "training (held out from weight learning)"
    else:
        validation = adapter.load_fixations("validation")
        trials = adapter.unique_trials(validation)
        pool = list(trials)
        split_name = "validation"
    keys = pool[: args.limit] if args.limit else pool

    priors = None
    if args.with_prior:
        print("building category priors from the training split...", file=sys.stderr)
        priors = coco_search.build_priors(train, os.path.join(args.out, "priors"), adapter)

    arms, skipped = score_arms(args.binary, adapter, trials, keys, weights_by_task, factors,
                               args.cap, args.out, priors, controls=args.controls, profile=args.config)

    print("\nCOCO-Search18, %s split — fixations-to-target (cap+1 = never found)" % split_name)
    print("weights learned per category from %d training image(s)%s" % (
        args.examples, ", whole box" if args.whole_box else ", salient region in box"))
    if skipped:
        print("  (skipped %d trials with missing images or weights)" % skipped)
    rows = report(arms, args.cap)

    summary = {"config": vars(args), "split": split_name, "rows": rows,
               "weights": weights_by_task}
    if args.controls:
        summary["blind_weights"] = blind_weights(weights_by_task)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsummary: %s" % os.path.join(args.out, "summary.json"))
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
