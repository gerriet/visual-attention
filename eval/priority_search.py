#!/usr/bin/env python3
"""Priority-map search study (roadmap M17): does adding top-down and
history/value channels to the master map make task-driven search faster? [H5]

Synthetic arm: seeded dynamic scenes with one red target among coloured
distractors (tools/make_dynamic_scene.py --target). Four arms, identical
except for the priority-map configuration:

  bottom-up      pure salience (the thesis map)
  feature-td     Guided-Search dimension weighting: colour feature boosted.
                 Honest expectation: little help when *all* objects are
                 colourful — weights select feature dimensions, not values.
  target-td      dense target-colour channel (priority: target_color: red)
  full-priority  target-td + object-value + location-history facilitation

Metrics per run, aggregated over seeds with bootstrap 95% CIs:
  time-to-target   frames until the focus first lands on the target
                   (never found = full penalty of `frames`)
  hold-fraction    after first acquisition, fraction of remaining frames
                   focused on the target (facilitation vs IOR tension)

One command: eval/priority_search.py --seeds 20   (needs the eval venv for
scene generation; scoring is stdlib-only)
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import bootstrap_ci  # noqa: E402

ARMS = ["bottom-up", "feature-td", "target-td", "full-priority"]

BASE_FEATURES = """\
features:
  color: {weight: %(color_weight)s}
  eccentricity: {weight: 1.0}
  symmetry: {weight: 1.0}
  onset: {weight: 1.0}
  intensity: {enabled: false}
  orientation: {enabled: false}
"""

PIPELINE = """\
pipeline:
  fusion: weighted-sum
  selection: nms
"""

PRIORITY_TARGET = """\
priority:
  top_down_weight: 1.5
  target_color: red
"""

PRIORITY_FULL = """\
priority:
  top_down_weight: 1.5
  target_color: red
  object_value_weight: 0.6
  location_history_weight: 0.2
"""


def arm_config(arm):
    color_weight = "3.0" if arm == "feature-td" else "1.0"
    text = PIPELINE + BASE_FEATURES % {"color_weight": color_weight}
    if arm == "target-td":
        text += PRIORITY_TARGET
    elif arm == "full-priority":
        text += PRIORITY_FULL
    return text + "output:\n  display: false\n"


def make_scene(out_dir, seed, args):
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                        "tools", "make_dynamic_scene.py"),
           "--out", out_dir, "--frames", str(args.frames), "--objects", str(args.objects),
           "--speed", str(args.speed), "--seed", str(seed), "--target", "0"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(out_dir, "gt.json")) as fh:
        return json.load(fh)


def run_attend(binary, scene_dir, config_path, scan_path):
    cmd = [binary, "--attend", scene_dir, "--config", config_path,
           "--no-save-frames", "--emit-scanpath", scan_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(scan_path) as fh:
        return json.load(fh)["scanpath"]


def target_positions(gt):
    target = next(o for o in gt["objects"] if o["id"] == gt["target"])
    return {p["frame"]: (p["x"], p["y"]) for p in target["positions"] if p["visible"]}


def score(gt, scanpath, match_radius):
    """(time_to_target, hold_fraction) for one run."""
    positions = target_positions(gt)
    n_frames = gt["frames"]
    on_target = {}
    for entry in scanpath:
        pos = positions.get(entry["frame"])
        if pos is None:
            continue
        d2 = (entry["x"] - pos[0]) ** 2 + (entry["y"] - pos[1]) ** 2
        on_target[entry["frame"]] = d2 <= match_radius * match_radius

    hit_frames = sorted(f for f, hit in on_target.items() if hit)
    if not hit_frames:
        return float(n_frames), 0.0
    first = hit_frames[0]
    later = [f for f in on_target if f > first]
    hold = (sum(1 for f in later if on_target[f]) / len(later)) if later else 1.0
    return float(first), hold


def study(args):
    results = {arm: {"time": [], "hold": []} for arm in ARMS}
    for seed in range(args.seeds):
        scene_dir = os.path.join(args.out, "scenes", "seed_%03d" % seed)
        gt = make_scene(scene_dir, seed, args)
        for arm in ARMS:
            config_path = os.path.join(args.out, "configs", arm + ".yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as fh:
                fh.write(arm_config(arm))
            scan_path = os.path.join(args.out, "runs", "seed_%03d_%s.json" % (seed, arm))
            os.makedirs(os.path.dirname(scan_path), exist_ok=True)
            scanpath = run_attend(args.binary, scene_dir, config_path, scan_path)
            t, hold = score(gt, scanpath, args.match_radius)
            results[arm]["time"].append(t)
            results[arm]["hold"].append(hold)
        print("  seed %d/%d done" % (seed + 1, args.seeds), file=sys.stderr)
    return results


def summarize(results, penalty):
    summary = {}
    for arm in ARMS:
        times, holds = results[arm]["time"], results[arm]["hold"]
        t_lo, t_hi = bootstrap_ci(times)
        h_lo, h_hi = bootstrap_ci(holds)
        # score() returns exactly `penalty` (= frames) for a never-found run;
        # anything strictly below it is a genuine acquisition.
        summary[arm] = {
            "time_to_target": sum(times) / len(times), "time_ci": [t_lo, t_hi],
            "hold_fraction": sum(holds) / len(holds), "hold_ci": [h_lo, h_hi],
            "found_rate": sum(1 for t in times if t < penalty) / len(times),
            "runs": len(times),
        }
    return summary


def format_table(summary, frames):
    header = "%-14s %18s %18s" % ("arm", "time-to-target", "hold-fraction")
    lines = ["priority-map search (H5), %d seeds, penalty=%d frames" % (
        summary[ARMS[0]]["runs"], frames), header, "-" * len(header)]
    for arm in ARMS:
        s = summary[arm]
        lines.append("%-14s %8.1f [%4.1f,%5.1f] %8.2f [%4.2f,%4.2f]" % (
            arm, s["time_to_target"], s["time_ci"][0], s["time_ci"][1],
            s["hold_fraction"], s["hold_ci"][0], s["hold_ci"][1]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/priority_search")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--match-radius", type=float, default=28.0)
    ap.add_argument("--json", action="store_true", help="also print raw metrics as JSON")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)

    results = study(args)
    summary = summarize(results, args.frames)
    print(format_table(summary, args.frames))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"config": vars(args), "summary": summary, "raw": results}, fh, indent=2)
    print("summary: %s" % os.path.join(args.out, "summary.json"))
    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
