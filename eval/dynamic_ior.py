#!/usr/bin/env python3
"""Dynamic-IOR study (roadmap M12): does object-based inhibition of return beat
space-based (and no) IOR in a dynamic scene?  [H1]

Runs the AttentionSystem over a generated scene (tools/make_dynamic_scene.py)
under three behaviors that are identical except in what they inhibit —
`greedy` (none), `spatial-ior` (locations), `object-ior` (objects) — then scores
each emitted scanpath against the ground truth on:

  coverage        distinct objects ever attended / total          (higher better)
  mean latency    frames from an object first appearing to first   (lower better)
                  being attended (never-attended = full penalty)
  revisit waste   fraction of fixations that re-hit an already-     (lower better)
                  seen object while an unseen one was visible
  perseveration   fraction of fixations on the same object as the   (lower better)
                  previous frame

H1 predicts object-ior >= spatial-ior >= greedy on coverage/latency, and greedy
worst on perseveration.
"""

import argparse
import json
import math
import os
import subprocess
import sys


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def visible_objects(gt, frame):
    """List of (id, x, y) for objects visible at `frame`."""
    out = []
    for obj in gt["objects"]:
        if frame < len(obj["positions"]):
            p = obj["positions"][frame]
            if p["visible"]:
                out.append((obj["id"], p["x"], p["y"]))
    return out


def first_visible_frame(obj):
    for p in obj["positions"]:
        if p["visible"]:
            return p["frame"]
    return 0


def attended_object(gt, frame, x, y, match_radius):
    """The visible object whose centroid is nearest (x, y) within match_radius."""
    best_id, best_d2 = None, match_radius * match_radius
    for oid, ox, oy in visible_objects(gt, frame):
        d2 = (ox - x) ** 2 + (oy - y) ** 2
        if d2 <= best_d2:
            best_d2, best_id = d2, oid
    return best_id


def score(gt, scanpath, match_radius):
    n_obj = len(gt["objects"])
    n_frames = gt["frames"]
    covered, first_attended = set(), {}
    waste = persev = scored = 0
    prev = None

    for entry in sorted(scanpath, key=lambda e: e["frame"]):
        f = entry["frame"]
        oid = attended_object(gt, f, entry["x"], entry["y"], match_radius)
        if oid is None:
            prev = None
            continue
        scored += 1
        if oid in covered:
            uncovered_visible = [i for (i, _, _) in visible_objects(gt, f) if i not in covered]
            if uncovered_visible:
                waste += 1
        else:
            covered.add(oid)
            first_attended.setdefault(oid, f)
        if prev is not None and oid == prev:
            persev += 1
        prev = oid

    latencies = []
    for obj in gt["objects"]:
        fv = first_visible_frame(obj)
        if obj["id"] in first_attended:
            latencies.append(max(0, first_attended[obj["id"]] - fv))
        else:
            latencies.append(n_frames)  # never attended: full penalty
    return {
        "coverage": len(covered) / n_obj if n_obj else 0.0,
        "mean_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "revisit_waste": waste / scored if scored else 0.0,
        "perseveration": persev / scored if scored else 0.0,
    }


def run_arm(binary, scene_dir, config, behavior, out_dir, ior_radius=None, tracking=()):
    os.makedirs(out_dir, exist_ok=True)
    scan_path = os.path.join(out_dir, "scanpath_%s.json" % behavior)
    cmd = [binary, "--attend", scene_dir, "--behavior", behavior,
           "--emit-scanpath", scan_path, "--output", os.path.join(out_dir, behavior)]
    if config:
        cmd += ["--config", config]
    if ior_radius is not None:
        cmd += ["--ior-radius", str(ior_radius)]
    cmd += list(tracking)  # e.g. --motion-prediction / --appearance-matching
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return load_json(scan_path)["scanpath"]


ARMS = ["greedy", "spatial-ior", "object-ior"]


def study(binary, scene_dir, config, out_dir, match_radius, ior_radius=None, tracking=()):
    gt = load_json(os.path.join(scene_dir, "gt.json"))
    rows = {}
    for behavior in ARMS:
        scanpath = run_arm(binary, scene_dir, config, behavior, out_dir, ior_radius, tracking)
        rows[behavior] = score(gt, scanpath, match_radius)
    return gt, rows


def format_table(rows):
    header = "%-14s %10s %12s %14s %14s" % (
        "behavior", "coverage", "mean_lat", "revisit_waste", "perseveration")
    lines = [header, "-" * len(header)]
    labels = {"greedy": "greedy (none)", "spatial-ior": "spatial-ior", "object-ior": "object-ior"}
    for arm in ARMS:
        m = rows[arm]
        lines.append("%-14s %10.3f %12.2f %14.3f %14.3f" % (
            labels[arm], m["coverage"], m["mean_latency"], m["revisit_waste"], m["perseveration"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene dir (frames + gt.json)")
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--config", default="configs/attend.yaml")
    ap.add_argument("--out", default="results/dynamic_ior")
    ap.add_argument("--match-radius", type=float, default=28.0,
                    help="max distance (px) from focus to an object centroid to count as attended")
    ap.add_argument("--ior-radius", type=float, default=None,
                    help="spatial-IOR tag radius (px); tight radius + fast motion is where object-IOR wins")
    ap.add_argument("--motion-prediction", action="store_true",
                    help="object-file correspondence tracks predicted position (holds identity under motion)")
    ap.add_argument("--appearance-matching", action="store_true",
                    help="fold appearance (region colour) into correspondence (holds identity through crossings)")
    ap.add_argument("--json", action="store_true", help="also print the raw metrics as JSON")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    if not os.path.exists(os.path.join(args.scene, "gt.json")):
        sys.exit("no gt.json in %s (generate with tools/make_dynamic_scene.py)" % args.scene)

    tracking = []
    if args.motion_prediction:
        tracking.append("--motion-prediction")
    if args.appearance_matching:
        tracking.append("--appearance-matching")
    _, rows = study(args.binary, args.scene, args.config, args.out, args.match_radius,
                    args.ior_radius, tuple(tracking))
    print(format_table(rows))
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
