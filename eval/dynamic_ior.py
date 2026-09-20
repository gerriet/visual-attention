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

  staleness       mean, over all frames and visible objects, of the  (lower better)
                  frames since that object was last attended (since
                  it appeared, if never). Unlike revisit waste it
                  does not saturate once every object has been seen
                  once: it scores how evenly attention keeps cycling
                  through the scene for the *whole* video.
  off-object      share of fixations on no object                    (lower better)
  labels/object   distinct object-file labels per attended object    (1.0 = perfect
                  — the tracker's identity switches                    identity)

H1 predicts object-ior >= spatial-ior >= greedy on coverage/latency, and greedy
worst on perseveration.

Two modes. `--scene DIR` scores one generated scene. `--regime NAME` (or `all`)
is the study: it generates `--seeds` scenes per regime from `--seed0`, runs
every arm on each, and reports the mean per arm with the *paired* bootstrap
difference to `spatial-ior` over scenes (the arms see identical scenes).

Arms of the study (what inhibition rides on / how identity is held):
  greedy            no inhibition
  spatial-ior       decaying location tags
  spatial-ior-mc    location tags that drift with the velocity of the object
                    they were left on — the strengthened space-based baseline
  object-ior        object files, the thesis's correspondence (nearest centroid)
  object-ior+aids   + motion-predicted and appearance correspondence
  object-ior+id     + persistent identity (docs/DYNAMIC_IOR_STUDY.md)

Development seeds are 0-9; confirmatory runs use fresh seeds (1000+) with the
configuration frozen beforehand (docs/HYPOTHESIS_CLOSURE_PLAN.md).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import bootstrap_ci, paired_bootstrap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The three regimes of docs/DYNAMIC_IOR_STUDY.md. `ior_radius` None = the
# behavior's default (60 px); the tight radii are what lets a fast object escape
# a location tag — the regime object-based IOR was predicted to win.
REGIMES = {
    "standard": {"scene": ["--objects", "4", "--frames", "40", "--speed", "6", "--occlude"], "ior_radius": None},
    "fast": {"scene": ["--objects", "4", "--frames", "40", "--speed", "40"], "ior_radius": 20},
    "occlusion": {"scene": ["--objects", "4", "--frames", "40", "--speed", "20", "--occlude",
                            "--occlude-len", "10"], "ior_radius": 18},
}

# name -> (behavior, tracking flags, persistent identity)
STUDY_ARMS = {
    "greedy": ("greedy", (), False),
    "spatial-ior": ("spatial-ior", (), False),
    "spatial-ior-mc": ("spatial-ior-mc", (), False),
    "object-ior": ("object-ior", (), False),
    "object-ior+aids": ("object-ior", ("--motion-prediction", "--appearance-matching"), False),
    "object-ior+id": ("object-ior", ("--motion-prediction", "--appearance-matching"), True),
}
REFERENCE_ARM = "spatial-ior"
METRICS = ("coverage", "mean_latency", "staleness", "revisit_waste", "perseveration", "off_object",
           "labels_per_object")

PERSISTENT_IDENTITY_YAML = """
attention_system:
  object_files:
    motion_prediction: true
    appearance_matching: true
    persistent_identity: true
"""


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


def staleness(gt, attended_by_frame):
    """Mean frames since each visible object was last attended (since it first
    appeared, if never), over all frames and visible objects."""
    last_seen, total, count = {}, 0.0, 0
    appeared = {obj["id"]: first_visible_frame(obj) for obj in gt["objects"]}
    for f in range(gt["frames"]):
        oid = attended_by_frame.get(f)
        if oid is not None:
            last_seen[oid] = f
        for i, _, _ in visible_objects(gt, f):
            total += f - last_seen.get(i, appeared[i])
            count += 1
    return total / count if count else 0.0


def score(gt, scanpath, match_radius):
    n_obj = len(gt["objects"])
    n_frames = gt["frames"]
    covered, first_attended = set(), {}
    waste = persev = scored = off = 0
    prev = None
    attended_by_frame, labels = {}, {}

    for entry in sorted(scanpath, key=lambda e: e["frame"]):
        f = entry["frame"]
        oid = attended_object(gt, f, entry["x"], entry["y"], match_radius)
        if oid is None:
            off += 1
            prev = None
            continue
        attended_by_frame[f] = oid
        if "label" in entry:
            labels.setdefault(oid, set()).add(entry["label"])
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
        "staleness": staleness(gt, attended_by_frame),
        "off_object": off / len(scanpath) if scanpath else 0.0,
        "labels_per_object": (sum(len(v) for v in labels.values()) / len(labels)) if labels else 0.0,
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


def identity_config(config):
    """A temp copy of `config` with persistent identity switched on."""
    with open(config) as fh:
        text = fh.read()
    if "attention_system:" in text:
        sys.exit("%s already has an attention_system: block; the +id arm adds its own" % config)
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text + PERSISTENT_IDENTITY_YAML)
    handle.close()
    return handle.name


def run_regime(binary, config, regime, seeds, seed0, out_dir, match_radius, arms):
    """{arm: [per-scene metric dict]} over `seeds` generated scenes."""
    preset = REGIMES[regime]
    id_config = identity_config(config)
    rows = {arm: [] for arm in arms}
    try:
        for seed in range(seed0, seed0 + seeds):
            scene = os.path.join(out_dir, regime, "scene_%d" % seed)
            subprocess.run([sys.executable, os.path.join(REPO, "tools", "make_dynamic_scene.py"), "--out", scene,
                            "--seed", str(seed)] + preset["scene"], check=True, stdout=subprocess.DEVNULL)
            gt = load_json(os.path.join(scene, "gt.json"))
            for arm in arms:
                behavior, tracking, persistent = STUDY_ARMS[arm]
                scanpath = run_arm(binary, scene, id_config if persistent else config, behavior,
                                   os.path.join(scene, "arms", arm), preset["ior_radius"], tracking)
                rows[arm].append(score(gt, scanpath, match_radius))
    finally:
        os.unlink(id_config)
    return rows


def summarize_regime(rows, reference=REFERENCE_ARM):
    """Per arm and metric: mean, its CI over scenes, and the paired difference
    to the reference arm over the same scenes."""
    summary = {}
    for arm, scenes in rows.items():
        summary[arm] = {}
        for metric in METRICS:
            values = [r[metric] for r in scenes]
            entry = {"mean": sum(values) / len(values), "ci": list(bootstrap_ci(values)), "n": len(values)}
            if arm != reference and reference in rows:
                entry["minus_reference"] = list(paired_bootstrap(values, [r[metric] for r in rows[reference]]))
            summary[arm][metric] = entry
    return summary


def format_regime(regime, summary, reference=REFERENCE_ARM, metrics=("mean_latency", "staleness", "revisit_waste")):
    summary = {arm: m for arm, m in summary.items() if arm != "scenes"}
    n = next(iter(summary.values()))["coverage"]["n"]
    lines = ["", "regime: %s  (%d scenes; differences are paired, arm minus %s, 95%% CI)" % (regime, n, reference)]
    header = "%-16s %8s %8s %8s" % ("arm", "cover", "off-obj", "labels") + "".join(
        "  %-32s" % m for m in metrics)
    lines += [header, "-" * len(header)]
    for arm, m in summary.items():
        line = "%-16s %8.3f %8.3f %8.2f" % (arm, m["coverage"]["mean"], m["off_object"]["mean"],
                                           m["labels_per_object"]["mean"])
        for metric in metrics:
            cell = "%6.2f" % m[metric]["mean"]
            if "minus_reference" in m[metric]:
                cell += "  %+.2f [%+.2f, %+.2f]" % tuple(m[metric]["minus_reference"])
            line += "  %-32s" % cell
        lines.append(line)
    return "\n".join(lines)


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
    ap.add_argument("--scene", default=None, help="score one scene dir (frames + gt.json)")
    ap.add_argument("--regime", default=None, choices=sorted(REGIMES) + ["all"],
                    help="the study: generate --seeds scenes of this regime and run every arm on each")
    ap.add_argument("--seeds", type=int, default=10, help="scenes per regime (study mode)")
    ap.add_argument("--seed0", type=int, default=0,
                    help="first scene seed; 0-9 are development seeds, confirmatory runs use 1000+")
    ap.add_argument("--arms", default=",".join(STUDY_ARMS), help="study arms, comma-separated")
    ap.add_argument("--reference", default=REFERENCE_ARM, choices=sorted(STUDY_ARMS),
                    help="the arm the paired differences are taken against")
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--config", default="configs/thesis/attend.yaml",
                    help="pipeline config; H1 is a claim of the thesis, so the thesis-track "
                         "second-stage profile is the default (docs/adr/0005)")
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
    if args.regime:
        arms = [a.strip() for a in args.arms.split(",")]
        unknown = [a for a in arms if a not in STUDY_ARMS]
        if unknown:
            sys.exit("unknown arm(s): %s (available: %s)" % (", ".join(unknown), ", ".join(STUDY_ARMS)))
        report = {}
        for regime in (sorted(REGIMES) if args.regime == "all" else [args.regime]):
            rows = run_regime(args.binary, args.config, regime, args.seeds, args.seed0, args.out,
                              args.match_radius, arms)
            report[regime] = summarize_regime(rows, args.reference)
            report[regime]["scenes"] = rows  # per-scene metrics: any other paired comparison, later
            print(format_regime(regime, report[regime], args.reference))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "summary.json"), "w") as fh:
            json.dump({"config": vars(args), "regimes": report}, fh, indent=2)
        print("\nsummary: %s" % os.path.join(args.out, "summary.json"))
        return
    if not args.scene:
        sys.exit("nothing to do: pass --scene DIR or --regime NAME")
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
