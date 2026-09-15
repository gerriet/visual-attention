#!/usr/bin/env python3
"""Object files as a video token cache (roadmap M19, H7 = H1 x H6).

A video VLM pays visual tokens for every frame it is shown. The attention
system's second stage keeps *object files* — tracked identities with
persistent memory — so a front-end can show the VLM each object once, at
native resolution, instead of re-sending frames. Spatial memory can't: it
cannot tell "the same object moved" from "a new object arrived", so a
location-keyed front-end re-sends moved objects (wasted tokens) and skips
newcomers that appear where an earlier crop was (misses) — the moving-object
failure mode H1 was built around. This harness measures that asymmetry as VLM
accuracy at a matched token budget.

Scenes are synthetic (tools/make_dynamic_scene.py --tags --late): coloured
disks move and bounce, some arrive mid-video, and each carries a small code
("K7") legible only at native resolution. One multiple-choice question per
object — "what code is on the <colour> disk?" — so accuracy is the share of
objects whose detail actually reached the VLM. Arms (same VLM, same question):

  frames-full     T evenly spaced native frames — the naive reference (~10x)
  frames-uniform  the same T frames downsampled to the crop arms' budget — the
                  standard budget-matched video-VLM input
  space-ior       one overview frame + K native-res crops at the fixations of
                  the spatial-IOR behavior, deduplicated by *location* (the
                  only memory a first-stage-only system has)
  object-ior      one overview frame + K crops, one per *object file* (first
                  fixation of each tracked identity) — ours
  oracle          one overview frame + one crop per ground-truth object at its
                  first visible frame — the upper bound

With --gt-identity, two decomposition arms take a behavior's fixations but
deduplicate them by *ground-truth* identity (a perfect tracker that also knows
which regions are objects): space-ior-gtid, object-ior-gtid. object-ior vs
object-ior-gtid is what the object-file tracker loses to label switches;
space-ior vs object-ior-gtid is what identity-keyed memory is worth.

The attention system runs on a downscaled copy of the video (--proc-max-side;
the defaults reproduce the M12 scene geometry), crops come from the native
frames. Tokens: the provider-independent patch estimate always, the backend's
real count with --count-tokens. The mock backend answers iff the arm's images
show the code whole and legible (--min-target-px), so the harness runs end to
end in CI; M12's scanpath scores (coverage, latency, waste) ride along.

  eval/vlm_video.py --seeds 5 --count-tokens            # local Ollama (Qwen)
  eval/vlm_video.py --seeds 1 --backend mock --check    # CI smoke
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dynamic_ior  # noqa: E402
from study_common import bootstrap_ci  # noqa: E402
from vlm_backends import VISUAL_TOKEN_PATCH, create_backend  # noqa: E402
from vlm_frontend import _box_visible, make_view  # noqa: E402

ARMS = ("frames-full", "frames-uniform", "space-ior", "space-ior-gtid", "object-ior", "object-ior-gtid", "oracle")
DEFAULT_ARMS = ("frames-full", "frames-uniform", "space-ior", "object-ior", "oracle")
BEHAVIORS = {"space-ior": "spatial-ior", "object-ior": "object-ior"}
TAG_OVERLAP = 0.9  # a crop that clips the code doesn't deliver it


def _pil():
    from PIL import Image
    return Image


def frame_path(scene_dir, f):
    return os.path.join(scene_dir, "frame_%04d.png" % f)


class Frames:
    """Native frames of a scene, loaded on first use."""

    def __init__(self, scene_dir):
        self.scene_dir, self._cache = scene_dir, {}

    def __call__(self, f):
        if f not in self._cache:
            with _pil().open(frame_path(self.scene_dir, f)) as im:
                self._cache[f] = im.convert("RGB")
        return self._cache[f]


def make_scene(scene_dir, seed, args):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "make_dynamic_scene.py"), "--out", scene_dir,
           "--frames", str(args.frames), "--objects", str(args.objects),
           "--width", str(args.width), "--height", str(args.height), "--radius", str(args.radius),
           "--speed", str(args.speed), "--seed", str(seed),
           "--tags", "--tag-size", str(args.tag_size), "--late", str(args.late)]
    if args.occlude:
        cmd.append("--occlude")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return dynamic_ior.load_json(os.path.join(scene_dir, "gt.json"))


def downscale(scene_dir, proc_dir, n_frames, proc_max_side):
    """Write the bounded-size copy the attention system runs on; returns the
    native -> proc scale."""
    Image = _pil()
    os.makedirs(proc_dir, exist_ok=True)
    scale = 1.0
    for f in range(n_frames):
        with Image.open(frame_path(scene_dir, f)) as im:
            w, h = im.size
            scale = min(1.0, proc_max_side / max(w, h))
            small = im.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            small.save(frame_path(proc_dir, f))
    return scale


def attend(binary, proc_dir, config, behavior, out_dir, scale, tracking):
    """Run one stage-2 behavior over the proc video; its scanpath, frame-ordered,
    in native coordinates."""
    scanpath = dynamic_ior.run_arm(binary, proc_dir, config, behavior, out_dir, tracking=tracking)
    return [{"frame": e["frame"], "label": e["label"], "x": e["x"] / scale, "y": e["y"] / scale}
            for e in sorted(scanpath, key=lambda e: e["frame"])]


def object_picks(scanpath, k):
    """One crop per object file — the first fixation of each tracked identity.
    Identity is the memory: a moved object keeps its file and isn't re-sent; a
    newcomer gets a new file and is."""
    seen, picks = set(), []
    for e in scanpath:
        if len(picks) >= k:
            break
        if e["label"] in seen:
            continue
        seen.add(e["label"])
        picks.append((e["frame"], e["x"], e["y"]))
    return picks


def location_picks(scanpath, k, radius):
    """One crop per new *location* — a fixation farther than `radius` from
    every earlier crop. Blind to identity, it re-sends a moved object and skips
    a newcomer arriving where an earlier crop was."""
    picks = []
    for e in scanpath:
        if len(picks) >= k:
            break
        if any(math.hypot(e["x"] - x, e["y"] - y) < radius for _, x, y in picks):
            continue
        picks.append((e["frame"], e["x"], e["y"]))
    return picks


def oracle_picks(gt, k):
    """One crop per ground-truth object, at its first visible frame."""
    firsts = []
    for obj in gt["objects"]:
        p = next((p for p in obj["positions"] if p["visible"]), None)
        if p:
            firsts.append((p["frame"], p["x"], p["y"]))
    return sorted(firsts)[:k]


def identity_picks(scanpath, gt, k, radius):
    """A behavior's fixations deduplicated by ground-truth identity: the first
    fixation on each object (within `radius`), fixations on no object skipped."""
    seen, picks = set(), []
    for e in scanpath:
        if len(picks) >= k:
            break
        oid = dynamic_ior.attended_object(gt, e["frame"], e["x"], e["y"], radius)
        if oid is None or oid in seen:
            continue
        seen.add(oid)
        picks.append((e["frame"], e["x"], e["y"]))
    return picks


def identity_stats(scanpath, gt, radius):
    """How well object-file labels track identity: distinct labels per attended
    object (1.0 = perfect), and the share of fixations on no object."""
    labels, off = {}, 0
    for e in scanpath:
        oid = dynamic_ior.attended_object(gt, e["frame"], e["x"], e["y"], radius)
        if oid is None:
            off += 1
        else:
            labels.setdefault(oid, set()).add(e["label"])
    return {"labels_per_object": _mean([len(s) for s in labels.values()]) or 0.0,
            "off_object": off / len(scanpath) if scanpath else 0.0}


def tag_box(obj, frame):
    """The object's code extent (native px) at `frame`, or None if hidden."""
    p = obj["positions"][frame]
    if not p["visible"]:
        return None
    tw, th = obj["tag_box"]
    return (p["x"] - tw / 2, p["y"] - th / 2, p["x"] + tw / 2, p["y"] + th / 2)


def delivered(views, obj, min_px):
    """Does some view show this object's code whole and at least min_px on its
    shorter side on screen? (The mock's legibility oracle.)"""
    for view in views:
        box = tag_box(obj, view["frame"])
        if box and _box_visible([view], box, min_px, TAG_OVERLAP):
            return True
    return False


def shown_objects(view, gt):
    """The objects whose codes a crop shows whole (at any size)."""
    shown = set()
    for obj in gt["objects"]:
        box = tag_box(obj, view["frame"])
        if box and _box_visible([view], box, 0, TAG_OVERLAP):
            shown.add(obj["id"])
    return shown


def crop_view(image, frame, cx, cy, side):
    """A native-res square crop centred on (cx, cy), shifted to stay inside."""
    w, h = image.size
    x0 = int(max(0, min(w - side, cx - side / 2)))
    y0 = int(max(0, min(h - side, cy - side / 2)))
    box = (x0, y0, min(w, x0 + side), min(h, y0 + side))
    return dict(make_view(image.crop(box), box, 1.0), frame=frame)


def scaled_view(image, frame, scale):
    """The whole frame at `scale` (never upsampled)."""
    Image = _pil()
    w, h = image.size
    scale = min(1.0, scale)
    small = image if scale >= 1.0 else image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                                     Image.LANCZOS)
    return dict(make_view(small, (0, 0, w, h), scale), frame=frame)


def build_arms(frames, gt, picks, arms_wanted, backend, params):
    """Return {arm: list_of_views}. The crop arms share one overview frame; the
    frames-uniform arm is budget-matched to the nominal crop arm (overview + K
    crops), so it doesn't depend on how many crops any one arm found."""
    n, w, h = gt["frames"], gt["width"], gt["height"]
    mid = n // 2
    overview = scaled_view(frames(mid), mid, params["global_side"] / max(w, h))
    arms = {arm: [overview] + [crop_view(frames(f), f, x, y, params["crop_side"]) for f, x, y in arm_picks]
            for arm, arm_picks in picks.items()}
    t = params["frames_shown"]
    shown = sorted({round(i * (n - 1) / (t - 1)) for i in range(t)}) if t > 1 else [mid]
    if "frames-full" in arms_wanted:
        arms["frames-full"] = [scaled_view(frames(f), f, params["full_max_side"] / max(w, h)) for f in shown]
    if "frames-uniform" in arms_wanted:
        budget = (backend.estimate_visual_tokens(overview["image"].size)
                  + params["k"] * backend.estimate_visual_tokens((params["crop_side"], params["crop_side"])))
        r = VISUAL_TOKEN_PATCH * math.sqrt(budget / len(shown)) / math.sqrt(w * h)
        arms["frames-uniform"] = [scaled_view(frames(f), f, r) for f in shown]
    return {arm: arms[arm] for arm in ARMS if arm in arms}


def describe(arm, views, n_frames):
    """What the images are, for the prompt."""
    if arm.startswith("frames"):
        return ("The images are frames %s of a %d-frame video of moving coloured disks."
                % (", ".join(str(v["frame"]) for v in views), n_frames))
    text = "Image 1 is an overview (frame %d) of a %d-frame video of moving coloured disks." % (
        views[0]["frame"], n_frames)
    if len(views) > 1:
        text += " The other images are close-up crops from frames %s." % ", ".join(
            str(v["frame"]) for v in views[1:])
    return text


def make_questions(gt, seed):
    """Per object: its code among three distractors (other objects' codes
    first — the plausible confusions — then random codes)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from make_dynamic_scene import TAG_DIGITS, TAG_LETTERS
    rng = random.Random(seed)
    tags = [o["tag"] for o in gt["objects"]]
    questions = []
    for obj in gt["objects"]:
        distractors = [t for t in tags if t != obj["tag"]]
        rng.shuffle(distractors)
        distractors = distractors[:3]
        while len(distractors) < 3:
            code = rng.choice(TAG_LETTERS) + rng.choice(TAG_DIGITS)
            if code != obj["tag"] and code not in distractors:
                distractors.append(code)
        choices = distractors + [obj["tag"]]
        rng.shuffle(choices)
        questions.append({"object": obj["id"], "color": obj["color"], "answer": obj["tag"], "choices": choices})
    return questions


def run_scene(backend, seed, args):
    scene_dir = os.path.join(args.out, "scene_%d" % seed)
    gt = make_scene(scene_dir, seed, args)
    proc_dir = os.path.join(scene_dir, "proc")
    scale = downscale(scene_dir, proc_dir, gt["frames"], args.proc_max_side)
    tracking = () if args.no_tracking_aids else ("--motion-prediction", "--appearance-matching")
    k = args.k or args.objects
    params = {"k": k, "crop_side": args.crop_side, "global_side": args.global_side,
              "frames_shown": args.frames_shown, "full_max_side": args.full_max_side}
    arms_wanted = [a.strip() for a in args.arms.split(",")]

    radius = args.match_radius / scale  # attributing a fixation to an object (native px)
    picks, scanpath_scores = {}, {}
    for arm, behavior in BEHAVIORS.items():
        if arm not in arms_wanted and arm + "-gtid" not in arms_wanted:
            continue
        path = attend(args.binary, proc_dir, args.config, behavior,
                      os.path.join(scene_dir, "attend"), scale, tracking)
        scanpath_scores[behavior] = dict(dynamic_ior.score(gt, path, radius), **identity_stats(path, gt, radius))
        if arm in arms_wanted:
            picks[arm] = (object_picks(path, k) if arm == "object-ior"
                          else location_picks(path, k, args.crop_side / 2))
        if arm + "-gtid" in arms_wanted:
            picks[arm + "-gtid"] = identity_picks(path, gt, k, radius)
    if "oracle" in arms_wanted:
        picks["oracle"] = oracle_picks(gt, k)
    arms = build_arms(Frames(scene_dir), gt, picks, arms_wanted, backend, params)

    objects = {o["id"]: o for o in gt["objects"]}
    record = {"seed": seed, "objects": len(objects), "scanpath": scanpath_scores, "arms": {}}
    for arm, views in arms.items():
        crops = views[1:] if arm in picks else []
        covered = set().union(*(shown_objects(v, gt) for v in crops))
        rows, real = [], []
        for q in make_questions(gt, seed):
            visible = delivered(views, objects[q["object"]], args.min_target_px)
            payload = {
                "images": [v["image"] for v in views],
                "question": "%s Each disk carries a short code. What code is written on the %s disk?"
                            % (describe(arm, views, gt["frames"]), q["color"]),
                "choices": q["choices"],
                "oracle": {"target_visible": visible, "answer": q["answer"]},
            }
            letter = backend.answer(payload)  # may be None (abstain -> scored wrong)
            index = ord(letter) - 65 if letter is not None else -1
            chosen = q["choices"][index] if 0 <= index < len(q["choices"]) else None
            rows.append({"object": q["object"], "correct": int(chosen == q["answer"]), "delivered": visible})
            if args.count_tokens:
                real.append(backend.count_tokens(payload))
        record["arms"][arm] = {
            "tokens": sum(backend.estimate_visual_tokens(v["image"].size) for v in views),
            "real_tokens": sum(real) / len(real) if real and all(real) else None,
            "crops": len(crops),
            "objects_covered": len(covered),  # objects whose code some crop shows whole
            "questions": rows,
        }
    return record


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize(records):
    arms = [a for a in ARMS if records and a in records[0]["arms"]]
    summary = {}
    for arm in arms:
        questions = [q for r in records for q in r["arms"][arm]["questions"]]
        acc = [q["correct"] for q in questions]
        lo, hi = bootstrap_ci(acc)
        row = {
            "accuracy": _mean(acc), "accuracy_ci": [lo, hi], "n_questions": len(acc),
            "mean_tokens": _mean([r["arms"][arm]["tokens"] for r in records]),
            "delivered_rate": _mean([int(q["delivered"]) for q in questions]),
            "accuracy_if_delivered": _mean([q["correct"] for q in questions if q["delivered"]]),
            "accuracy_if_missed": _mean([q["correct"] for q in questions if not q["delivered"]]),
        }
        if "frames-full" in arms:
            row["token_fraction"] = _mean([r["arms"][arm]["tokens"] / r["arms"]["frames-full"]["tokens"]
                                           for r in records])
        real = [r["arms"][arm]["real_tokens"] for r in records]
        if all(real):
            row["mean_real_tokens"] = _mean(real)
        crops = sum(r["arms"][arm]["crops"] for r in records)
        if crops:
            row["crops_per_video"] = crops / len(records)
            # Objects whose code some crop shows whole: per object, and per crop spent.
            row["crop_coverage"] = _mean([r["arms"][arm]["objects_covered"] / r["objects"] for r in records])
            row["crop_efficiency"] = sum(r["arms"][arm]["objects_covered"] for r in records) / crops
        summary[arm] = row
    behaviors = sorted({b for r in records for b in r["scanpath"]})
    summary["scanpath"] = {
        b: {m: _mean([r["scanpath"][b][m] for r in records if b in r["scanpath"]])
            for m in ("coverage", "mean_latency", "revisit_waste", "perseveration",
                      "labels_per_object", "off_object")}
        for b in behaviors}
    return summary


def format_table(summary):
    arms = [a for a in ARMS if a in summary]
    header = "%-15s %9s %16s %8s %10s %10s %9s" % (
        "arm", "accuracy", "95% CI", "tokens", "token-frac", "delivered", "crop-cov")
    lines = ["VLM video front-end (H7): accuracy vs visual tokens per question", header, "-" * len(header)]
    for arm in arms:
        s = summary[arm]
        lines.append("%-15s %9.3f  [%5.3f,%5.3f] %8.0f %10s %10.3f %9s" % (
            arm, s["accuracy"], s["accuracy_ci"][0], s["accuracy_ci"][1], s["mean_tokens"],
            "%.3f" % s["token_fraction"] if "token_fraction" in s else "-", s["delivered_rate"],
            "%.2f" % s["crop_coverage"] if "crop_coverage" in s else "-"))
    for behavior, m in summary["scanpath"].items():
        lines.append("scanpath %-12s coverage %.2f  latency %.1f  waste %.2f  perseveration %.2f  "
                     "labels/object %.1f  off-object %.2f" % (
                         behavior, m["coverage"], m["mean_latency"], m["revisit_waste"], m["perseveration"],
                         m["labels_per_object"], m["off_object"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--config", default="configs/attend.yaml", help="pipeline config for --attend")
    ap.add_argument("--out", default="results/vlm_video")
    ap.add_argument("--backend", default="ollama", help="ollama (default; local) | claude | mock")
    ap.add_argument("--model", default=None, help="backend model (defaults as in vlm_frontend.py)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma list of arms (default: %s)" % ",".join(DEFAULT_ARMS))
    ap.add_argument("--gt-identity", action="store_true",
                    help="add the space-ior-gtid / object-ior-gtid decomposition arms (perfect-tracker dedup)")
    ap.add_argument("--seeds", type=int, default=3, help="number of scenes")
    ap.add_argument("--seed0", type=int, default=0, help="first scene seed")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--objects", type=int, default=6, help="objects per scene (<= 8, distinct colours)")
    ap.add_argument("--late", type=int, default=2, help="objects arriving mid-video")
    ap.add_argument("--speed", type=float, default=24.0, help="native px per frame (M12's 6 at 320 px)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--radius", type=int, default=64, help="disk radius, native px (M12's 16 at 320 px)")
    ap.add_argument("--tag-size", type=int, default=20, help="font size of the codes (native px)")
    ap.add_argument("--occlude", action="store_true", help="occlude object 0 for a window")
    ap.add_argument("--k", type=int, default=None, help="crops per video (default: --objects)")
    ap.add_argument("--crop-side", type=int, default=192, help="crop size (native px)")
    ap.add_argument("--global-side", type=int, default=512, help="overview frame long side (px)")
    ap.add_argument("--frames-shown", type=int, default=4, help="T: frames in the frames-* arms")
    ap.add_argument("--full-max-side", type=int, default=1280, help="frames-full cap (px)")
    ap.add_argument("--proc-max-side", type=int, default=320, help="attention processing size (px)")
    ap.add_argument("--match-radius", type=float, default=28.0, help="scanpath scoring radius (proc px)")
    ap.add_argument("--min-target-px", type=int, default=10,
                    help="legibility floor: the code's shorter side on screen (px), for the mock / delivered")
    ap.add_argument("--no-tracking-aids", action="store_true",
                    help="plain object-file correspondence (default adds motion prediction + appearance matching)")
    ap.add_argument("--count-tokens", action="store_true", help="also record the backend's real token count")
    ap.add_argument("--resume", action="store_true", help="keep the seeds already in --out/results.json")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless the oracle reads every code and frames-uniform doesn't (plumbing)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    if args.objects > 8:
        sys.exit("--objects: at most 8 (one distinct colour each)")
    if args.gt_identity:
        args.arms += ",space-ior-gtid,object-ior-gtid"
    unknown = set(a.strip() for a in args.arms.split(",")) - set(ARMS)
    if unknown:
        sys.exit("unknown arm(s): %s (have: %s)" % (", ".join(sorted(unknown)), ", ".join(ARMS)))
    backend = create_backend(args.backend, **({"model": args.model} if args.model else {}))

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    records = []
    if args.resume and os.path.exists(results_path):
        records = dynamic_ior.load_json(results_path)
    done = {r["seed"] for r in records}
    for seed in range(args.seed0, args.seed0 + args.seeds):
        if seed in done:
            continue
        records.append(run_scene(backend, seed, args))
        with open(results_path, "w") as fh:  # after every scene, for --resume
            json.dump(records, fh, indent=2)
        print("  scene %d done" % seed, file=sys.stderr)

    summary = summarize(records)
    print(format_table(summary))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"config": vars(args), "backend": args.backend, "model": getattr(backend, "model", None),
                   "summary": summary}, fh, indent=2)
    if args.json:
        print(json.dumps(summary, indent=2))

    if args.check:
        # Plumbing, not the hypothesis: the oracle crops deliver every code and
        # the budget-matched frames don't.
        if "oracle" in summary and summary["oracle"]["accuracy"] != 1.0:
            sys.exit("check failed: oracle accuracy %.3f, expected 1.0" % summary["oracle"]["accuracy"])
        if "frames-uniform" in summary and "oracle" in summary and \
                not summary["frames-uniform"]["accuracy"] < summary["oracle"]["accuracy"]:
            sys.exit("check failed: frames-uniform read the codes as well as the oracle")

    print("summary: %s" % os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
