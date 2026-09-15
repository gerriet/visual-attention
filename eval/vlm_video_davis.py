#!/usr/bin/env python3
"""Object files as a video token cache on real video (roadmap M19, H7): DAVIS-2017.

The synthetic study (eval/vlm_video.py) carries the hypothesis; this runs the
same arms on the 30 DAVIS-2017 val sequences (61 annotated objects, 480p,
real motion, occlusion and clutter) to test the identity and segmentation
machinery and object coverage on real footage. At 480p the "detail only
legible at native resolution" story is weak for the big objects (people,
animals, cars); the small ones (phones, rope, kite, box, gun) are where
native-resolution crops should matter.

Arms, all on the same VLM; the crop arms and frames-uniform share one budget:

  frames-full     T evenly spaced native frames — the naive reference
  frames-uniform  the same T frames downsampled to the crop budget
  space-ior       overview frame + K crops at spatial-IOR fixations,
                  deduplicated by *location*
  object-ior      overview frame + K crops, one per *object file*
  oracle          overview frame + one crop per annotated object, around its
                  mask in the frame where it is largest
  (--gt-identity) space-ior-gtid / object-ior-gtid: the behaviors' fixations
                  deduplicated by the *annotated* identity — a perfect tracker

A crop covers the attended object: the object file's box (the mask box for the
oracle) grown by a margin, at native resolution, and never larger than
--crop-side — a larger region is downscaled to it, so every crop costs the
same. K defaults to the number of annotated objects + 1.

Two measures:
  delivered  model-free — an object counts when some view shows >= 60% of its
             mask with the mask's shorter side >= --min-target-px on screen
  accuracy   per category present in the video, "Which of these can be seen in
             the video?" — that category + 3 absent ones (from the categories
             of all 30 sequences; chance 0.25). The mock answers correctly iff
             an object of that category was delivered.

  eval/vlm_video_davis.py --backend mock --gt-identity            # model-free
  eval/vlm_video_davis.py --count-tokens --config configs/attend_identity.yaml
"""

import argparse
import json
import math
import os
import random
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dynamic_ior  # noqa: E402
from study_common import bootstrap_ci  # noqa: E402
from vlm_backends import VISUAL_TOKEN_PATCH, create_backend  # noqa: E402
from vlm_frontend import make_view  # noqa: E402
from vlm_video import scaled_view  # noqa: E402

ARMS = ("frames-full", "frames-uniform", "space-ior", "space-ior-gtid", "object-ior", "object-ior-gtid", "oracle")
DEFAULT_ARMS = ("frames-full", "frames-uniform", "space-ior", "object-ior", "oracle")
BEHAVIORS = {"space-ior": "spatial-ior", "object-ior": "object-ior"}
MASK_OVERLAP = 0.6  # share of an object's mask a view must show
SMALL_OBJECT_PX = 64  # median mask short side (native px) below which an object counts as small


def _pil():
    from PIL import Image
    return Image


class Sequence:
    """One DAVIS sequence: native frames and per-frame object masks, on demand."""

    def __init__(self, name, root):
        from datasets import davis2017
        self.name = name
        self.items = list(davis2017.iter_frames(name, root))
        self.categories = davis2017.OBJECT_CATEGORIES[name]
        self._load_masks = davis2017.load_masks
        self._frames, self._masks = {}, {}

    @property
    def n_frames(self):
        return len(self.items)

    @property
    def size(self):
        return self.frame(0).size

    def objects(self):
        return sorted(self.categories)

    def frame(self, f):
        if f not in self._frames:
            with _pil().open(self.items[f][1]) as im:
                self._frames[f] = im.convert("RGB")
        return self._frames[f]

    def masks(self, f):
        if f not in self._masks:
            annotation = self.items[f][2]
            self._masks[f] = self._load_masks(annotation) if annotation else {}
        return self._masks[f]


def mask_box(mask):
    """(x0, y0, x1, y1) of a bool mask, exclusive upper bounds; None if empty."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def downscale(seq, proc_dir, proc_max_side):
    """Write the bounded-size copy the attention system runs on; returns the
    native -> proc scale."""
    Image = _pil()
    os.makedirs(proc_dir, exist_ok=True)
    w, h = seq.size
    scale = min(1.0, proc_max_side / max(w, h))
    for f in range(seq.n_frames):
        small = seq.frame(f).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        small.save(os.path.join(proc_dir, "frame_%04d.png" % f))
    return scale


def attend(binary, proc_dir, config, behavior, out_dir, scale, tracking):
    """One stage-2 behavior's scanpath, frame-ordered, in native coordinates,
    with each focus's object-file box."""
    scanpath = dynamic_ior.run_arm(binary, proc_dir, config, behavior, out_dir, tracking=tracking)
    out = []
    for e in sorted(scanpath, key=lambda e: e["frame"]):
        bx, by, bw, bh = e.get("bbox", (e["x"], e["y"], 1, 1))
        out.append({"frame": e["frame"], "label": e["label"], "x": e["x"] / scale, "y": e["y"] / scale,
                    "box": (bx / scale, by / scale, (bx + bw) / scale, (by + bh) / scale)})
    return out


def attributed_object(seq, frame, x, y, radius):
    """The annotated object at (x, y): the mask containing the point, else the
    nearest mask pixel within `radius`; None if none."""
    import numpy as np
    xi, yi = int(round(x)), int(round(y))
    best, best_d = None, radius
    for oid, mask in seq.masks(frame).items():
        h, w = mask.shape
        if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
            return oid
        y0, y1 = max(0, yi - int(radius)), min(h, yi + int(radius) + 1)
        x0, x1 = max(0, xi - int(radius)), min(w, xi + int(radius) + 1)
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if len(xs):
            d = float(np.min(np.hypot(xs + x0 - x, ys + y0 - y)))
            if d <= best_d:
                best, best_d = oid, d
    return best


def first_per_label(scanpath, k):
    """One pick per object file — the first fixation of each tracked identity."""
    seen, picks = set(), []
    for e in scanpath:
        if len(picks) >= k:
            break
        if e["label"] not in seen:
            seen.add(e["label"])
            picks.append(e)
    return picks


def new_locations(scanpath, k, radius):
    """One pick per new *location* — farther than `radius` from every earlier pick."""
    picks = []
    for e in scanpath:
        if len(picks) >= k:
            break
        if all(math.hypot(e["x"] - p["x"], e["y"] - p["y"]) >= radius for p in picks):
            picks.append(e)
    return picks


def first_per_object(seq, scanpath, k, radius):
    """A behavior's fixations deduplicated by *annotated* identity (a perfect
    tracker that also knows which regions are objects)."""
    seen, picks = set(), []
    for e in scanpath:
        if len(picks) >= k:
            break
        oid = attributed_object(seq, e["frame"], e["x"], e["y"], radius)
        if oid is not None and oid not in seen:
            seen.add(oid)
            picks.append(e)
    return picks


def oracle_picks(seq, k):
    """One pick per annotated object: its mask box in the frame where the mask
    is largest (its first frame is often an entry, the object still tiny)."""
    picks = []
    for oid in seq.objects():
        best = None
        for f in range(seq.n_frames):
            mask = seq.masks(f).get(oid)
            area = int(mask.sum()) if mask is not None else 0
            if area and (best is None or area > best[0]):
                best = (area, f, mask)
        if best:
            _, f, mask = best
            box = mask_box(mask)
            picks.append({"frame": f, "x": (box[0] + box[2]) / 2, "y": (box[1] + box[3]) / 2, "box": box})
    return sorted(picks, key=lambda p: p["frame"])[:k]


def identity_stats(seq, scanpath, radius):
    """Scanpath scores against the masks: coverage (annotated objects ever
    attended / all), distinct labels per attended object (1.0 = perfect), share
    of fixations on no annotated object."""
    labels, off = {}, 0
    for e in scanpath:
        oid = attributed_object(seq, e["frame"], e["x"], e["y"], radius)
        if oid is None:
            off += 1
        else:
            labels.setdefault(oid, set()).add(e["label"])
    return {"coverage": len(labels) / len(seq.objects()),
            "labels_per_object": _mean([len(s) for s in labels.values()]) or 0.0,
            "off_object": off / len(scanpath) if scanpath else 0.0}


def box_crop_view(image, frame, box, side, margin=0.25):
    """A crop covering `box` plus a margin — at least `side` wide, downscaled to
    `side` when larger, so every crop costs the same."""
    Image = _pil()
    w, h = image.size
    x0, y0, x1, y1 = box
    s = min(max(side, (1 + 2 * margin) * max(x1 - x0, y1 - y0)), max(w, h))
    bx0 = int(max(0, min(w - s, (x0 + x1 - s) / 2)))
    by0 = int(max(0, min(h - s, (y0 + y1 - s) / 2)))
    source = (bx0, by0, int(min(w, bx0 + s)), int(min(h, by0 + s)))
    crop, scale = image.crop(source), 1.0
    if max(crop.size) > side:
        scale = side / max(crop.size)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    return dict(make_view(crop, source, scale), frame=frame)


def view_shows(seq, view, oid, min_px):
    """Does this view show >= MASK_OVERLAP of the object's mask, its visible
    part at least min_px on its shorter side on screen?"""
    import numpy as np
    mask = seq.masks(view["frame"]).get(oid)
    if mask is None:
        return False
    total = int(mask.sum())
    if total == 0:
        return False
    x0, y0, x1, y1 = view["source_box"]
    inside = mask[y0:y1, x0:x1]
    if inside.sum() < MASK_OVERLAP * total:
        return False
    ys, xs = np.nonzero(inside)
    short = min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    return short * view["scale"] >= min_px


def small_objects(seq):
    """Objects whose median mask short side (native px) is below SMALL_OBJECT_PX."""
    import statistics
    sides = {}
    for f in range(0, seq.n_frames, 5):
        for oid, mask in seq.masks(f).items():
            box = mask_box(mask)
            if box:
                sides.setdefault(oid, []).append(min(box[2] - box[0], box[3] - box[1]))
    return {oid for oid, s in sides.items() if statistics.median(s) < SMALL_OBJECT_PX}


def build_arms(seq, picks, arms_wanted, backend, params):
    """{arm: views}. The crop arms share one overview frame; frames-uniform is
    budget-matched to the nominal crop arm (overview + K crops)."""
    n = seq.n_frames
    w, h = seq.size
    mid = n // 2
    overview = scaled_view(seq.frame(mid), mid, params["global_side"] / max(w, h))
    arms = {arm: [overview] + [box_crop_view(seq.frame(p["frame"]), p["frame"], p["box"], params["crop_side"])
                               for p in arm_picks]
            for arm, arm_picks in picks.items()}
    t = params["frames_shown"]
    shown = sorted({round(i * (n - 1) / (t - 1)) for i in range(t)}) if t > 1 else [mid]
    if "frames-full" in arms_wanted:
        arms["frames-full"] = [scaled_view(seq.frame(f), f, params["full_max_side"] / max(w, h)) for f in shown]
    if "frames-uniform" in arms_wanted:
        budget = (backend.estimate_visual_tokens(overview["image"].size)
                  + params["k"] * backend.estimate_visual_tokens((params["crop_side"], params["crop_side"])))
        r = VISUAL_TOKEN_PATCH * math.sqrt(budget / len(shown)) / math.sqrt(w * h)
        arms["frames-uniform"] = [scaled_view(seq.frame(f), f, r) for f in shown]
    return {arm: arms[arm] for arm in ARMS if arm in arms}


def describe(arm, views, n_frames):
    if arm.startswith("frames"):
        return "The images are frames %s of a %d-frame video." % (
            ", ".join(str(v["frame"]) for v in views), n_frames)
    text = "Image 1 is an overview (frame %d) of a %d-frame video." % (views[0]["frame"], n_frames)
    if len(views) > 1:
        text += " The other images are close-up crops from frames %s." % ", ".join(str(v["frame"]) for v in views[1:])
    return text


def make_questions(categories, vocabulary, seed):
    """Per category present: that category among three absent ones."""
    rng = random.Random(seed)
    present = sorted(set(categories.values()))
    absent = sorted(set(vocabulary) - set(present))
    questions = []
    for category in present:
        choices = rng.sample(absent, 3) + [category]
        rng.shuffle(choices)
        questions.append({"category": category, "answer": category, "choices": choices,
                          "objects": sorted(oid for oid, c in categories.items() if c == category)})
    return questions


def vocabulary():
    from datasets import davis2017
    return sorted({c for cats in davis2017.OBJECT_CATEGORIES.values() for c in cats.values()})


def run_sequence(backend, name, args):
    seq = Sequence(name, args.root)
    out_dir = os.path.join(args.out, name)
    scale = downscale(seq, os.path.join(out_dir, "proc"), args.proc_max_side)
    tracking = () if args.no_tracking_aids else ("--motion-prediction", "--appearance-matching")
    k = args.k or len(seq.objects()) + 1
    wanted = [a.strip() for a in args.arms.split(",")]
    radius = args.match_radius

    picks, scan = {}, {}
    for arm, behavior in BEHAVIORS.items():
        if arm not in wanted and arm + "-gtid" not in wanted:
            continue
        path = attend(args.binary, os.path.join(out_dir, "proc"), args.config, behavior,
                      os.path.join(out_dir, "attend"), scale, tracking)
        scan[behavior] = identity_stats(seq, path, radius)
        if arm in wanted:
            picks[arm] = first_per_label(path, k) if arm == "object-ior" else new_locations(path, k, args.crop_side / 2)
        if arm + "-gtid" in wanted:
            picks[arm + "-gtid"] = first_per_object(seq, path, k, radius)
    if "oracle" in wanted:
        picks["oracle"] = oracle_picks(seq, k)
    params = {"k": k, "crop_side": args.crop_side, "global_side": args.global_side,
              "frames_shown": args.frames_shown, "full_max_side": args.full_max_side}
    arms = build_arms(seq, picks, wanted, backend, params)

    small = small_objects(seq)
    questions = make_questions(seq.categories, vocabulary(), zlib.crc32(name.encode()))
    record = {"sequence": name, "objects": len(seq.objects()), "small_objects": sorted(small), "k": k,
              "scanpath": scan, "arms": {}}
    for arm, views in arms.items():
        crops = views[1:] if arm in picks else []
        delivered = {oid: any(view_shows(seq, v, oid, args.min_target_px) for v in views) for oid in seq.objects()}
        covered = {oid for oid in seq.objects() if any(view_shows(seq, v, oid, 0) for v in crops)}
        rows, real = [], []
        for q in questions:
            visible = any(delivered[oid] for oid in q["objects"])
            payload = {
                "images": [v["image"] for v in views],
                "question": "%s Which of these can be seen in the video?" % describe(arm, views, seq.n_frames),
                "choices": q["choices"],
                "oracle": {"target_visible": visible, "answer": q["answer"]},
            }
            letter = backend.answer(payload)  # may be None (abstain -> scored wrong)
            index = ord(letter) - 65 if letter is not None else -1
            chosen = q["choices"][index] if 0 <= index < len(q["choices"]) else None
            rows.append({"category": q["category"], "correct": int(chosen == q["answer"]), "delivered": visible})
            if args.count_tokens:
                real.append(backend.count_tokens(payload))
        record["arms"][arm] = {
            "tokens": sum(backend.estimate_visual_tokens(v["image"].size) for v in views),
            "real_tokens": sum(real) / len(real) if real and all(real) else None,
            "crops": len(crops),
            "objects_covered": len(covered),
            "delivered": {str(oid): d for oid, d in delivered.items()},
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
        objects = [(oid in r["small_objects"], d) for r in records
                   for oid, d in ((int(o), d) for o, d in r["arms"][arm]["delivered"].items())]
        row = {
            "accuracy": _mean(acc), "accuracy_ci": [lo, hi], "n_questions": len(acc),
            "delivered_rate": _mean([int(d) for _, d in objects]),
            "delivered_small": _mean([int(d) for s, d in objects if s]),
            "n_small": sum(1 for s, _ in objects if s),
            "mean_tokens": _mean([r["arms"][arm]["tokens"] for r in records]),
        }
        if "frames-full" in arms:
            row["token_fraction"] = _mean([r["arms"][arm]["tokens"] / r["arms"]["frames-full"]["tokens"]
                                           for r in records])
        real = [r["arms"][arm]["real_tokens"] for r in records]
        if all(real):
            row["mean_real_tokens"] = _mean(real)
        crops = sum(r["arms"][arm]["crops"] for r in records)
        if crops:
            row["crop_coverage"] = _mean([r["arms"][arm]["objects_covered"] / r["objects"] for r in records])
            row["crop_efficiency"] = sum(r["arms"][arm]["objects_covered"] for r in records) / crops
        summary[arm] = row
    behaviors = sorted({b for r in records for b in r["scanpath"]})
    summary["scanpath"] = {b: {m: _mean([r["scanpath"][b][m] for r in records if b in r["scanpath"]])
                               for m in ("coverage", "labels_per_object", "off_object")} for b in behaviors}
    return summary


def format_table(summary):
    arms = [a for a in ARMS if a in summary]
    header = "%-15s %9s %16s %8s %10s %10s %10s %9s" % (
        "arm", "accuracy", "95% CI", "tokens", "token-frac", "delivered", "small", "crop-cov")
    lines = ["VLM video front-end on DAVIS-2017 (H7)", header, "-" * len(header)]
    for arm in arms:
        s = summary[arm]
        lines.append("%-15s %9.3f  [%5.3f,%5.3f] %8.0f %10s %10.3f %10s %9s" % (
            arm, s["accuracy"], s["accuracy_ci"][0], s["accuracy_ci"][1], s["mean_tokens"],
            "%.3f" % s["token_fraction"] if "token_fraction" in s else "-", s["delivered_rate"],
            "%.3f" % s["delivered_small"] if s["delivered_small"] is not None else "-",
            "%.2f" % s["crop_coverage"] if "crop_coverage" in s else "-"))
    for behavior, m in summary["scanpath"].items():
        lines.append("scanpath %-12s coverage %.2f  labels/object %.1f  off-object %.2f" % (
            behavior, m["coverage"], m["labels_per_object"], m["off_object"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(HERE), "data", "DAVIS"),
                    help="DAVIS-2017 root (480p layout)")
    ap.add_argument("--sequences", default=None, help="comma list (default: every val sequence)")
    ap.add_argument("--limit", type=int, default=0, help="max sequences (0 = all)")
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--config", default="configs/attend.yaml", help="pipeline config for --attend")
    ap.add_argument("--out", default="results/vlm_video_davis")
    ap.add_argument("--backend", default="ollama", help="ollama (default; local) | claude | mock")
    ap.add_argument("--model", default=None, help="backend model (defaults as in vlm_frontend.py)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma list of arms (default: %s)" % ",".join(DEFAULT_ARMS))
    ap.add_argument("--gt-identity", action="store_true",
                    help="add the space-ior-gtid / object-ior-gtid decomposition arms (perfect-tracker dedup)")
    ap.add_argument("--k", type=int, default=None, help="crops per video (default: annotated objects + 1)")
    ap.add_argument("--crop-side", type=int, default=224, help="crop size cap (px)")
    ap.add_argument("--global-side", type=int, default=427, help="overview frame long side (px)")
    ap.add_argument("--frames-shown", type=int, default=4, help="T: frames in the frames-* arms")
    ap.add_argument("--full-max-side", type=int, default=854, help="frames-full cap (px)")
    ap.add_argument("--proc-max-side", type=int, default=480, help="attention processing size (px)")
    ap.add_argument("--match-radius", type=float, default=20.0,
                    help="attribute a fixation to a mask within this distance (native px)")
    ap.add_argument("--min-target-px", type=int, default=24,
                    help="delivered: the visible mask's shorter side on screen (px)")
    ap.add_argument("--no-tracking-aids", action="store_true",
                    help="plain object-file correspondence (default adds motion prediction + appearance matching)")
    ap.add_argument("--count-tokens", action="store_true", help="also record the backend's real token count")
    ap.add_argument("--resume", action="store_true", help="keep the sequences already in --out/results.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    from datasets import davis2017  # numpy + PIL: only past --help

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    if not davis2017.available(args.root):
        sys.exit("DAVIS-2017 not found under %s — see eval/datasets/davis2017.py" % args.root)
    if args.gt_identity:
        args.arms += ",space-ior-gtid,object-ior-gtid"
    unknown = set(a.strip() for a in args.arms.split(",")) - set(ARMS)
    if unknown:
        sys.exit("unknown arm(s): %s (have: %s)" % (", ".join(sorted(unknown)), ", ".join(ARMS)))
    names = args.sequences.split(",") if args.sequences else davis2017.sequences(args.root, "val")
    if args.limit:
        names = names[:args.limit]
    backend = create_backend(args.backend, **({"model": args.model} if args.model else {}))

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    records = []
    if args.resume and os.path.exists(results_path):
        records = dynamic_ior.load_json(results_path)
    done = {r["sequence"] for r in records}
    for name in names:
        if name in done:
            continue
        records.append(run_sequence(backend, name, args))
        with open(results_path, "w") as fh:  # after every sequence, for --resume
            json.dump(records, fh, indent=2)
        print("  %s done" % name, file=sys.stderr)

    summary = summarize(records)
    print(format_table(summary))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"config": vars(args), "backend": args.backend, "model": getattr(backend, "model", None),
                   "summary": summary}, fh, indent=2)
    if args.json:
        print(json.dumps(summary, indent=2))
    print("summary: %s" % os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
