#!/usr/bin/env python3
"""Do fixations land on faces? (H4 follow-up)

H4's verdict offers an explanation for why the thesis model's scanpaths beat
random but not a centre baseline on MIT1003: *"MIT1003 is full of faces and
text"* and the model has no channel for either. `configs/h4_face.yaml` adds the
channel; on the scanpath-similarity metrics it changes nothing.

That is two different claims, and the scanpath metrics cannot separate them:

  (a) the explanation is wrong — human gaze on these images is not mainly
      about faces, or not in a way the model could exploit; or
  (b) the explanation is right and the *metrics* are insensitive — which is
      not a wild suspicion here, since the same study already showed ScanMatch
      cannot tell a constant-centre path from the inter-observer ceiling
      (docs/SCANPATH_VS_HUMAN.md).

This script measures the thing directly instead: on the images where a face
detector fires, what fraction of fixations land inside a face box —

    human  ·  the thesis model  ·  the thesis model + the face channel

If humans are far above the model and the face arm closes the gap, (b) is the
answer and the face channel works while the metrics cannot see it. If humans
are *not* above the model, (a) is the answer and the explanation H4 offered is
simply wrong — which is the more useful outcome, because it stops a plausible
story from being repeated.

    eval/face_coverage.py --limit 200

Needs the eval venv and data/MIT1003.
"""
import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import bootstrap_ci, paired_bootstrap  # noqa: E402


def import_mit1003():
    from datasets import mit1003
    return mit1003


def detect_faces(binary, image, work_dir):
    """Face boxes via the pipeline's own face channel, read back from the map.

    The feature paints a Gaussian per detection and normalizes the strongest to
    1, so thresholding at 0.5 recovers one connected blob per face that is at
    least half as strong as the best one. Using the channel rather than calling
    OpenCV separately keeps this measuring *what the model can see*, which is
    the question.
    """
    import numpy as np
    from PIL import Image
    os.makedirs(work_dir, exist_ok=True)
    cfg = os.path.join(work_dir, "face_only.yaml")
    with open(cfg, "w") as fh:
        fh.write("pipeline:\n  fusion: weighted-sum\n  selection: ior\n"
                 "features:\n  face:\n    weight: 1.0\n"
                 "  color-munsell: {enabled: false}\n  color: {enabled: false}\n"
                 "  intensity: {enabled: false}\n  orientation: {enabled: false}\n"
                 "  eccentricity: {enabled: false}\n  symmetry: {enabled: false}\n"
                 "output:\n  display: false\n")
    out = os.path.join(work_dir, "feat")
    subprocess.run([binary, str(image), "--config", cfg, "--no-display", "--emit-features", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    path = os.path.join(out, "feature_face.png")
    if not os.path.exists(path):
        return []
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() <= 0:
        return []
    arr = arr / arr.max()
    mask = (arr >= 0.5).astype(np.uint8)
    try:
        import cv2
        count, labels = cv2.connectedComponents(mask)
        boxes = []
        for label in range(1, count):
            ys, xs = np.where(labels == label)
            if len(xs) >= 16:
                boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
        return boxes
    except ImportError:
        ys, xs = np.where(mask > 0)
        return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))] if len(xs) else []


def inside(point, boxes, pad):
    x, y = point
    for (x0, y0, x1, y1) in boxes:
        if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
            return True
    return False


def model_fixations(binary, image, config, work_dir, tag, cap):
    os.makedirs(work_dir, exist_ok=True)
    out = os.path.join(work_dir, tag + ".json")
    cmd = [binary, str(image), "--no-display", "--emit-json", out]
    if config:
        cmd += ["--config", config]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as fh:
        data = json.load(fh)
    fixations = data.get("fixations") or data.get("peaks") or []
    return [(f["x"], f["y"]) for f in fixations[:cap]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/face_coverage")
    ap.add_argument("--limit", type=int, default=200, help="stimuli to consider (0 = all)")
    ap.add_argument("--sample-seed", type=int, default=5)
    ap.add_argument("--cap", type=int, default=10, help="fixations per model scanpath")
    ap.add_argument("--pad", type=int, default=12, help="px of slack around a face box")
    ap.add_argument("--thesis-config", default="configs/thesis/thesis.yaml")
    ap.add_argument("--face-config", default="configs/h4_face.yaml")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s" % args.binary)
    mit = import_mit1003()
    if not mit.available():
        sys.exit("MIT1003 not found under data/MIT1003 — see eval/datasets/mit1003.py")

    # (stimulus, fixation_map, fixation_points) triples
    stimuli = list(mit.iter_stimuli())
    random.Random(args.sample_seed).shuffle(stimuli)
    if args.limit:
        stimuli = stimuli[: args.limit]

    human, thesis, face = [], [], []
    with_faces = 0
    for n, (stimulus, _fix_map, fix_pts) in enumerate(stimuli):
        work = os.path.join(args.out, "runs", stimulus.stem)
        boxes = detect_faces(args.binary, stimulus, work)
        if not boxes:
            continue
        gaze = mit.load_fixation_points(fix_pts)
        if not gaze:
            continue
        with_faces += 1

        # One value per *image*, not per fixation: the image is the independent
        # unit, and pooling fixations would make a crowded image count more.
        human.append(sum(1 for p in gaze if inside(p, boxes, args.pad)) / len(gaze))
        for config, bucket, tag in ((args.thesis_config, thesis, "thesis"),
                                    (args.face_config, face, "face")):
            points = model_fixations(args.binary, stimulus, config, work, tag, args.cap)
            bucket.append(sum(1 for p in points if inside(p, boxes, args.pad)) / len(points)
                          if points else 0.0)
        if (n + 1) % 25 == 0:
            print("  %d/%d stimuli (%d with faces)" % (n + 1, len(stimuli), with_faces), file=sys.stderr)

    if with_faces == 0:
        sys.exit("no stimulus in the sample had a detected face")

    print("\nFixations landing on a detected face (MIT1003, %d images with faces)" % with_faces)
    print("%-28s %8s %18s" % ("", "mean", "95% CI"))
    rows = []
    for name, values in (("human", human), ("thesis model", thesis), ("thesis + face channel", face)):
        lo, hi = bootstrap_ci(values)
        print("%-28s %8.3f [%6.3f, %6.3f]" % (name, sum(values) / len(values), lo, hi))
        rows.append({"name": name, "mean": sum(values) / len(values), "ci": [lo, hi], "n": len(values)})

    print("\npaired differences over images:")
    for label, a, b in (("human - thesis", human, thesis),
                        ("human - (thesis+face)", human, face),
                        ("(thesis+face) - thesis", face, thesis)):
        mean, lo, hi = paired_bootstrap(a, b)
        flag = "" if lo <= 0 <= hi else "   (excludes zero)"
        print("  %-24s %+7.3f [%+7.3f, %+7.3f]%s" % (label, mean, lo, hi, flag))
        rows.append({"name": label, "paired_delta": mean, "delta_ci": [lo, hi]})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"config": vars(args), "images_with_faces": with_faces, "rows": rows}, fh, indent=2)
    print("\nsummary: %s" % os.path.join(args.out, "summary.json"))
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
