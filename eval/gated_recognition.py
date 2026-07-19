#!/usr/bin/env python3
"""Gated-recognition study (roadmap M13): does recognition restricted to
attended ROIs reach near-full-frame accuracy at a fraction of the compute? [H2]

Both arms run the same detector (hog-person) over the same stream through the
same binary; only the gating differs:

  full-frame  --process-cadence full-frame   whole frame, every frame (baseline)
  gated       --process-cadence dwell|frame  only the attended ROI

vtest mode (self-relative, no external ground truth): the full-frame arm's
detections are the reference; the gated arm is scored on how many of them it
recovers — strict (same frame, IoU-matched) and within a ±window of frames
(attention doesn't look everywhere at once; the fair claim is recovery within
a small latency) — at what fraction of the pixels / wall-clock.

DAVIS mode (real ground truth): person boxes from the DAVIS-2017 masks
(eval/datasets/davis2017.py, hand-curated person ids); *both* arms are scored
on person-frame recall, so the baseline's own accuracy is measured too.

One command each:
  eval/gated_recognition.py --video data/samples/video/vtest.avi --frames 300
  eval/gated_recognition.py --davis            # all curated person sequences
"""

import argparse
import json
import os
import subprocess
import sys

GATED_CADENCES = ["dwell", "frame"]


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def run_attend(binary, source, out_json, cadence, config, frames, processors="hog-person"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    cmd = [binary, "--attend", source, "--processors", processors,
           "--process-cadence", cadence, "--no-save-frames",
           "--emit-scanpath", out_json]
    if config:
        cmd += ["--config", config]
    if frames:
        cmd += ["--frames", str(frames)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return load_json(out_json)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detections_by_frame(result, label, min_conf):
    """{frame: [box, ...]} of the run's detections with this label."""
    frames = {}
    for annotation in result.get("annotations", []):
        for det in annotation.get("detections", []):
            if det["label"] == label and det["confidence"] >= min_conf:
                frames.setdefault(annotation["frame"], []).append(tuple(det["box"]))
    return frames


def compute_cost(result):
    """Total processor compute of a run: (pixels, ms)."""
    processors = result.get("processing", {}).get("processors", [])
    return (sum(p["pixels"] for p in processors), sum(p["ms"] for p in processors))


def matched(box, candidates, iou_thr):
    return any(iou(box, c) >= iou_thr for c in candidates)


def score_recovery(reference, gated, iou_thr, window):
    """Fraction of reference detections the gated run recovers (strict / windowed)."""
    total = strict = windowed = 0
    for frame, boxes in reference.items():
        for box in boxes:
            total += 1
            if matched(box, gated.get(frame, []), iou_thr):
                strict += 1
            near = [c for f in range(frame - window, frame + window + 1)
                    for c in gated.get(f, [])]
            if matched(box, near, iou_thr):
                windowed += 1
    if total == 0:
        return {"reference": 0, "strict": 0.0, "windowed": 0.0}
    return {"reference": total, "strict": strict / total, "windowed": windowed / total}


def score_against_gt(gt_boxes, detections, iou_thr, window):
    """Person-frame recall vs ground truth: gt_boxes = {frame: {person: box}}."""
    total = strict = windowed = 0
    for frame, persons in gt_boxes.items():
        for person, box in persons.items():
            total += 1
            if matched(box, detections.get(frame, []), iou_thr):
                strict += 1
            for f in range(frame - window, frame + window + 1):
                other = gt_boxes.get(f, {}).get(person)
                if other is not None and matched(other, detections.get(f, []), iou_thr):
                    windowed += 1
                    break
    if total == 0:
        return {"person_frames": 0, "strict": 0.0, "windowed": 0.0}
    return {"person_frames": total, "strict": strict / total, "windowed": windowed / total}


def arm_summary(result, label, min_conf):
    pixels, ms = compute_cost(result)
    return {"pixels": pixels, "ms": ms,
            "detections": detections_by_frame(result, label, min_conf)}


def relative(value, reference):
    return value / reference if reference else 0.0


def study_video(binary, video, config, out_dir, frames, iou_thr, window, min_conf, cadences):
    """The self-relative vtest arm. Returns {cadence: metrics} + baseline cost."""
    full = run_attend(binary, video, os.path.join(out_dir, "full.json"), "full-frame", config, frames)
    reference = arm_summary(full, "person", min_conf)

    rows = {}
    for cadence in cadences:
        gated = run_attend(binary, video, os.path.join(out_dir, f"gated_{cadence}.json"),
                           cadence, config, frames)
        summary = arm_summary(gated, "person", min_conf)
        row = score_recovery(reference["detections"], summary["detections"], iou_thr, window)
        row["pixels_fraction"] = relative(summary["pixels"], reference["pixels"])
        row["ms_fraction"] = relative(summary["ms"], reference["ms"])
        rows[cadence] = row
    return {"reference_detections": sum(len(b) for b in reference["detections"].values()),
            "reference_pixels": reference["pixels"], "reference_ms": reference["ms"],
            "arms": rows}


def import_davis():
    """The DAVIS adapter needs numpy/PIL (eval venv); import it lazily so the
    vtest arm stays stdlib-only."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets import davis2017
    return davis2017


def davis_gt_boxes(sequence, person_ids):
    davis2017 = import_davis()

    gt = {}
    for index, _, annotation in davis2017.iter_frames(sequence):
        if annotation is None:
            continue
        present = davis2017.load_person_boxes(annotation, person_ids)
        if present:
            gt[index] = present
    return gt


def study_davis(binary, config, out_dir, iou_thr, window, min_conf, cadences, sequences=None):
    davis2017 = import_davis()

    chosen = sequences or sorted(davis2017.PERSON_OBJECTS)
    results = {}
    for sequence in chosen:
        person_ids = davis2017.PERSON_OBJECTS[sequence]
        frames_dir = str(davis2017.frames_dir(sequence))
        gt = davis_gt_boxes(sequence, person_ids)
        seq_out = os.path.join(out_dir, sequence)

        rows = {}
        full = run_attend(binary, frames_dir, os.path.join(seq_out, "full.json"),
                          "full-frame", config, None)
        reference = arm_summary(full, "person", min_conf)
        rows["full-frame"] = score_against_gt(gt, reference["detections"], iou_thr, window)
        rows["full-frame"].update({"pixels_fraction": 1.0, "ms_fraction": 1.0})

        for cadence in cadences:
            gated = run_attend(binary, frames_dir, os.path.join(seq_out, f"gated_{cadence}.json"),
                               cadence, config, None)
            summary = arm_summary(gated, "person", min_conf)
            row = score_against_gt(gt, summary["detections"], iou_thr, window)
            row["pixels_fraction"] = relative(summary["pixels"], reference["pixels"])
            row["ms_fraction"] = relative(summary["ms"], reference["ms"])
            rows[cadence] = row
        results[sequence] = rows
    return results


def format_video_table(study):
    header = "%-12s %10s %10s %12s %12s" % ("cadence", "strict", "windowed", "px-fraction", "ms-fraction")
    lines = ["vtest (self-relative: recovery of %s full-frame detections)" %
             study["reference_detections"], header, "-" * len(header)]
    for cadence, m in study["arms"].items():
        lines.append("%-12s %10.3f %10.3f %12.3f %12.3f" % (
            cadence, m["strict"], m["windowed"], m["pixels_fraction"], m["ms_fraction"]))
    return "\n".join(lines)


def format_davis_table(results):
    header = "%-14s %-12s %10s %10s %12s" % ("sequence", "arm", "strict", "windowed", "px-fraction")
    lines = ["DAVIS person sequences (person-frame recall vs mask ground truth)",
             header, "-" * len(header)]
    for sequence, rows in results.items():
        for arm, m in rows.items():
            lines.append("%-14s %-12s %10.3f %10.3f %12.3f" % (
                sequence, arm, m["strict"], m["windowed"], m["pixels_fraction"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--config", default="configs/attend.yaml")
    ap.add_argument("--out", default="results/gated_recognition")
    ap.add_argument("--video", default=None, help="video for the self-relative arm (e.g. vtest.avi)")
    ap.add_argument("--frames", type=int, default=300, help="frame cap for --video (0 = all)")
    ap.add_argument("--davis", action="store_true", help="run the DAVIS ground-truth arm")
    ap.add_argument("--sequence", action="append", help="restrict DAVIS to this sequence (repeatable)")
    ap.add_argument("--cadence", action="append", choices=GATED_CADENCES,
                    help="gated cadences to run (default: both)")
    ap.add_argument("--iou", type=float, default=0.3, help="IoU match threshold (HOG boxes are loose)")
    ap.add_argument("--window", type=int, default=15,
                    help="± frames for the windowed (recovery-within-latency) score")
    ap.add_argument("--min-conf", type=float, default=0.2,
                    help="detection confidence floor, applied to both arms")
    ap.add_argument("--json", action="store_true", help="also print raw metrics as JSON")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    if not args.video and not args.davis:
        sys.exit("nothing to do: pass --video <file> and/or --davis")

    cadences = args.cadence or GATED_CADENCES
    everything = {}
    if args.video:
        study = study_video(args.binary, args.video, args.config, args.out,
                            args.frames or None, args.iou, args.window, args.min_conf, cadences)
        everything["video"] = study
        print(format_video_table(study))
    if args.davis:
        results = study_davis(args.binary, args.config, args.out, args.iou,
                              args.window, args.min_conf, cadences, args.sequence)
        everything["davis"] = results
        print(format_davis_table(results))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(everything, fh, indent=2)
    print("summary: %s" % os.path.join(args.out, "summary.json"))
    if args.json:
        print(json.dumps(everything, indent=2))


if __name__ == "__main__":
    main()
