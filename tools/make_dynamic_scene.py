#!/usr/bin/env python3
"""Generate a synthetic dynamic scene for the dynamic-IOR study (roadmap M12).

Bright coloured disks move on a dark background and bounce off the walls; a disk
may be occluded for a stretch of frames. The output is a directory of frames
(frame_0000.png ...) that the CLI consumes with `--attend`, plus a ground-truth
JSON (dynamic-scene-gt/v1) giving each object's per-frame position, box and
visibility. The scorer (eval/dynamic_ior.py) reads that ground truth to decide,
per frame, which object attention actually landed on.

Deterministic given --seed. numpy + Pillow only (the eval venv has both).
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

# Distinct, saturated colours (RGB) so the colour feature has something to do.
PALETTE = [
    (230, 60, 60), (60, 200, 60), (70, 110, 240), (235, 200, 40),
    (220, 110, 30), (200, 70, 210), (40, 210, 210), (240, 130, 170),
]


def build_scene(args):
    rng = np.random.RandomState(args.seed)
    w, h = args.width, args.height
    r = args.radius

    objects = []
    for i in range(args.objects):
        pos = np.array([rng.uniform(r, w - r), rng.uniform(r, h - r)], dtype=float)
        angle = rng.uniform(0, 2 * np.pi)
        vel = args.speed * np.array([np.cos(angle), np.sin(angle)])
        objects.append({
            "id": i,
            "pos": pos,
            "vel": vel,
            "color": PALETTE[i % len(PALETTE)],
            "radius": r,
            # One object may be occluded for a window of frames (tests recovery).
            "occluded_from": (args.frames // 3 if (args.occlude and i == 0) else -1),
            "occluded_to": (args.frames // 3 + args.occlude_len if (args.occlude and i == 0) else -1),
            "positions": [],
        })

    for f in range(args.frames):
        for obj in objects:
            p, v, rad = obj["pos"], obj["vel"], obj["radius"]
            p += v
            for axis, limit in ((0, w), (1, h)):  # bounce off the walls
                if p[axis] < rad:
                    p[axis] = rad
                    v[axis] = abs(v[axis])
                elif p[axis] > limit - rad:
                    p[axis] = limit - rad
                    v[axis] = -abs(v[axis])
            visible = not (obj["occluded_from"] <= f < obj["occluded_to"])
            obj["positions"].append({
                "frame": f, "x": int(round(p[0])), "y": int(round(p[1])),
                "w": 2 * rad, "h": 2 * rad, "visible": visible,
            })
    return objects


def render(objects, args):
    os.makedirs(args.out, exist_ok=True)
    w, h = args.width, args.height
    yy, xx = np.mgrid[0:h, 0:w]
    for f in range(args.frames):
        img = np.full((h, w, 3), 25, dtype=np.uint8)  # dark background
        for obj in objects:
            pos = obj["positions"][f]
            if not pos["visible"]:
                continue
            rad = obj["radius"]
            mask = (xx - pos["x"]) ** 2 + (yy - pos["y"]) ** 2 <= rad * rad
            for c in range(3):
                img[..., c][mask] = obj["color"][c]
        Image.fromarray(img).save(os.path.join(args.out, "frame_%04d.png" % f))


def write_ground_truth(objects, args):
    gt = {
        "schema": "dynamic-scene-gt/v1",
        "width": args.width, "height": args.height, "frames": args.frames,
        "seed": args.seed, "speed": args.speed,
        "objects": [{"id": o["id"], "positions": o["positions"]} for o in objects],
    }
    with open(os.path.join(args.out, "gt.json"), "w") as fh:
        json.dump(gt, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output frame directory")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--objects", type=int, default=4)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--radius", type=int, default=16)
    ap.add_argument("--speed", type=float, default=6.0, help="pixels per frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occlude", action="store_true", help="occlude object 0 for a window")
    ap.add_argument("--occlude-len", type=int, default=6)
    args = ap.parse_args()

    objects = build_scene(args)
    render(objects, args)
    write_ground_truth(objects, args)
    print("wrote %d frames + gt.json to %s (%d objects, speed %.1f)"
          % (args.frames, args.out, args.objects, args.speed))


if __name__ == "__main__":
    main()
