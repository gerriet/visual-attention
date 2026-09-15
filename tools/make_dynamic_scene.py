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
PALETTE_NAMES = ["red", "green", "blue", "yellow", "orange", "magenta", "cyan", "pink"]

# --tags (M19): each object carries a short code, legible only at native
# resolution. Easily confused glyphs (O/0, I/1, S/5, B/8, Z/2) are left out.
TAG_LETTERS = "ACDEFGHJKLMNPRTUVWXY"
TAG_DIGITS = "34679"


def object_color(i, args):
    """Distractors cycle the non-red palette; the search target (if any) is red."""
    if args.target is None:
        return PALETTE[i % len(PALETTE)]
    if i == args.target:
        return PALETTE[0]  # red — the colour-defined search target
    distractors = PALETTE[1:]
    return distractors[i % len(distractors)]


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
            "color": object_color(i, args),
            "radius": r,
            # One object may be occluded for a window of frames (tests recovery).
            "occluded_from": (args.frames // 3 if (args.occlude and i == 0) else -1),
            "occluded_to": (args.frames // 3 + args.occlude_len if (args.occlude and i == 0) else -1),
            "onset": 0,
            "positions": [],
        })

    # --late (M19): the last N objects arrive during the video, evenly spaced
    # over its first three quarters (invisible until then).
    late = min(args.late, args.objects)
    for j, obj in enumerate(objects[args.objects - late:]):
        obj["onset"] = int(round(0.75 * args.frames * (j + 1) / (late + 1)))
    if args.tags:
        # A separate stream, so a seed's trajectories are the same with or without tags.
        codes = [a + d for a in TAG_LETTERS for d in TAG_DIGITS]
        picks = np.random.RandomState(args.seed + 7919).choice(len(codes), size=args.objects, replace=False)
        for obj, k in zip(objects, picks):
            obj["tag"] = codes[k]

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
            visible = f >= obj["onset"] and not (obj["occluded_from"] <= f < obj["occluded_to"])
            obj["positions"].append({
                "frame": f, "x": int(round(p[0])), "y": int(round(p[1])),
                "w": 2 * rad, "h": 2 * rad, "visible": visible,
            })
    return objects


def tag_font(args):
    from PIL import ImageFont
    return ImageFont.load_default(size=args.tag_size)


def draw_tags(frame, objects, f, font):
    """Each visible object's code at its centre, in black or white ink by the
    disk's brightness."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(frame)
    for obj in objects:
        pos = obj["positions"][f]
        if pos["visible"]:
            r, g, b = obj["color"]
            ink = (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else (255, 255, 255)
            draw.text((pos["x"], pos["y"]), obj["tag"], font=font, fill=ink, anchor="mm")


def render(objects, args):
    os.makedirs(args.out, exist_ok=True)
    w, h = args.width, args.height
    yy, xx = np.mgrid[0:h, 0:w]
    font = tag_font(args) if args.tags else None
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
        frame = Image.fromarray(img)
        if args.tags:
            draw_tags(frame, objects, f, font)
        frame.save(os.path.join(args.out, "frame_%04d.png" % f))


def write_ground_truth(objects, args):
    font = tag_font(args) if args.tags else None

    def record(o):
        # Additive fields only when their flag is set, so M12 scenes are unchanged.
        r = {"id": o["id"], "positions": o["positions"]}
        if args.late:
            r["onset"] = o["onset"]
        if args.tags:
            left, top, right, bottom = font.getbbox(o["tag"], anchor="mm")
            r.update({"tag": o["tag"], "color": PALETTE_NAMES[PALETTE.index(o["color"])],
                      "tag_box": [right - left, bottom - top]})  # code extent (px), centred
        return r

    gt = {
        "schema": "dynamic-scene-gt/v1",
        "width": args.width, "height": args.height, "frames": args.frames,
        "seed": args.seed, "speed": args.speed,
        "objects": [record(o) for o in objects],
    }
    if args.target is not None:
        gt["target"] = args.target  # additive: the search target's object id (M17)
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
    ap.add_argument("--target", type=int, default=None,
                    help="mark this object id as the (red) search target — M17 priority-map study")
    ap.add_argument("--tags", action="store_true",
                    help="M19: draw a short code on each object, legible only at native resolution")
    ap.add_argument("--tag-size", type=int, default=20, help="font size of the --tags codes (px)")
    ap.add_argument("--late", type=int, default=0, help="M19: the last N objects arrive during the video")
    args = ap.parse_args()

    if args.tags and args.objects > len(PALETTE):
        ap.error("--tags needs distinct colours: at most %d objects" % len(PALETTE))

    if args.target is not None and not (0 <= args.target < args.objects):
        ap.error("--target %d is out of range for --objects %d (ids 0..%d)"
                 % (args.target, args.objects, args.objects - 1))

    objects = build_scene(args)
    render(objects, args)
    write_ground_truth(objects, args)
    print("wrote %d frames + gt.json to %s (%d objects, speed %.1f)"
          % (args.frames, args.out, args.objects, args.speed))


if __name__ == "__main__":
    main()
