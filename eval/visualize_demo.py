#!/usr/bin/env python3
"""Compose an attention trace into a video: what the model saw, and what it knew.

Reads an `attention-trace/v1` directory (written by `attention --attend
--emit-trace`, see docs/INTERCHANGE_FORMAT.md) plus the camera images, and lays
out, per frame: the scene with the object files outlined and the focus marked,
a card per object file with an arrow to the thing it refers to, the feature maps
and the fused saliency, the neural field's activity, and a timeline showing
which object file held the focus when.

    eval/visualize_demo.py --trace results/demo_trace --frames results/demo_scene/left \
        --out results/demo.mp4

Two layouts: `--style full` is the instrument panel (everything above);
`--style clean` is the scene, the saliency and the timeline, for a README.

Design notes, because they are decisions and not taste:

* One colour per object-file identity, held for the file's whole life. When
  correspondence carries an object through the handover the colour does not
  change; when it breaks, the colour does. The picture cannot flatter the model.
* Maps are drawn on their trace's *fixed* scale, never per-frame normalized, so
  a change in brightness means the response changed. This is the same point as
  the dossier's finding A, which per-frame normalization hid for two months.
* Inactive object files grey out and stay on screen rather than vanishing: "the
  model still knows about it" is part of the state being shown.
* Every active object file is outlined, but only the carded ones get a colour
  and a label; the rest are thin grey boxes with a count. The field holds many
  more clusters than the scene holds objects -- it segments parts, not people --
  and both hiding them and drawing them all at full weight would misrepresent
  the model, in opposite directions. See docs/DEMO_STEREO_SCENE.md 6b.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- look ---------------------------------------------------------------------

BG = (18, 20, 24)
PANEL = (28, 31, 36)
INK = (232, 234, 238)
DIM = (140, 146, 156)
PROTO = (122, 130, 142)   # the field's other clusters: visible, deliberately quiet
FOCUS = (255, 255, 255)

# Twelve hues that stay distinguishable next to each other and on a grey scene.
IDENTITY_COLOURS = [
    (232, 93, 78), (86, 180, 233), (240, 182, 60), (120, 200, 120), (190, 130, 230),
    (72, 200, 190), (236, 140, 90), (150, 165, 235), (210, 110, 160), (170, 200, 80),
    (110, 150, 200), (225, 160, 200),
]

# saliency / features: dark -> warm -> white.  field: blue (inhibited) -> black -> orange.
RAMP_HEAT = [(8, 8, 20), (70, 20, 90), (170, 48, 80), (235, 120, 40), (255, 220, 140), (255, 255, 255)]
RAMP_FIELD = [(40, 90, 190), (14, 16, 22), (14, 16, 22), (240, 150, 40), (255, 240, 180)]


def ramp(stops, t):
    t = min(1.0, max(0.0, t)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a, b = stops[i], stops[i + 1]
    return tuple(int(a[c] + (b[c] - a[c]) * f) for c in range(3))


def colourise(path, stops, lo, hi, size):
    """A 16-bit map on its fixed scale -> an RGB thumbnail. The stored value is
    linear over [lo, hi], so its position along the ramp is just value/max."""
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    full = 65535.0 if arr.max() > 255.0 else 255.0
    index = np.clip(arr / full, 0.0, 1.0)
    lut = np.array([ramp(stops, v / 255.0) for v in range(256)], dtype=np.uint8)
    rgb = lut[(index * 255.0).astype(np.uint8)]
    return Image.fromarray(rgb, "RGB").resize(size, Image.BILINEAR)


def identity_colour(label):
    return IDENTITY_COLOURS[label % len(IDENTITY_COLOURS)]


def font(size):
    try:
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


# --- layout -------------------------------------------------------------------

CANVAS = (1920, 1080)
VIEW = (20, 20, 1120, 840)        # x, y, w, h of the scene panel
CARDS_X, CARDS_W = 1160, 740
STRIP_Y, STRIP_H = 872, 124
TIMELINE = (20, 1006, 1880, 62)
MAX_CARDS = 5


def draw_scene(canvas, draw, image, objects, focus, carded, fonts):
    """The camera image with every active object file outlined.

    Two weights, and the distinction is the point rather than decoration. The
    field typically holds far more clusters than the scene holds objects — on
    this take about seventeen for five staged things — because it segments at
    the scale of *parts*: legs, arms and torsos are clusters of the same size
    and nearly the same saliency as the ball (docs/DEMO_STEREO_SCENE.md 6b).
    Drawing them all at full weight makes a thicket and hides the claim; drawing
    only the carded ones would quietly overstate what the model does. So the
    carded files get the full outline and a leader line, and the rest are drawn
    thin: present, honest, and visibly not the thing being tracked.
    """
    x, y, w, h = VIEW
    scale = w / image.width
    canvas.paste(image.resize((w, h), Image.LANCZOS), (x, y))

    placed = []
    others = 0
    # Thin ones first, so a carded outline is never overdrawn by a proto-object.
    for obj in objects:
        if not obj["active"] or obj["label"] in carded:
            continue
        bx, by, bw, bh = [v * scale for v in obj["bbox"]]
        draw.rectangle((x + bx, y + by, x + bx + bw, y + by + bh), outline=PROTO, width=1)
        others += 1
    for obj in objects:
        if not obj["active"] or obj["label"] not in carded:
            continue
        bx, by, bw, bh = [v * scale for v in obj["bbox"]]
        box = (x + bx, y + by, x + bx + bw, y + by + bh)
        is_focus = focus is not None and focus["label"] == obj["label"]
        colour = identity_colour(obj["label"])
        draw.rectangle(box, outline=colour, width=4 if is_focus else 2)
        if is_focus:
            draw.rectangle([box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4], outline=FOCUS, width=2)
        draw.text((box[0] + 3, box[1] - 19), "#%d" % obj["label"], font=fonts["small"], fill=colour)
        placed.append((obj["label"], (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    if others:
        note = "+%d more clusters the field holds (parts, not objects)" % others
        ny = y + h - 46
        draw.rectangle((x + 8, ny - 3, x + 18 + 8, ny + 7), outline=PROTO, width=1)
        draw.text((x + 34, ny - 4), note, font=fonts["small"], fill=PROTO)
    return dict((label, (cx, cy)) for label, cx, cy in placed)


def card_rows(obj, feature_names, frame):
    age = frame - obj["created_frame"]
    rows = [("age", "%d frames" % age)]
    if obj["last_selected_frame"] >= 0:
        rows.append(("last looked at", "%d frames ago" % (frame - obj["last_selected_frame"])))
    else:
        rows.append(("last looked at", "never"))
    rows.append(("times selected", str(obj["selection_count"])))
    rows.append(("size", "%d px" % obj["size"]))
    if obj.get("class"):
        rows.append(("recognised", "%s (%.2f)" % (obj["class"], obj.get("class_confidence", 0))))
    return rows


def choose_cards(objects, focus):
    """Which object files get a card: the focus, then the most recently seen.

    Returns them in draw order; the scene panel needs the same set, so this runs
    before either is drawn rather than inside draw_cards.
    """
    # The focus first, then the most salient. Ranking by last_seen_frame (the
    # obvious choice) is useless: every active file was seen on this frame, so it
    # ties and the tie-break falls through to the label, which cards whichever
    # five clusters happen to have the lowest numbers -- a head and an arm, while
    # the ball goes uncarded. Saliency is both meaningful and stable between
    # frames, which matters when this is played back at 25 fps.
    def rank(obj):
        focused = focus is not None and focus["label"] == obj["label"]
        return (0 if focused else 1, -obj["saliency"], -obj["last_seen_frame"], obj["label"])

    return sorted(objects, key=rank)[:MAX_CARDS]


def draw_cards(canvas, draw, shown, focus, feature_names, frame, fonts, anchors):
    """A card per object file, the focus first, then the most recently seen."""
    y = VIEW[1]
    for obj in shown:
        colour = identity_colour(obj["label"])
        active = obj["active"]
        height = 150
        box = (CARDS_X, y, CARDS_X + CARDS_W, y + height)
        draw.rounded_rectangle(box, 10, fill=PANEL, outline=colour if active else (60, 64, 70), width=3)
        draw.rounded_rectangle((box[0], box[1], box[0] + 8, box[3]), 4, fill=colour if active else (70, 74, 80))

        title = "object file #%d" % obj["label"]
        draw.text((CARDS_X + 24, y + 12), title, font=fonts["title"], fill=INK if active else DIM)
        if not active:
            draw.text((CARDS_X + 230, y + 16), "out of sight — still remembered",
                      font=fonts["small"], fill=DIM)
        swatch = tuple(int(max(0, min(255, c))) for c in reversed(obj["appearance"]))  # stored BGR
        draw.rounded_rectangle((CARDS_X + CARDS_W - 54, y + 12, CARDS_X + CARDS_W - 18, y + 44), 6,
                               fill=swatch, outline=(70, 74, 80))

        for i, (key, value) in enumerate(card_rows(obj, feature_names, frame)):
            ty = y + 44 + i * 20
            draw.text((CARDS_X + 24, ty), key, font=fonts["small"], fill=DIM)
            draw.text((CARDS_X + 190, ty), value, font=fonts["small"], fill=INK if active else DIM)

        feats = obj.get("features") or []
        if feats and feature_names:
            bx = CARDS_X + 380
            draw.text((bx, y + 44), "feature means", font=fonts["small"], fill=DIM)
            for i, (name, value) in enumerate(zip(feature_names, feats)):
                by = y + 64 + i * 16
                draw.text((bx, by), name[:12], font=fonts["tiny"], fill=DIM)
                draw.rectangle((bx + 86, by + 3, bx + 86 + 180, by + 11), fill=(44, 48, 54))
                draw.rectangle((bx + 86, by + 3, bx + 86 + int(180 * min(1.0, value)), by + 11),
                               fill=colour if active else (80, 84, 90))

        if active and obj["label"] in anchors:
            ax, ay = anchors[obj["label"]]
            draw.line((CARDS_X - 4, y + 28, ax, ay), fill=colour, width=2)
            draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=colour)
        y += height + 12


def draw_strip(canvas, draw, frame_dir, names, scales, fonts):
    """The feature maps, the fused saliency, and the field — fixed scales."""
    panels = [(n, "feature_%s.png" % n, RAMP_HEAT, scales.get("feature", [0, 1])) for n in names]
    panels.append(("saliency (fused)", "saliency.png", RAMP_HEAT, scales.get("saliency", [0, 1])))
    panels.append(("neural field", "field.png", RAMP_FIELD, scales.get("field", [-2, 2])))
    width = (CANVAS[0] - 40 - 12 * (len(panels) - 1)) // len(panels)
    height = STRIP_H - 20
    x = 20
    for title, filename, stops, (lo, hi) in panels:
        path = os.path.join(frame_dir, filename)
        if os.path.exists(path):
            canvas.paste(colourise(path, stops, lo, hi, (width, height)), (x, STRIP_Y))
        else:
            draw.rectangle((x, STRIP_Y, x + width, STRIP_Y + height), fill=PANEL)
            draw.text((x + 8, STRIP_Y + height // 2), "not applicable", font=fonts["tiny"], fill=DIM)
        draw.rectangle((x, STRIP_Y, x + width, STRIP_Y + height), outline=(60, 64, 70))
        draw.text((x + 4, STRIP_Y + height + 3), title, font=fonts["small"], fill=DIM)
        x += width + 12


def draw_timeline(draw, history, frame, total, fonts):
    """One row per object file, marked where it held the focus: inhibition of
    return is the pattern of *not* returning to a row just left."""
    x, y, w, h = TIMELINE
    draw.rectangle((x, y, x + w, y + h), fill=PANEL)
    labels = sorted(history)
    if not labels:
        return
    # Every file that ever held the focus gets a row. Dropping the older ones to
    # keep the rows legible would hide exactly what the panel is for: on this
    # take sixteen files are selected at some point, and the fact that there are
    # that many *is* the result.
    row_h = h / len(labels)
    named = row_h >= 9
    gutter = 44 if named else 10
    for i, label in enumerate(labels):
        ry = y + i * row_h
        colour = identity_colour(label)
        if named:
            draw.text((x + 4, ry), "#%d" % label, font=fonts["tiny"], fill=colour)
        for f in history[label]:
            fx = x + gutter + (w - gutter - 12) * (f / max(total - 1, 1))
            draw.rectangle((fx, ry + 1, fx + 2, ry + row_h - 2), fill=colour)
    cx = x + gutter + (w - gutter - 12) * (frame / max(total - 1, 1))
    draw.line((cx, y, cx, y + h), fill=FOCUS, width=1)
    # No caption here on purpose: the only free space collides with the feature
    # strip's labels. What the band means is in the module docstring and in
    # docs/DEMO_STEREO_SCENE.md.


def compose(trace_dir, frames_dir, out_dir, style, fonts):
    index = json.load(open(os.path.join(trace_dir, "trace.json")))
    total = index["frames"]
    scales = index.get("map_scales", {})
    images = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg")))
    history = {}

    for n in range(total):
        frame_dir = os.path.join(trace_dir, "frame_%04d" % n)
        state = json.load(open(os.path.join(frame_dir, "state.json")))
        focus = state.get("focus")
        if focus:
            history.setdefault(focus["label"], []).append(n)

        canvas = Image.new("RGB", CANVAS, BG)
        draw = ImageDraw.Draw(canvas)
        image = Image.open(os.path.join(frames_dir, images[min(n, len(images) - 1)])).convert("RGB")

        shown = choose_cards(state["objects"], focus)
        carded = set(o["label"] for o in shown)
        anchors = draw_scene(canvas, draw, image, state["objects"], focus, carded, fonts)
        if style == "full":
            draw_cards(canvas, draw, shown, focus, state.get("features", []),
                       state["frame"], fonts, anchors)
            draw_strip(canvas, draw, frame_dir, state.get("features", []), scales, fonts)
        draw_timeline(draw, history, n, total, fonts)

        draw.text((VIEW[0] + 10, VIEW[1] + VIEW[3] - 26),
                  "frame %d / %d   %.2f s" % (n, total - 1, n / 25.0), font=fonts["small"], fill=INK)
        if focus is None:
            draw.text((VIEW[0] + 10, VIEW[1] + 8), "no focus", font=fonts["title"], fill=DIM)
        else:
            draw.text((VIEW[0] + 10, VIEW[1] + 8), "looking at object file #%d" % focus["label"],
                      font=fonts["title"], fill=identity_colour(focus["label"]))
        canvas.save(os.path.join(out_dir, "%05d.png" % n))
        if (n + 1) % 50 == 0:
            print("  composed %d/%d" % (n + 1, total), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="attention-trace/v1 directory")
    ap.add_argument("--frames", required=True, help="the camera images the trace was computed on")
    ap.add_argument("--out", required=True, help="output .mp4 (or .gif)")
    ap.add_argument("--style", default="full", choices=["full", "clean"])
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--keep", default="", help="also keep the composed PNGs here")
    args = ap.parse_args()

    fonts = {"title": font(22), "small": font(15), "tiny": font(12)}
    work = args.keep or tempfile.mkdtemp(prefix="demo_frames_")
    os.makedirs(work, exist_ok=True)
    compose(args.trace, args.frames, work, args.style, fonts)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; composed frames are in %s" % work)
        return
    codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    if args.out.endswith(".gif"):
        codec = ["-vf", "fps=12,scale=960:-1:flags=lanczos"]
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(work, "%05d.png"),
                    *codec, args.out], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote %s" % args.out)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
