"""COCO-Search18 dataset adapter (Yang, Chen, Zelinsky et al., CVPR 2020).

Goal-directed search fixations: 10 subjects × 18 target categories over COCO
images, target-present (TP) subset — the external-validity arm of the M17
priority-map study (H5). Train/validation fixations are public; test is
withheld (benchmark track).

Layout expected under data/COCO-Search18/ (gitignored):

    data/COCO-Search18/
      coco_search18_fixations_TP_train_split1.json
      coco_search18_fixations_TP_validation_split1.json
      images/<category>/<name>.jpg        (from the images archive)

Download (~30 MB fixations + ~1.1 GB images):

    curl -L -o fixations.zip \\
      http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-fixations-TP.zip
    curl -L -o images.zip \\
      http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip
    unzip fixations.zip -d data/COCO-Search18/
    unzip images.zip -d data/COCO-Search18/

Cite Yang et al. (CVPR 2020) when using the data.

Coordinate note: fixations (X/Y) are in 1680x1050 *display* coordinates —
images were resized to fit with aspect kept and zero padding; the target
`bbox` is in image coordinates. display_to_image() undoes the transform.
"""

import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "COCO-Search18"

DISPLAY_W, DISPLAY_H = 1680, 1050

CATEGORIES = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard",
              "knife", "laptop", "microwave", "mouse", "oven", "potted plant",
              "sink", "stop sign", "toilet", "tv"]


def available(root=DEFAULT_ROOT):
    root = Path(root)
    return (root / "coco_search18_fixations_TP_validation_split1.json").exists()


def load_fixations(split="validation", root=DEFAULT_ROOT, split_file=1):
    """All trial records of a split ('train' / 'validation'). One record per
    (image, task, subject): name, task, bbox [x,y,w,h] (image coords),
    X/Y fixation lists (display coords), correct, RT."""
    root = Path(root)
    path = root / f"coco_search18_fixations_TP_{split}_split{split_file}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"COCO-Search18 fixations not found at {path} — see this module's docstring")
    with open(path) as fh:
        return json.load(fh)


def image_path(record, root=DEFAULT_ROOT):
    """The image file of a trial record (images/<task>/<name>.jpg)."""
    return Path(root) / "images" / record["task"] / record["name"]


def display_to_image(x, y, image_w, image_h):
    """Map a display-coordinate fixation to image pixel coordinates (the
    display showed the image resized to fit 1680x1050, aspect kept, centred
    with zero padding)."""
    scale = min(DISPLAY_W / image_w, DISPLAY_H / image_h)
    x_off = (DISPLAY_W - image_w * scale) / 2.0
    y_off = (DISPLAY_H - image_h * scale) / 2.0
    return (x - x_off) / scale, (y - y_off) / scale


def fixations_to_target(record, image_w, image_h):
    """1-based index of the first fixation inside the target bbox, or None if
    the trial never fixates it. Fixations are converted to image coordinates."""
    bx, by, bw, bh = record["bbox"]
    for i, (dx, dy) in enumerate(zip(record["X"], record["Y"])):
        x, y = display_to_image(dx, dy, image_w, image_h)
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return i + 1
    return None


def unique_trials(records):
    """One representative entry per (image, task): the record list of every
    subject who ran that trial, keyed by (name, task), sorted for determinism."""
    grouped = {}
    for record in records:
        grouped.setdefault((record["name"], record["task"]), []).append(record)
    return dict(sorted(grouped.items()))
