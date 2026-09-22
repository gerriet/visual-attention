"""DAVIS 2017 (Pont-Tuset et al. 2017) as ground truth for the dynamic-IOR study.

The instance masks (Annotations/480p/<seq>/NNNNN.png, palette images whose
pixel value is the object id) give, per frame and object, whether it is visible
and where it is. This module writes the study's `dynamic-scene-gt/v1` file for a
sequence — per object a position list with centroid and box — and adds what a
synthetic scene does not have: the masks themselves, so that "which object was
attended" is decided by the focus lying on the object's mask (dilated by a
margin), not by a centroid radius that means nothing for a person or a bike.

Layout expected (the public release, after unzipping):
  <root>/JPEGImages/480p/<seq>/NNNNN.jpg
  <root>/Annotations/480p/<seq>/NNNNN.png
  <root>/ImageSets/2017/{train,val}.txt
"""
import json
import os

import numpy as np
from PIL import Image

MIN_VISIBLE_PIXELS = 50  # an object with fewer mask pixels in a frame counts as not visible


def sequences(root, split):
    path = os.path.join(root, "ImageSets", "2017", split + ".txt")
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def frame_dir(root, seq):
    return os.path.join(root, "JPEGImages", "480p", seq)


def mask_dir(root, seq):
    return os.path.join(root, "Annotations", "480p", seq)


def load_mask(path):
    """Object ids per pixel (0 = background), from the palette PNG."""
    return np.asarray(Image.open(path).convert("P"), dtype=np.uint8)


def ground_truth(root, seq):
    """`dynamic-scene-gt/v1` for a sequence, with `masks` pointing at its
    annotation directory. Object ids are the mask values (1-based)."""
    files = sorted(f for f in os.listdir(mask_dir(root, seq)) if f.endswith(".png"))
    if not files:
        raise SystemExit("no annotations for %s under %s" % (seq, root))
    first = load_mask(os.path.join(mask_dir(root, seq), files[0]))
    height, width = first.shape
    ids = set()
    per_frame = []
    for name in files:
        mask = load_mask(os.path.join(mask_dir(root, seq), name))
        ids |= set(int(v) for v in np.unique(mask) if v != 0)
        per_frame.append(mask)
    objects = []
    for oid in sorted(ids):
        positions = []
        for f, mask in enumerate(per_frame):
            ys, xs = np.nonzero(mask == oid)
            visible = len(xs) >= MIN_VISIBLE_PIXELS
            if visible:
                positions.append({"frame": f, "x": int(round(xs.mean())), "y": int(round(ys.mean())),
                                  "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1),
                                  "visible": True})
            else:
                positions.append({"frame": f, "x": 0, "y": 0, "w": 0, "h": 0, "visible": False})
        objects.append({"id": oid, "positions": positions})
    return {
        "schema": "dynamic-scene-gt/v1",
        "source": "davis2017",
        "sequence": seq,
        "width": int(width),
        "height": int(height),
        "frames": len(files),
        "masks": os.path.abspath(mask_dir(root, seq)),
        "mask_files": files,
        "objects": objects,
    }


def write_ground_truth(root, seq, out_path):
    gt = ground_truth(root, seq)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(gt, fh)
    return gt


class MaskLookup:
    """Which object a focus point lies on, by the (dilated) instance mask of the
    frame. Masks are loaded lazily and cached per frame."""

    def __init__(self, gt, margin_px):
        self.gt = gt
        self.margin = int(margin_px)
        self._cache = {}

    def _dilated(self, frame):
        if frame in self._cache:
            return self._cache[frame]
        mask = load_mask(os.path.join(self.gt["masks"], self.gt["mask_files"][frame]))
        if self.margin > 0:
            # Grow every object's mask by the margin, nearest object wins where
            # dilations meet (a distance transform per object is exact; a box
            # filter over the id map is not — do it per object).
            from scipy import ndimage  # optional dependency, present in eval/.venv
            grown = np.zeros_like(mask)
            distance = np.full(mask.shape, np.inf)
            for oid in np.unique(mask):
                if oid == 0:
                    continue
                d = ndimage.distance_transform_edt(mask != oid)
                take = (d <= self.margin) & (d < distance)
                grown[take] = oid
                distance[take] = d[take]
            mask = grown
        self._cache[frame] = mask
        return mask

    def object_at(self, frame, x, y):
        if frame >= self.gt["frames"]:
            return None
        mask = self._dilated(frame)
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]):
            return None
        oid = int(mask[yi, xi])
        return oid if oid != 0 else None
