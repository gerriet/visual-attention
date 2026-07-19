"""DAVIS 2017 dataset adapter (Pont-Tuset et al., the DAVIS challenge).

Per-frame, per-object segmentation masks over short video sequences — the
exact "which object is attended" ground truth the dynamic studies need
(roadmap M12 scoring, M13 gated recognition).

Layout expected under data/DAVIS/ (gitignored):

    data/DAVIS/
      JPEGImages/480p/<sequence>/00000.jpg ...
      Annotations/480p/<sequence>/00000.png ...   (indexed PNG, object ids 1..N)
      ImageSets/2017/train.txt val.txt

Download (about 800 MB, trainval 480p):

    curl -L -o DAVIS-2017-trainval-480p.zip \\
      https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
    unzip DAVIS-2017-trainval-480p.zip -d data/

The JPEGImages/480p/<sequence>/ directories are plain frame directories — the
attention CLI consumes them directly (`attention --attend <dir>`).

DAVIS carries no class labels; PERSON_OBJECTS below is a small hand-curated
map of val sequences to the annotation ids that are people (verified by
looking at the masks). Extend it as needed — and re-verify when you do.
"""

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "DAVIS"

# Hand-curated: val sequences whose listed annotation ids are human beings
# (verified against first-frame mask overlays, 2026-07). Used by the
# gated-recognition study (H2) as person localization ground truth.
# Note "loading": id 2 is the carried box, not a person.
PERSON_OBJECTS = {
    "breakdance": [1],
    "dance-twirl": [1],
    "india": [1, 2, 3],
    "judo": [1, 2],
    "loading": [1, 3],
    "parkour": [1],
}


def available(root=DEFAULT_ROOT):
    root = Path(root)
    return (root / "JPEGImages" / "480p").is_dir() and (root / "Annotations" / "480p").is_dir()


def sequences(root=DEFAULT_ROOT, split="val"):
    """Sequence names of a split ('train' / 'val'), or all present if None."""
    root = Path(root)
    if not available(root):
        raise FileNotFoundError(
            f"DAVIS 2017 not found under {root} — see this module's docstring for download steps")
    if split is None:
        return sorted(p.name for p in (root / "JPEGImages" / "480p").iterdir() if p.is_dir())
    listing = root / "ImageSets" / "2017" / f"{split}.txt"
    return [line.strip() for line in listing.read_text().splitlines() if line.strip()]


def frames_dir(sequence, root=DEFAULT_ROOT):
    """The sequence's frame directory — feed this straight to `attention --attend`."""
    return Path(root) / "JPEGImages" / "480p" / sequence


def iter_frames(sequence, root=DEFAULT_ROOT):
    """Yield (frame_index, image_path, annotation_path) triples in stream order."""
    root = Path(root)
    images = sorted(frames_dir(sequence, root).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no frames for DAVIS sequence '{sequence}' under {root}")
    for index, image in enumerate(images):
        annotation = root / "Annotations" / "480p" / sequence / f"{image.stem}.png"
        yield index, image, (annotation if annotation.exists() else None)


def load_masks(annotation_path):
    """Object masks of one frame: {object_id: bool array (H, W)}, ids 1..N."""
    indices = np.asarray(Image.open(annotation_path))
    return {int(i): indices == i for i in np.unique(indices) if i != 0}


def load_person_boxes(annotation_path, person_ids):
    """Tight bounding boxes {object_id: (x, y, w, h)} of the given person ids
    in one frame; ids absent from the frame are absent from the dict."""
    masks = load_masks(annotation_path)
    boxes = {}
    for object_id in person_ids:
        mask = masks.get(object_id)
        if mask is None or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        boxes[object_id] = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    return boxes
