"""DAVIS 2017 dataset adapter (Pont-Tuset et al., the DAVIS challenge).

Per-frame, per-object segmentation masks over short video sequences — the
exact "which object is attended" ground truth the dynamic studies need
(roadmap M12 scoring, M13 gated recognition).

Layout expected under data/DAVIS/ (gitignored); `resolution` picks the
sub-directory, "480p" or "Full-Resolution":

    data/DAVIS/
      JPEGImages/480p/<sequence>/00000.jpg ...
      Annotations/480p/<sequence>/00000.png ...   (indexed PNG, object ids 1..N)
      ImageSets/2017/train.txt val.txt

Download (about 800 MB, trainval 480p):

    curl -L -o DAVIS-2017-trainval-480p.zip \\
      https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
    unzip DAVIS-2017-trainval-480p.zip -d data/

Full resolution (about 4 GB, up to 4K frames — the M19 video front-end needs
it: at 480p every arm sees the objects, see docs/VLM_VIDEO.md). It unpacks
into the same DAVIS/ tree, so both resolutions can live side by side:

    curl -L -o DAVIS-2017-trainval-Full-Resolution.zip \\
      https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip
    unzip DAVIS-2017-trainval-Full-Resolution.zip -d data/

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

# Hand-curated: the category of every annotated object in the 30 val
# sequences (verified against mask overlays of the frame showing the most ids,
# 2026-09; M19's video front-end asks which categories a video shows). The
# small ones — phones, rope, kite, box, gun — are where native-resolution
# crops should matter.
OBJECT_CATEGORIES = {
    "bike-packing": {1: "bicycle", 2: "person"},
    "blackswan": {1: "swan"},
    "bmx-trees": {1: "bicycle", 2: "person"},
    "breakdance": {1: "person"},
    "camel": {1: "camel"},
    "car-roundabout": {1: "car"},
    "car-shadow": {1: "car"},
    "cows": {1: "cow"},
    "dance-twirl": {1: "person"},
    "dog": {1: "dog"},
    "dogs-jump": {1: "dog", 2: "dog", 3: "person"},
    "drift-chicane": {1: "car"},
    "drift-straight": {1: "car"},
    "goat": {1: "goat"},
    "gold-fish": {1: "fish", 2: "fish", 3: "fish", 4: "fish", 5: "fish"},
    "horsejump-high": {1: "horse", 2: "person"},
    "india": {1: "person", 2: "person", 3: "person"},
    "judo": {1: "person", 2: "person"},
    "kite-surf": {1: "kite", 2: "surfboard", 3: "person"},
    "lab-coat": {1: "phone", 2: "phone", 3: "person", 4: "person", 5: "person"},
    "libby": {1: "dog"},
    "loading": {1: "person", 2: "box", 3: "person"},
    "mbike-trick": {1: "person", 2: "motorbike"},
    "motocross-jump": {1: "person", 2: "motorbike"},
    "paragliding-launch": {1: "backpack", 2: "person", 3: "paraglider"},
    "parkour": {1: "person"},
    "pigs": {1: "pig", 2: "pig", 3: "pig"},
    "scooter-black": {1: "person", 2: "scooter"},
    "shooting": {1: "gun", 2: "person", 3: "rope"},
    "soapbox": {1: "soapbox", 2: "person", 3: "person"},
}


RESOLUTIONS = ("480p", "Full-Resolution")


def available(root=DEFAULT_ROOT, resolution="480p"):
    root = Path(root)
    return (root / "JPEGImages" / resolution).is_dir() and (root / "Annotations" / resolution).is_dir()


def sequences(root=DEFAULT_ROOT, split="val", resolution="480p"):
    """Sequence names of a split ('train' / 'val'), or all present if None."""
    root = Path(root)
    if not available(root, resolution):
        raise FileNotFoundError(
            f"DAVIS 2017 ({resolution}) not found under {root} — see this module's docstring for download steps")
    if split is None:
        return sorted(p.name for p in (root / "JPEGImages" / resolution).iterdir() if p.is_dir())
    listing = root / "ImageSets" / "2017" / f"{split}.txt"
    return [line.strip() for line in listing.read_text().splitlines() if line.strip()]


def frames_dir(sequence, root=DEFAULT_ROOT, resolution="480p"):
    """The sequence's frame directory — feed this straight to `attention --attend`."""
    return Path(root) / "JPEGImages" / resolution / sequence


def iter_frames(sequence, root=DEFAULT_ROOT, resolution="480p"):
    """Yield (frame_index, image_path, annotation_path) triples in stream order."""
    root = Path(root)
    images = sorted(frames_dir(sequence, root, resolution).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no frames for DAVIS sequence '{sequence}' ({resolution}) under {root}")
    for index, image in enumerate(images):
        annotation = root / "Annotations" / resolution / sequence / f"{image.stem}.png"
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
