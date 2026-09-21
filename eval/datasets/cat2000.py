"""CAT2000 dataset adapter (Borji & Itti, 2015).

A second stills dataset for the human-comparison study (roadmap M11) spanning
20 scene categories. Layout expected under data/CAT2000/ (gitignored):

    data/CAT2000/trainSet/
      Stimuli/<Category>/<n>.jpg
      FIXATIONMAPS/<Category>/<n>.jpg   (continuous ground-truth map)
      FIXATIONLOCS/<Category>/<n>.mat   (binary fixation-location map, var 'fixLocs')

Download (trainSet, ~4 GB):

    curl -O http://saliency.mit.edu/trainSet.zip
    unzip trainSet.zip -d data/CAT2000/

IMPORTANT — what CAT2000 provides. The public release ships *pooled* fixation
locations and continuous maps, **not** ordered per-observer scanpaths. So
CAT2000 feeds the map metrics (AUC/NSS/CC/SIM) and a fixation-coverage
cross-check across categories — it is the second-dataset saliency check.
The ordered MultiMatch/ScanMatch comparison runs on MIT1003, whose raw archive
does carry per-observer sequences (see datasets/mit1003.py).
"""

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "CAT2000"


def _train(root):
    return Path(root) / "trainSet"


def available(root=DEFAULT_ROOT):
    train = _train(root)
    return (train / "Stimuli").is_dir() and (train / "FIXATIONLOCS").is_dir()


def categories(root=DEFAULT_ROOT):
    stim = _train(root) / "Stimuli"
    if not stim.is_dir():
        raise FileNotFoundError(
            f"CAT2000 not found under {stim} — see this module's docstring for download steps")
    return sorted(p.name for p in stim.iterdir() if p.is_dir())


def iter_stimuli(root=DEFAULT_ROOT, category=None):
    """Yield (stimulus_path, fixation_map_path, fixation_locs_path) triples,
    optionally restricted to one category."""
    train = _train(root)
    if not available(root):
        raise FileNotFoundError(
            f"CAT2000 not found under {train} — see this module's docstring for download steps")
    cats = [category] if category else categories(root)
    for cat in cats:
        for stimulus in sorted((train / "Stimuli" / cat).glob("*.jpg")):
            fix_map = train / "FIXATIONMAPS" / cat / stimulus.name
            fix_locs = train / "FIXATIONLOCS" / cat / (stimulus.stem + ".mat")
            if fix_map.exists() and fix_locs.exists():
                yield stimulus, fix_map, fix_locs


def load_fixation_points(fix_locs_path):
    """Pooled fixation locations [(x, y), ...] from a FIXATIONLOCS .mat
    (the `fixLocs` binary map). Needs scipy."""
    from scipy.io import loadmat  # optional dep — see eval/requirements.txt

    mat = loadmat(str(fix_locs_path))
    locs = mat.get("fixLocs")
    if locs is None:  # tolerate a differently-named single variable
        arrays = [v for k, v in mat.items() if not k.startswith("__")]
        locs = arrays[0] if arrays else None
    if locs is None:
        raise ValueError("no fixLocs binary map found in %s" % fix_locs_path)
    ys, xs = np.nonzero(np.asarray(locs) > 0)
    return list(zip(xs.tolist(), ys.tolist()))


def load_fixation_map(fix_map_path):
    """Continuous ground-truth fixation map as float32 in [0, 1]."""
    array = np.asarray(Image.open(fix_map_path).convert("L"))
    return array.astype(np.float32) / 255.0
