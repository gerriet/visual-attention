"""MIT1003 dataset adapter (Judd et al., "Where people look", ICCV 2009).

Layout expected under data/MIT1003/ (gitignored):

    data/MIT1003/
      ALLSTIMULI/         *.jpeg   (1003 stimulus images)
      ALLFIXATIONMAPS/    *_fixMap.jpg  (blurred fixation maps)
                          *_fixPts.jpg  (binary fixation points)

Download (about 400 MB):

    curl -O https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip
    curl -O https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip
    unzip ALLSTIMULI.zip -d data/MIT1003/
    unzip ALLFIXATIONMAPS.zip -d data/MIT1003/

The *_fixPts.jpg images give discrete fixation locations (nonzero pixels) for
NSS/AUC; the *_fixMap.jpg images are ground-truth maps for CC/SIM/KL. Raw
per-observer scanpaths require the MATLAB DATA archive and scipy (deferred to
roadmap M7).
"""

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "MIT1003"


def available(root=DEFAULT_ROOT):
    root = Path(root)
    return (root / "ALLSTIMULI").is_dir() and (root / "ALLFIXATIONMAPS").is_dir()


def iter_stimuli(root=DEFAULT_ROOT):
    """Yield (stimulus_path, fixation_map_path, fixation_points_path) triples."""
    root = Path(root)
    if not available(root):
        raise FileNotFoundError(
            f"MIT1003 not found under {root} — see this module's docstring for download steps")
    for stimulus in sorted((root / "ALLSTIMULI").glob("*.jpeg")):
        stem = stimulus.stem
        fix_map = root / "ALLFIXATIONMAPS" / f"{stem}_fixMap.jpg"
        fix_pts = root / "ALLFIXATIONMAPS" / f"{stem}_fixPts.jpg"
        if fix_map.exists() and fix_pts.exists():
            yield stimulus, fix_map, fix_pts


def load_fixation_points(fix_pts_path):
    """Discrete fixation locations [(x, y), ...] from a *_fixPts.jpg image."""
    array = np.asarray(Image.open(fix_pts_path).convert("L"))
    ys, xs = np.nonzero(array > 127)
    return list(zip(xs.tolist(), ys.tolist()))


def load_fixation_map(fix_map_path):
    """Ground-truth fixation map as float32 in [0, 1]."""
    array = np.asarray(Image.open(fix_map_path).convert("L"))
    return array.astype(np.float32) / 255.0
