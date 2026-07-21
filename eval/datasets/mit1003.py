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
NSS/AUC; the *_fixMap.jpg images are ground-truth maps for CC/SIM/KL.

Per-observer ordered scanpaths (roadmap M11, H4) come from the raw eye-tracking
archive (needs scipy):

    data/MIT1003/DATA/<subject>/<stimulus_stem>.mat

    curl -O https://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip
    unzip DATA.zip -d data/MIT1003/

Each .mat holds one struct (named after the stimulus) whose `DATA` field
carries the raw gaze samples in `eyeData` — an N x >=2 array of (x, y) at the
tracker's sample rate. iter_scanpaths() runs a documented I-DT fixation filter
over those samples to recover the ordered fixation sequence per observer.

NOTE ON SCHEMA: the `eyeData` column order and sample rate below match Judd's
released archive (x, y in the first two columns; 240 Hz). If your copy differs,
adjust EYE_X_COL / EYE_Y_COL / SAMPLE_RATE_HZ — the extraction itself is
schema-independent. The parser is unit-tested against a synthetic fixture; the
real numbers land when DATA.zip is present.
"""

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "MIT1003"

# eyeData layout (Judd MIT1003 archive) — adjust here if your copy differs.
EYE_X_COL, EYE_Y_COL = 0, 1
SAMPLE_RATE_HZ = 240.0


def available(root=DEFAULT_ROOT):
    root = Path(root)
    return (root / "ALLSTIMULI").is_dir() and (root / "ALLFIXATIONMAPS").is_dir()


def scanpaths_available(root=DEFAULT_ROOT):
    return (Path(root) / "DATA").is_dir()


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


# --- Per-observer ordered scanpaths (roadmap M11) ----------------------------

def idt_fixations(samples, size, dispersion_frac=0.05, min_duration_s=0.1,
                  sample_rate=SAMPLE_RATE_HZ):
    """Extract ordered fixations from raw gaze samples via I-DT
    (Salvucci & Goldberg 2000): a fixation is a run of samples whose spatial
    dispersion stays below a threshold for at least min_duration_s.

    samples  N x 2 array of (x, y) gaze positions (pixels)
    size     (w, h) of the stimulus, used for the dispersion threshold and to
             drop off-screen samples
    Returns [(x, y), ...] fixation centroids in order.
    """
    w, h = size
    disp_thresh = dispersion_frac * min(w, h)
    min_samples = max(2, int(round(min_duration_s * sample_rate)))

    pts = np.asarray(samples, dtype=np.float64)
    # Keep on-screen, finite samples.
    ok = np.isfinite(pts).all(axis=1) & (pts[:, 0] >= 0) & (pts[:, 0] < w) \
        & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[ok]

    fixations = []
    i, n = 0, len(pts)
    while i < n:
        j = i + min_samples
        if j > n:
            break
        window = pts[i:j]
        if _dispersion(window) > disp_thresh:
            i += 1
            continue
        # Grow the window while dispersion stays under threshold.
        while j < n and _dispersion(pts[i:j + 1]) <= disp_thresh:
            j += 1
        fixations.append((float(pts[i:j, 0].mean()), float(pts[i:j, 1].mean())))
        i = j
    return fixations


def _dispersion(window):
    """I-DT dispersion: (xmax-xmin) + (ymax-ymin)."""
    return (window[:, 0].max() - window[:, 0].min()) + (window[:, 1].max() - window[:, 1].min())


def _extract_eye_samples(mat):
    """Pull the N x 2 (x, y) gaze array out of a loaded MIT1003 .mat dict,
    tolerant of the struct nesting. Raises with guidance if not found."""
    for key, value in mat.items():
        if key.startswith("__"):
            continue  # scipy metadata (__header__, __globals__, ...)
        struct = value
        # Descend through the struct to a `DATA` field with `eyeData`.
        try:
            data = struct["DATA"][0, 0] if struct.dtype.names and "DATA" in struct.dtype.names else struct
            eye = data["eyeData"][0, 0] if data.dtype.names and "eyeData" in data.dtype.names else data
            arr = np.asarray(eye, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] > max(EYE_X_COL, EYE_Y_COL):
                return arr[:, [EYE_X_COL, EYE_Y_COL]]
        except (TypeError, IndexError, KeyError, AttributeError):
            continue
    raise ValueError("no eyeData (N x >=2 gaze array) found in the .mat — see mit1003 docstring")


def iter_scanpaths(stimulus_stem, size, root=DEFAULT_ROOT):
    """Yield (subject, [(x, y), ...]) ordered fixation sequences for one
    stimulus, one entry per observer under DATA/. `size` is the stimulus (w, h).
    Needs scipy."""
    from scipy.io import loadmat  # optional dep — see eval/requirements.txt

    data_root = Path(root) / "DATA"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"MIT1003 DATA archive not found under {data_root} — see this module's docstring")
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        mat_path = subject_dir / f"{stimulus_stem}.mat"
        if not mat_path.exists():
            continue
        samples = _extract_eye_samples(loadmat(str(mat_path)))
        fixations = idt_fixations(samples, size)
        if fixations:
            yield subject_dir.name, fixations
