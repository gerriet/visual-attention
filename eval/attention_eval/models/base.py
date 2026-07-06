"""SaliencyModel interface + interchange writer shared by all modern models."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class SaliencyModel(ABC):
    """A saliency model: image (H, W, 3) uint8 -> saliency (H, W) float in [0, 1]."""

    #: registry key / interchange variant name
    name = "abstract"

    @abstractmethod
    def compute(self, image):
        """Return a float32 saliency map in [0, 1], same H×W as the image."""
        raise NotImplementedError


def normalize01(array):
    array = array.astype(np.float32)
    lo, hi = float(array.min()), float(array.max())
    if hi - lo < 1e-12:
        return np.zeros_like(array)
    return (array - lo) / (hi - lo)


def nms_peaks(saliency, max_count=10, min_distance=30, threshold=0.2):
    """Iterative peak selection: repeatedly take the global max above `threshold`
    and suppress a disk of radius `min_distance` around it. Returns
    [(x, y, value), ...] in descending-value (scanpath) order."""
    work = saliency.copy().astype(np.float32)
    h, w = work.shape
    yy, xx = np.mgrid[0:h, 0:w]
    peaks = []
    for _ in range(max_count):
        idx = int(np.argmax(work))
        y, x = divmod(idx, w)
        value = float(work[y, x])
        if value < threshold:
            break
        peaks.append((x, y, value))
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= min_distance * min_distance
        work[disk] = -1.0
    return peaks


def load_image_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def write_interchange(json_path, image_path, saliency, peaks, model_name, params=None, timing_ms=None):
    """Write an attention-result/v1 JSON + sibling 16-bit saliency PNG."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = saliency.shape
    map_name = json_path.stem + "_saliency.png"
    map16 = (np.clip(saliency, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    # uint16 array -> Pillow infers the 16-bit "I;16" mode (no deprecated mode= arg)
    Image.fromarray(map16).save(json_path.parent / map_name)

    doc = {
        "schema": "attention-result/v1",
        "source": {"image": str(image_path), "width": int(w), "height": int(h)},
        "generator": {"name": "attention-eval", "variant": model_name},
        "params": params or {},
        "saliency_map": map_name,
        "fixations": [{"n": i, "x": int(x), "y": int(y), "value": round(float(v), 6)}
                      for i, (x, y, v) in enumerate(peaks)],
        "timing_ms": timing_ms or {},
    }
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2)
    return json_path


def run_model(model, image_path, out_json, max_count=10, min_distance=30, threshold=0.2):
    """Compute a model's saliency for an image and write the interchange files."""
    image = load_image_rgb(image_path)
    saliency = normalize01(model.compute(image))
    peaks = nms_peaks(saliency, max_count=max_count, min_distance=min_distance, threshold=threshold)
    return write_interchange(out_json, image_path, saliency, peaks, model.name)


# --- registry -------------------------------------------------------------

def _build_registry():
    from .center_bias import CenterBias
    from .spectral_residual import SpectralResidual

    models = {
        SpectralResidual.name: SpectralResidual,
        CenterBias.name: CenterBias,
    }
    # DeepGaze is optional (needs torch + weights); register lazily so importing
    # this package never requires torch.
    try:
        from .deepgaze import DeepGazeIIE
        models[DeepGazeIIE.name] = DeepGazeIIE
    except Exception:
        pass
    return models


MODELS = _build_registry()


def get_model(name, **kwargs):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODELS))}")
    return MODELS[name](**kwargs)
