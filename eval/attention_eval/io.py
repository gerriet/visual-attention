"""Loading of attention-result/v1 interchange files."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


class Result:
    """One attention result: metadata, fixation sequence, saliency map."""

    def __init__(self, data, saliency, path):
        self.data = data
        self.saliency = saliency  # float32 ndarray in [0, 1], or None
        self.path = Path(path)

    @property
    def fixations(self):
        """Fixations as a list of (x, y) tuples in scanpath order."""
        return [(f["x"], f["y"]) for f in self.data.get("fixations", [])]

    @property
    def fixation_values(self):
        return [f.get("value", 0.0) for f in self.data.get("fixations", [])]

    @property
    def variant(self):
        return self.data.get("generator", {}).get("variant", "?")

    @property
    def size(self):
        source = self.data.get("source", {})
        return (source.get("width"), source.get("height"))


def load_map(path):
    """Load a 16-bit grayscale PNG saliency map as float32 in [0, 1]."""
    image = Image.open(path)
    array = np.asarray(image)
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    return array.astype(np.float32)


def load_result(json_path):
    """Load a result JSON and its sibling saliency map (if present)."""
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    schema = data.get("schema", "")
    if not schema.startswith("attention-result/"):
        raise ValueError(f"{json_path} is not an attention-result file (schema={schema!r})")

    saliency = None
    map_name = data.get("saliency_map")
    if map_name:
        map_path = json_path.parent / map_name
        if map_path.exists():
            saliency = load_map(map_path)

    return Result(data, saliency, json_path)
