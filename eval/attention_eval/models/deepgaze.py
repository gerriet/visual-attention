"""DeepGaze IIE adapter — a learned model behind the SaliencyModel interface.

This is the "learned models Python-side" slot of the modern track. It is not
run in CI because it needs heavy optional dependencies and pretrained weights:

    eval/.venv/bin/pip install torch
    eval/.venv/bin/pip install git+https://github.com/matthias-k/DeepGaze.git

DeepGaze IIE downloads its weights on first use (via torch.hub) and needs a
centre-bias log-density; a uniform centre bias is used here for a self-contained
run. Imports are lazy so this module (and the models registry) never require
torch — the model is registered but raises a clear, actionable error if
instantiated without its dependencies.
"""

import numpy as np

from .base import SaliencyModel, normalize01

_INSTALL_HINT = (
    "DeepGaze IIE needs torch and the deepgaze_pytorch package:\n"
    "  eval/.venv/bin/pip install torch\n"
    "  eval/.venv/bin/pip install git+https://github.com/matthias-k/DeepGaze.git"
)


class DeepGazeIIE(SaliencyModel):
    name = "deepgaze-iie"

    def __init__(self, device="cpu"):
        try:
            import torch
            import deepgaze_pytorch
        except Exception as exc:  # pragma: no cover - exercised only with torch installed
            raise RuntimeError(f"{_INSTALL_HINT}\n(import failed: {exc})") from exc

        self._torch = torch
        self._device = device
        self._model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device).eval()

    def compute(self, image):  # pragma: no cover - requires torch + weights
        torch = self._torch
        rgb = np.asarray(image)[..., :3]
        h, w = rgb.shape[:2]

        # Uniform (flat) centre-bias log density — DeepGaze expects one input.
        centerbias = np.zeros((h, w), dtype=np.float32)
        centerbias -= float(np.log(np.exp(centerbias).sum()))

        image_t = torch.tensor(rgb.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32, device=self._device)
        cb_t = torch.tensor(centerbias[np.newaxis], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            log_density = self._model(image_t, cb_t)
        saliency = np.exp(log_density.cpu().numpy()[0, 0])
        return normalize01(saliency)
