"""Modern saliency models that emit the attention-result interchange format.

Each model computes a saliency map and writes the same JSON + 16-bit PNG the
C++ pipeline emits (docs/INTERCHANGE_FORMAT.md), so the evaluation harness
treats the thesis model and modern models as interchangeable peers.

Runnable models depend only on numpy + pillow:
  - spectral-residual : Hou & Zhang (2007), the canonical modern-classical baseline
  - center-bias       : a Gaussian centre prior (a strong fixation-prediction baseline)

Learned models plug in behind the same SaliencyModel interface; a torch-gated
DeepGaze adapter is provided (see models/deepgaze.py) and used when its
dependencies and weights are installed.
"""

from .base import MODELS, SaliencyModel, get_model, run_model

__all__ = ["SaliencyModel", "MODELS", "get_model", "run_model"]
