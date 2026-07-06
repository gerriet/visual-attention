import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from attention_eval import io
from attention_eval.models import MODELS, get_model, run_model


def _sample_image(path, size=64):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    # a bright square so there is structure to find
    img[20:40, 30:50] = 250
    Image.fromarray(img, "RGB").save(path)


class TestModels(unittest.TestCase):
    def test_runnable_models_registered(self):
        self.assertIn("spectral-residual", MODELS)
        self.assertIn("center-bias", MODELS)

    def test_spectral_residual_map(self):
        model = get_model("spectral-residual")
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[10:20, 30:40] = 255
        sal = model.compute(img)
        self.assertEqual(sal.shape, (48, 64))
        self.assertTrue(np.isfinite(sal).all())
        self.assertGreaterEqual(sal.min(), 0.0)
        self.assertLessEqual(sal.max(), 1.0 + 1e-6)

    def test_center_bias_peaks_at_center(self):
        model = get_model("center-bias")
        sal = model.compute(np.zeros((100, 100, 3), dtype=np.uint8))
        peak = np.unravel_index(int(np.argmax(sal)), sal.shape)
        self.assertLess(abs(peak[0] - 50), 5)
        self.assertLess(abs(peak[1] - 50), 5)

    def test_interchange_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "img.png"
            _sample_image(img_path)
            out_json = Path(d) / "out.json"
            run_model(get_model("spectral-residual"), img_path, out_json)

            result = io.load_result(out_json)
            self.assertTrue(result.data["schema"].startswith("attention-result/"))
            self.assertEqual(result.variant, "spectral-residual")
            self.assertEqual(result.size, (64, 64))
            self.assertGreater(len(result.fixations), 0)
            self.assertIsNotNone(result.saliency)
            self.assertEqual(result.saliency.shape, (64, 64))
            # fixations within bounds
            for x, y in result.fixations:
                self.assertTrue(0 <= x < 64 and 0 <= y < 64)

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            get_model("no-such-model")


if __name__ == "__main__":
    unittest.main()
