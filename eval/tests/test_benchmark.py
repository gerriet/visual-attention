import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from attention_eval import benchmark


def _make_images(d, n=2):
    rng = np.random.default_rng(11)
    paths = []
    for i in range(n):
        img = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
        img[10:20, 25:35] = 250
        p = Path(d) / f"img{i}.png"
        Image.fromarray(img, "RGB").save(p)
        paths.append(p)
    return paths


class TestBenchmark(unittest.TestCase):
    def test_parse_spec(self):
        self.assertEqual(benchmark.parse_spec("spectral-residual"),
                         ("python", "spectral-residual", None, "spectral-residual"))
        self.assertEqual(benchmark.parse_spec("center-bias=cb"),
                         ("python", "center-bias", None, "cb"))
        self.assertEqual(benchmark.parse_spec("cpp:/bin/attention:/cfg/thesis.yaml"),
                         ("cpp", "/bin/attention", "/cfg/thesis.yaml", "thesis"))

    def test_cross_model_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            images = _make_images(d, 2)
            specs = [benchmark.parse_spec("spectral-residual"), benchmark.parse_spec("center-bias")]
            results = benchmark.run_all(specs, images, Path(d) / "out")
            rows = benchmark.aggregate_cross_model(results, "spectral-residual")

            self.assertIn("center-bias", rows)
            row = rows["center-bias"]
            for key in ("CC", "SIM", "KL", "matched", "dist", "edit"):
                self.assertIn(key, row)
                self.assertEqual(row[key], row[key])  # not NaN
            md = benchmark.cross_model_markdown(rows, "spectral-residual", len(images))
            self.assertIn("cross-model", md)
            self.assertIn("center-bias", md)

    def test_unknown_reference_errors(self):
        with tempfile.TemporaryDirectory() as d:
            images = _make_images(d, 1)
            with self.assertRaises(SystemExit):
                benchmark.main(["--images", str(images[0]), "--out", str(Path(d) / "o"),
                                "--model", "center-bias", "--reference", "does-not-exist"])

    def test_vs_gt_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            images = _make_images(d, 1)
            specs = [benchmark.parse_spec("center-bias")]
            results = benchmark.run_all(specs, images, Path(d) / "out")

            stem = images[0].stem
            gt_map = np.zeros((48, 48), dtype=np.float32)
            gt_map[22:26, 22:26] = 1.0  # a central ground-truth blob
            gt_points = [(24, 24), (23, 25)]
            rows = benchmark.aggregate_vs_gt(results, {stem: gt_map}, {stem: gt_points})

            self.assertIn("center-bias", rows)
            for key in ("AUC", "NSS", "CC", "SIM", "KL"):
                self.assertIn(key, rows["center-bias"])
            # center-bias should score AUC above chance on centrally-placed GT
            self.assertGreater(rows["center-bias"]["AUC"], 0.5)


if __name__ == "__main__":
    unittest.main()
