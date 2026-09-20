"""The opt-in VLM front-end arms — oracle crops and the top-down relevance map —
and the HR-Bench adapter's extraction (on a tiny generated parquet)."""

import base64
import io
import tempfile
import unittest
from pathlib import Path

from datasets import hrbench
from vlm_frontend import make_view, oracle_views, relevance_map, target_visible

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None

PARAMS = {"fovea_side": 224, "k": 3}


@unittest.skipIf(Image is None, "needs PIL (eval venv)")
class TestOracleViews(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (2000, 1500))
        self.global_view = make_view(Image.new("RGB", (512, 384)), (0, 0, 2000, 1500), 0.256)
        # A fovea arm with three saliency crops, one of them overlapping the target.
        self.fovea = [self.global_view] + [
            make_view(Image.new("RGB", (224, 224)), box, 1.0)
            for box in ((0, 0, 224, 224), (1000, 700, 1224, 924), (1500, 1100, 1724, 1324))]

    def test_crop_on_target_same_crop_count(self):
        target = [(1100, 800, 1130, 840)]
        views = oracle_views(self.image, self.global_view, target, self.fovea, PARAMS)
        self.assertEqual(len(views), len(self.fovea))
        self.assertTrue(target_visible(views[1:2], target, 0))
        # The saliency crop that overlaps the oracle window is not reused.
        self.assertNotIn((1000, 700, 1224, 924), [v["source_box"] for v in views])

    def test_large_target_window_grows_then_resizes_to_budget(self):
        target = [(100, 100, 700, 500)]  # 600 px wide > fovea_side
        crop = oracle_views(self.image, self.global_view, target, self.fovea, PARAMS)[1]
        self.assertLessEqual(max(crop["image"].size), PARAMS["fovea_side"])
        self.assertTrue(target_visible([crop], target, 0))


@unittest.skipIf(Image is None, "needs PIL (eval venv)")
class TestRelevanceMap(unittest.TestCase):
    def test_plateau_on_the_box_dark_elsewhere(self):
        raster = relevance_map((400, 300), [(200, 100, 240, 160)])
        self.assertEqual(raster.getpixel((220, 130)), 255)
        self.assertEqual(raster.getpixel((20, 280)), 0)


@unittest.skipIf(Image is None or pyarrow is None, "needs PIL + pyarrow")
class TestHRBenchExtraction(unittest.TestCase):
    def test_only_cycle_zero_extracted_with_choices(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        rows = [{"index": i, "question": "Which colour?", "A": "red", "B": "green", "C": "blue",
                 "D": "gray", "answer": "A", "category": "single", "cycle_category": i % 4,
                 "image": encoded} for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pyarrow.parquet.write_table(pyarrow.Table.from_pylist(rows), root / "hr_bench_4k.parquet")
            items = list(hrbench.iter_items("4k", root=root))
            self.assertEqual([it["question_id"] for it in items], [0, 4])
            self.assertEqual(items[0]["answer"], "red")
            self.assertEqual(sorted(items[0]["choices"]), ["blue", "gray", "green", "red"])
            # The letter follows the answer through the re-ordering.
            first = items[0]
            self.assertEqual(first["choices"][ord(first["answer_letter"]) - 65], "red")
            self.assertTrue(items[0]["image"].exists())
            self.assertEqual(items[0]["image"].suffix, ".jpg")

    def test_correct_letter_is_spread_and_stable(self):
        # Cycle 0 stores the correct option as "A" for every question; served
        # as stored, always answering "A" would score 1.0.
        options = ["right", "wrong-1", "wrong-2", "wrong-3"]
        letters = [hrbench.shuffled_choices(options, qid).index("right") for qid in range(200)]
        for position in range(4):
            self.assertGreater(letters.count(position), 30)
        self.assertEqual(hrbench.shuffled_choices(options, 7), hrbench.shuffled_choices(options, 7))


if __name__ == "__main__":
    unittest.main()
