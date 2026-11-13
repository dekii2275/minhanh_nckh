"""Check paired-video inference and handling of completely absent pose."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
import gt_crop_correction_experiment as experiment


class CropCorrectionTest(unittest.TestCase):
    def records(self):
        predictions = [{"id": str(i), "target": 0, "predicted": 0} for i in range(253)]
        return [
            {
                "method": method,
                "seed": seed,
                "accuracy_pct": 100.0,
                "macro_f1_pct": 100.0,
                "predictions": copy.deepcopy(predictions),
            }
            for method in experiment.METHODS
            for seed in experiment.SEEDS
        ]

    def test_bootstrap_uses_paired_video_and_seed_average(self):
        records = self.records()
        for r in records:
            if r["method"] == "mean5":
                r["predictions"][0]["predicted"] = 1
                r["accuracy_pct"] = 100 * 252 / 253
        result = experiment.paired_summary(records)
        self.assertAlmostEqual(result["mean5"]["gain_vs_baseline_pp"], -100 / 253)
        self.assertEqual(result["mean5"]["seeds"][0]["worsened_videos"], 1)
        self.assertEqual(result["fixed_swap"]["paired_video_bootstrap_95_pp"], [0.0, 0.0])

    def test_reordered_videos_are_rejected(self):
        records = self.records()
        records[-1]["predictions"].reverse()
        with self.assertRaises(ValueError):
            experiment.paired_summary(records)

    def test_entirely_missing_pose_is_not_invented(self):
        xy = np.full((8, 12, 2), np.nan, dtype=np.float32)
        conf = np.zeros((8, 12), dtype=np.float32)
        variants, filled = experiment.corrections.make_variants(xy, conf)
        self.assertFalse(filled.any())
        for method in experiment.METHODS:
            self.assertFalse(np.isfinite(variants[method][0]).any())
            np.testing.assert_array_equal(
                experiment.base.resample(variants[method][0]), np.zeros((64, 24))
            )


if __name__ == "__main__":
    unittest.main()
