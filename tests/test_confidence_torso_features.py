"""Analytic checks of confidence masks and stable torso normalization."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
import confidence_torso_features as f


def skeleton(frames=3):
    xy = np.full((frames, 12, 2), 100.0, np.float32)
    xy[:, 0] = [90, 80]
    xy[:, 1] = [110, 80]
    xy[:, 6] = [95, 120]
    xy[:, 7] = [105, 120]
    return xy, np.ones((frames, 12), np.float32)


class ConfidenceTorsoTest(unittest.TestCase):
    def test_known_torso_scale_and_center(self):
        xy, conf = skeleton()
        result, stats = f.normalize_torso(xy, conf)
        self.assertEqual(stats["scale_source"], "median_torso")
        self.assertEqual(stats["scale_px"], 40.0)
        np.testing.assert_allclose(result.reshape(3, 12, 2)[:, 0], [[-0.25, -1]] * 3)
        np.testing.assert_allclose(result.reshape(3, 12, 2)[:, 6:8].mean(1), 0.0)

    def test_translation_and_scale_invariance(self):
        xy, conf = skeleton()
        a, _ = f.normalize_torso(xy, conf)
        b, _ = f.normalize_torso(xy * 2.5 + [300, 200], conf)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_low_confidence_hip_outlier_does_not_move_root(self):
        xy, conf = skeleton()
        xy[1, 6:8] = 10000
        conf[1, 6:8] = 0.1
        result, stats = f.normalize_torso(xy, conf)
        self.assertEqual(stats["root_anchor_frames"], 2)
        np.testing.assert_allclose(result.reshape(3, 12, 2)[1, 0], [-0.25, -1])
        self.assertGreater(
            result[1, 12], 100.0
        )  # preserve low-confidence joint for the explicit mask

    def test_clip_scale_is_median_and_no_joint_is_interpolated(self):
        xy, conf = skeleton()
        xy[0, :2, 1] = 90
        xy[2, :2, 1] = 70
        xy[1, 4] = np.nan
        conf[1, 4] = 0
        result, stats = f.normalize_torso(xy, conf)
        self.assertEqual(stats["scale_px"], 40.0)
        np.testing.assert_array_equal(result.reshape(3, 12, 2)[1, 4], [0.0, 0.0])

    def test_confidence_mask_resampling_and_absence(self):
        xy, conf = skeleton()
        xy[1, 4] = np.nan
        conf[1, 4] = 0.9
        cm = f.confidence_channels(xy, conf, steps=5)
        np.testing.assert_allclose(cm[:, 4], [1.0, 0.5, 0.0, 0.5, 1.0])
        np.testing.assert_array_equal(cm[:, 16], [1.0, 0.0, 0.0, 0.0, 1.0])
        conf[:, 2] = np.nan
        np.testing.assert_array_equal(f.confidence_channels(xy, conf, steps=5)[:, [2, 14]], 0.0)

    def test_all_missing_stays_zero_and_original_coordinates_unchanged(self):
        xy = np.full((5, 12, 2), np.nan, np.float32)
        conf = np.zeros((5, 12), np.float32)
        variants, stats = f.make_variants(np.zeros((64, 24), np.float32), xy, conf)
        self.assertEqual(stats["scale_source"], "fixed_fallback")
        for result in variants.values():
            np.testing.assert_array_equal(result, np.zeros_like(result))
        xy, conf = skeleton()
        original = np.random.default_rng(0).normal(size=(64, 24)).astype(np.float32)
        variants, _ = f.make_variants(original, xy, conf)
        for method in ("baseline", "confidence_mask", "capacity_control"):
            np.testing.assert_array_equal(variants[method][:, :24], original)
        np.testing.assert_array_equal(variants["capacity_control"][:, 24:], 0.0)

    def test_shoulders_scale_fallback(self):
        xy, conf = skeleton()
        xy[:, 6:8] = np.nan
        conf[:, 6:8] = 0
        result, stats = f.normalize_torso(xy, conf)
        self.assertEqual(stats["scale_source"], "median_shoulders")
        self.assertEqual(stats["scale_px"], 20.0)
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
