"""Check geometry, missing-frame semantics and uncertainty of imputation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
import midhip_refinement_features as f


def sample(frames=5):
    xy = np.full((frames, 12, 2), 100.0, np.float32)
    xy[:, 0] = [90, 80]
    xy[:, 1] = [110, 80]
    xy[:, 6] = [95, 120]
    xy[:, 7] = [105, 120]
    return xy, np.ones((frames, 12), np.float32)


class MidhipRefinementTest(unittest.TestCase):
    def test_midhip_shoulder_known_geometry(self):
        xy, conf = sample()
        variants, d = f.variants(xy, conf)
        self.assertEqual(d["median_shoulder_scale_px"], 20.0)
        self.assertEqual(d["torso_scale_px"], 40.0)
        np.testing.assert_allclose(
            variants["median_shoulder"][:, :24], 2 * variants["reference"][:, :24]
        )
        np.testing.assert_allclose(
            variants["strict_midhip_shoulder"][:, :24], variants["median_shoulder"][:, :24]
        )

    def test_strict_invalid_geometry_zeros_coordinates_and_metadata(self):
        xy, conf = sample()
        conf[:, 6] = 0.1
        variants, d = f.variants(xy, conf)
        self.assertEqual(d["strict_valid_frames"], 0)
        np.testing.assert_array_equal(variants["strict_midhip_shoulder"], 0.0)
        self.assertTrue(np.any(variants["median_shoulder"][:, :24] != 0))

    def test_interpolated_joint_does_not_become_observed(self):
        xy, conf = sample()
        xy[2, 4] = np.nan
        conf[2, 4] = 0.0
        variants, d = f.variants(xy, conf)
        self.assertEqual(d["filled_joint_frames"], 1)
        np.testing.assert_array_equal(
            variants["interpolate3"][:, 24:], variants["reference"][:, 24:]
        )
        self.assertTrue(np.any(variants["interpolate3"][:, 8:10] != variants["reference"][:, 8:10]))

    def test_smooth_root_preserves_uncertainty_and_changes_only_translation(self):
        xy, conf = sample()
        xy[2, 6:8, 0] += 20
        variants, _ = f.variants(xy, conf)
        np.testing.assert_array_equal(
            variants["smooth_midhip"][:, 24:], variants["reference"][:, 24:]
        )
        delta = (variants["smooth_midhip"][:, :24] - variants["reference"][:, :24]).reshape(
            64, 12, 2
        )
        np.testing.assert_allclose(delta, np.repeat(delta[:, :1], 12, axis=1), atol=1e-6)

    def test_all_missing_and_short_clip(self):
        xy = np.full((1, 12, 2), np.nan, np.float32)
        conf = np.zeros((1, 12), np.float32)
        variants, _ = f.variants(xy, conf)
        for result in variants.values():
            self.assertEqual(result.shape, (64, 48))
            np.testing.assert_array_equal(result, np.zeros((64, 48)))

    def test_strict_clips_extreme_coordinates(self):
        xy, conf = sample()
        xy[:, 4] = 10000
        variants, _ = f.variants(xy, conf)
        self.assertLessEqual(np.abs(variants["strict_midhip_shoulder"][:, :24]).max(), f.CLIP)
        self.assertGreater(np.abs(variants["reference"][:, :24]).max(), f.CLIP)


if __name__ == "__main__":
    unittest.main()
