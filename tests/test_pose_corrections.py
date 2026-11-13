"""Tests for gap and smoothing boundaries without GT input."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
from evaluate_pose_corrections import (
    SWAP,
    displacement_error,
    interpolate_short,
    make_variants,
    smooth5,
)


class PoseCorrectionTest(unittest.TestCase):
    def test_interpolate_bounded_gap_and_keep_unresolved(self):
        xy = np.ones((8, 12, 2), np.float32) * 10
        conf = np.ones((8, 12), np.float32)
        xy[4, 0] = [30, 50]
        xy[1:4, 0] = np.nan
        conf[1:4, 0] = 0
        xy[:2, 1] = np.nan
        conf[:2, 1] = 0
        xy[2:6, 2] = 0
        conf[2:6, 2] = 0
        result, c, mask = interpolate_short(xy, conf)
        np.testing.assert_allclose(result[1, 0], [15, 20])
        np.testing.assert_allclose(result[3, 0], [25, 40])
        self.assertTrue(mask[1:4, 0].all())
        self.assertTrue(np.isnan(result[:2, 1]).all())
        self.assertFalse(mask[2:6, 2].any())
        self.assertTrue((result[2:6, 2] == 0).all())
        self.assertTrue((c[1:4, 0] == 1).all())

    def test_low_confidence_endpoints_are_not_trusted(self):
        xy = np.ones((5, 12, 2), np.float32) * 10
        conf = np.ones((5, 12), np.float32)
        xy[2, 0] = 999
        conf[2, 0] = 0.1
        out, _, filled = interpolate_short(xy, conf)
        np.testing.assert_array_equal(out[2, 0], [10, 10])
        self.assertTrue(filled[2, 0])

    def test_smoothing_respects_target_and_minimum_valid(self):
        xy = np.ones((5, 12, 2), np.float32) * 10
        conf = np.ones((5, 12), np.float32)
        xy[2, 0] = 40
        xy[2, 1] = np.nan
        conf[2, 1] = 0
        out = smooth5(xy, conf, "median")
        np.testing.assert_array_equal(out[2, 0], [10, 10])
        self.assertTrue(np.isnan(out[2, 1]).all())
        conf[:, 2] = 0
        conf[2, 2] = 1
        np.testing.assert_array_equal(smooth5(xy, conf, "mean")[2, 2], xy[2, 2])

    def test_baselines_apply_swap_once(self):
        xy = np.tile(np.arange(12, dtype=np.float32)[None, :, None] + 1, (5, 1, 2))
        conf = np.ones((5, 12), np.float32)
        variants, _ = make_variants(xy, conf)
        np.testing.assert_array_equal(variants["raw_identity"][0], xy)
        np.testing.assert_array_equal(variants["fixed_swap"][0], xy[:, SWAP])

    def test_unsigned_gt_motion_does_not_wrap(self):
        gt = np.array([[[12, 20]], [[10, 20]]], dtype=np.uint16)
        pred = np.array([[[12, 20]], [[10, 20]]], dtype=np.float32)
        np.testing.assert_array_equal(displacement_error(pred, gt, np.array([10.0, 10.0])), [[0.0]])


if __name__ == "__main__":
    unittest.main()
