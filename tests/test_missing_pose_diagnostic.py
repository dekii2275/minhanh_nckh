"""Tests for short-gap interpolation and bounded optical flow."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
from compare_missing_pose import eligible_gaps, flow_box, interpolate, pose_stats


class MissingPoseDiagnosticTest(unittest.TestCase):
    def test_only_short_internal_gaps_are_eligible(self):
        b = np.ones((12, 4), np.float32)
        b[[0, 2, 4, 5, 7, 8, 9, 10, 11]] = np.nan
        self.assertEqual(eligible_gaps(b), [(2, 3), (4, 6)])

    def test_interpolation_respects_joint_availability(self):
        p = np.ones((3, 12, 2), np.float32)
        p[1] = np.nan
        p[2] *= 3
        p[2, 0] = 0
        c = np.ones((3, 12), np.float32)
        c[1] = 0
        out, conf = interpolate(p, c, [(1, 2)])
        np.testing.assert_allclose(out[1, 1:], 2)
        self.assertTrue(np.isnan(out[1, 0]).all())
        self.assertEqual(conf[1, 0], 0)

    def test_invalid_bbox_is_excluded_from_pose_denominator(self):
        gt = np.ones((2, 12, 2))
        vis = np.ones((2, 12))
        pred = gt.copy()
        conf = vis.copy()
        boxes = np.array([[0, 0, 10, 10], [10, 0, 0, 10]])
        good, _, _ = pose_stats(gt, vis, boxes, pred, conf, np.ones(2, bool))
        self.assertEqual(int(good[0].sum()), 12)
        self.assertEqual(int(good[1].sum()), 0)

    def test_optical_flow_follows_shift(self):
        rng = np.random.default_rng(42)
        a = np.zeros((100, 100, 3), np.uint8)
        a[25:65, 25:65] = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        b = np.zeros_like(a)
        b[28:68, 29:69] = a[25:65, 25:65]
        box = np.array([23, 23, 67, 67], np.float32)
        shifted, n = flow_box(a, b, box)
        self.assertGreaterEqual(n, 4)
        np.testing.assert_allclose(shifted - box, [4, 3, 4, 3], atol=0.8)


if __name__ == "__main__":
    unittest.main()
