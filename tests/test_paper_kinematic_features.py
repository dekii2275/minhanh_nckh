"""Analytic checks for the independently adapted paper feature formulas."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
import paper_kinematic_features as f


class PaperKinematicTest(unittest.TestCase):
    def test_right_straight_and_degenerate_angles(self):
        p = np.zeros((3, 12, 2), np.float32)
        p[:, 0] = [0, 0]
        p[:, 2] = [1, 0]
        p[:, 4] = [1, 1]
        p[1, 4] = [2, 0]
        p[2, 4] = p[2, 2]
        a, _, _, _ = f.components(p.reshape(3, 24), np.ones((3, 12), bool))
        np.testing.assert_allclose(a[:, 2], [0.5, 1.0, 0.0], atol=1e-6)

    def test_missing_transition_does_not_create_speed(self):
        p = np.zeros((4, 12, 2), np.float32)
        p[1:] = 100
        valid = np.ones((4, 12), bool)
        valid[0] = False
        a, v, com, speed = f.components(p.reshape(4, 24), valid)
        np.testing.assert_array_equal(v, np.zeros((4, 24)))
        np.testing.assert_array_equal(speed, np.zeros((4, 1)))
        np.testing.assert_array_equal(com[0], [0, 0])

    def test_constant_translation_has_analytic_velocity_and_com_speed(self):
        p = np.zeros((4, 12, 2), np.float32)
        p[:, :, 0] = np.arange(4)[:, None]
        p[:, :, 1] = 2 * np.arange(4)[:, None]
        _, v, com, speed = f.components(p.reshape(4, 24), np.ones((4, 12), bool))
        np.testing.assert_allclose(v[1:].reshape(3, 12, 2), np.tile([1, 2], (3, 12, 1)))
        np.testing.assert_allclose(com, [[0, 0], [1, 2], [2, 4], [3, 6]])
        np.testing.assert_allclose(speed[1:], np.sqrt(5), atol=1e-6)

    def test_resampling_mask_requires_both_bracketing_observations(self):
        xy = np.ones((3, 12, 2), np.float32)
        xy[1, 0] = np.nan
        mask = f.resampled_validity(xy, steps=5)
        np.testing.assert_array_equal(mask[:, 0], [True, False, False, False, True])
        self.assertTrue(mask[:, 1:].all())

    def test_all_missing_and_feature_dimensions_preserve_coordinates(self):
        pos = np.zeros((64, 24), np.float32)
        mask = np.zeros((64, 12), bool)
        for mode, dims in f.DIMENSIONS.items():
            actual = f.enrich(pos, mask, mode)
            self.assertEqual(actual.shape, (64, dims))
            np.testing.assert_array_equal(actual, np.zeros((64, dims)))
        pos = np.random.default_rng(0).normal(size=(64, 24)).astype(np.float32)
        for mode in f.DIMENSIONS:
            np.testing.assert_array_equal(f.enrich(pos, mask, mode)[:, :24], pos)


if __name__ == "__main__":
    unittest.main()
