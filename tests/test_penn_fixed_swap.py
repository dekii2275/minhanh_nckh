import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penn_fixed_swap import SWAP, penn_fixed_swap, penn_fixed_swap_calibrated
from skeleton_calibrator import FullCalibrationPipeline


class TestPennFixedSwap(unittest.TestCase):
    def test_swaps_coordinates_and_confidence_once_without_mutating(self):
        xy = np.arange(4 * 12 * 2).reshape(4, 12, 2).astype(float)
        cf = np.arange(4 * 12).reshape(4, 12).astype(float)
        original = xy.copy()
        out, confidence = penn_fixed_swap(xy, cf)
        np.testing.assert_array_equal(out, xy[:, SWAP])
        np.testing.assert_array_equal(confidence, cf[:, SWAP])
        np.testing.assert_array_equal(xy, original)
        np.testing.assert_array_equal(penn_fixed_swap(out, confidence)[0], xy)

    def test_rejects_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            penn_fixed_swap(np.zeros((4, 17, 2)), np.zeros((4, 17)))

    def test_calibration_uses_no_alignment_then_one_swap(self):
        rng = np.random.default_rng(42)
        xy = rng.uniform(30, 200, size=(8, 12, 2))
        cf = np.ones((8, 12))
        expected = FullCalibrationPipeline(lr_mode=None).calibrate_array(xy, cf)[:, SWAP]
        actual, actual_cf = penn_fixed_swap_calibrated(xy, cf)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(actual_cf, cf[:, SWAP])


if __name__ == "__main__":
    unittest.main()
