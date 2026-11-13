"""Check variable temporal windows and original-rule equivalence."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
from temporal_swap_analysis import SWAP, temporal_decisions
from temporal_window_sensitivity import decisions_for_radii


class TemporalWindowTest(unittest.TestCase):
    def setUp(self):
        pose = np.array(
            [[40 if j % 2 == 0 else 100, 20 + (j // 2) * 20] for j in range(12)], np.float32
        )
        self.pose = np.repeat(pose[None, :, :], 25, axis=0)
        self.conf = np.ones((25, 12), np.float32)
        self.box = np.tile(np.array([0, 0, 160, 160], np.float32), (25, 1))

    def test_all_windows_correct_one_isolated_flip(self):
        self.pose[12] = self.pose[12, SWAP].copy()
        decision, _ = decisions_for_radii(self.pose, self.conf, self.box)
        for r in (2, 3, 5, 10):
            self.assertEqual(np.flatnonzero(decision[r]).tolist(), [12])
        old, _ = temporal_decisions(self.pose, self.conf, self.box)
        np.testing.assert_array_equal(decision[2], old)

    def test_globally_consistent_orientation_is_not_detectable(self):
        pose = self.pose[:, SWAP]
        decision, _ = decisions_for_radii(pose, self.conf, self.box)
        self.assertTrue(all(not x.any() for x in decision.values()))

    def test_missing_neighbor_prevents_inference(self):
        self.pose[12] = self.pose[12, SWAP].copy()
        self.box[11] = np.nan
        decision, reasons = decisions_for_radii(self.pose, self.conf, self.box)
        for r in (2, 3, 5, 10):
            self.assertFalse(decision[r][12])
            self.assertEqual(reasons[r][12], "neighbor_missing_or_ambiguous")


if __name__ == "__main__":
    unittest.main()
