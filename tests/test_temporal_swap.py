"""Tests for the four-neighbor isolated-swap diagnostic."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
from temporal_swap_analysis import SWAP, compare, temporal_decisions


class TemporalSwapTest(unittest.TestCase):
    def setUp(self):
        pose = np.array(
            [[40 if j % 2 == 0 else 100, 20 + (j // 2) * 20] for j in range(12)], np.float32
        )
        self.pose = np.repeat(pose[None, :, :], 5, axis=0)
        self.conf = np.ones((5, 12), np.float32)
        self.box = np.tile(np.array([0, 0, 160, 160], np.float32), (5, 1))

    def test_corrects_isolated_flip(self):
        self.pose[2] = self.pose[2, SWAP].copy()
        changed, reasons = temporal_decisions(self.pose, self.conf, self.box)
        self.assertEqual(np.flatnonzero(changed).tolist(), [2])
        self.assertEqual(reasons[2]["opposite_votes"], 4)

    def test_no_flips_for_consistent_or_globally_swapped_clip(self):
        for poses in (self.pose, self.pose[:, SWAP]):
            changed, _ = temporal_decisions(poses, self.conf, self.box)
            self.assertFalse(changed.any())

    def test_missing_neighbor_prevents_decision(self):
        self.pose[2] = self.pose[2, SWAP].copy()
        self.box[1] = np.nan
        changed, reasons = temporal_decisions(self.pose, self.conf, self.box)
        self.assertFalse(changed.any())
        self.assertEqual(reasons[2]["reason"], "neighbor_missing_or_ambiguous")

    def test_low_confidence_or_far_person_abstains(self):
        confidence = np.zeros(12, np.float32)
        self.assertIsNone(
            compare(self.pose[0], self.pose[0], confidence, confidence, self.box[0], self.box[0])
        )
        self.assertIsNone(
            compare(
                self.pose[0],
                self.pose[0],
                self.conf[0],
                self.conf[0],
                self.box[0],
                self.box[0] + 1000,
            )
        )


if __name__ == "__main__":
    unittest.main()
