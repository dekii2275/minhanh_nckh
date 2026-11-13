"""Verify frame-level visualization distinguishes filled and measured joints."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool/pose_correction_viewer"))
from pose_correction_viewer import frame_details, matches


class PoseCorrectionViewerTest(unittest.TestCase):
    def test_filled_frame_is_not_a_paired_comparison(self):
        before = np.full((1, 12, 2), np.nan, np.float32)
        after = np.ones((1, 12, 2), np.float32) * 10
        data = {
            "fixed_swap": before,
            "fixed_swap_conf": np.zeros((1, 12)),
            "interpolate3": after,
            "interpolate3_conf": np.ones((1, 12)),
            "filled": np.ones((1, 12), bool),
        }
        metrics = [
            {
                "gt": after[0].copy(),
                "gt_visible": np.ones(12, bool),
                "common": np.zeros(12, bool),
                "missing": True,
            }
        ]
        item = frame_details(data, metrics, 0, "interpolate3")
        self.assertIsNone(item["before"])
        self.assertIsNone(item["after"])
        self.assertEqual(item["after_visible_count"], 12)
        self.assertEqual(item["after_single_error"], 0)
        self.assertTrue(matches("filled", item, metrics[0]))
        self.assertFalse(matches("improved", item, metrics[0]))

    def test_moved_and_worsened_use_same_joint(self):
        before = np.ones((1, 12, 2), np.float32) * 10
        after = before.copy()
        after[0, 0, 0] += 2
        data = {
            "fixed_swap": before,
            "fixed_swap_conf": np.ones((1, 12)),
            "mean5": after,
            "mean5_conf": np.ones((1, 12)),
            "filled": np.ones((1, 12), bool),
        }
        common = np.zeros(12, bool)
        common[0] = True
        metrics = [
            {
                "gt": before[0].copy(),
                "gt_visible": np.ones(12, bool),
                "common": common,
                "missing": False,
            }
        ]
        item = frame_details(data, metrics, 0, "mean5")
        self.assertEqual(item["before"], 0)
        self.assertEqual(item["after"], 2)
        self.assertEqual(int(item["moved"].sum()), 1)
        self.assertFalse(item["filled"].any())
        self.assertTrue(matches("worsened", item, metrics[0]))
        self.assertFalse(matches("improved", item, metrics[0]))


if __name__ == "__main__":
    unittest.main()
