"""Ensure the swap viewer compares the same labeled joints as the audit."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool/swap_error_viewer"))
from swap_error_viewer import event, exceptions, frame_metrics, validate_metrics


class SwapViewerTest(unittest.TestCase):
    def test_matching_swap_and_missing_frame(self):
        points = np.zeros((2, 13, 2), np.float64)
        points[:, 1:13, 0] = np.arange(12) * 10 + 10
        points[:, 1:13, 1] = 20
        seq = SimpleNamespace(points=points, visible=np.ones((2, 13), bool))
        xy = np.ones((1, 17, 2), np.float64)
        xy[0, 5:17] = points[0, 1:13]
        data = {
            "offsets": np.array([0, 1, 1]),
            "xy": xy,
            "conf": np.ones((1, 17)),
            "gt_bbox": np.array([[1, 1, 200, 100], [1, 1, 200, 100]]),
        }
        rows = [{"gt_index": "0", "max_index": "0"}, {"gt_index": "-1", "max_index": "-1"}]
        m = frame_metrics(seq, data, rows)
        self.assertEqual(m[0]["common_count"], 12)
        self.assertAlmostEqual(m[0]["identity_error"], 0)
        self.assertAlmostEqual(m[0]["swap_error"], 10)
        self.assertTrue(event("identity_better", m[0], 1))
        self.assertTrue(event("no_match", m[1], 1))
        self.assertIsNone(m[1]["identity_error"])
        validate_metrics(
            m,
            {
                "paired_joints": "12",
                "gt_assisted_paired_error_px": "0",
                "gt_assisted_swap_paired_error_px": "10",
            },
        )

    def test_exceptions_require_swap_error_higher(self):
        entries = [
            {
                "id": "a",
                "gt_assisted_paired_error_px": "3",
                "gt_assisted_swap_paired_error_px": "4",
            },
            {
                "id": "b",
                "gt_assisted_paired_error_px": "4",
                "gt_assisted_swap_paired_error_px": "3",
            },
            {"id": "c", "gt_assisted_paired_error_px": "", "gt_assisted_swap_paired_error_px": ""},
        ]
        self.assertEqual([r["id"] for r in exceptions(entries)], ["a"])


if __name__ == "__main__":
    unittest.main()
