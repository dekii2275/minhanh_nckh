"""Checks for GT-box matching and missing-frame preprocessing."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
from compare_person_selection import ious, resample


class PersonSelectionDiagnosticTest(unittest.TestCase):
    def test_box_iou_and_empty_detections(self):
        boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
        np.testing.assert_allclose(ious(boxes, np.array([0, 0, 10, 10])), [1, 1 / 3, 0], rtol=1e-6)
        self.assertEqual(ious(np.empty((0, 4)), np.array([0, 0, 10, 10])).shape, (0,))

    def test_invalid_gt_cannot_match(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        self.assertTrue(np.isnan(ious(boxes, np.array([0, 0, 0, 10]))).all())
        self.assertTrue(np.isnan(ious(boxes, np.array([np.nan, 0, 10, 10]))).all())

    def test_missing_sequence_remains_finite(self):
        features = resample(np.full((3, 12, 2), np.nan, dtype=np.float32))
        self.assertEqual(features.shape, (64, 24))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((features == 0).all())


if __name__ == "__main__":
    unittest.main()
