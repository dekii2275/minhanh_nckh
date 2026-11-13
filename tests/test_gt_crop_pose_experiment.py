import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/gt_baselines"))
from gt_crop_pose_experiment import crop_rectangle


class CropRectangleTest(unittest.TestCase):
    def test_margin_and_image_bounds(self):
        self.assertEqual(crop_rectangle([20, 20, 60, 100], 100, 120), (10, 0, 70, 120))
        self.assertEqual(crop_rectangle([0, 0, 10, 20], 100, 100), (0, 0, 13, 25))

    def test_missing_invalid_and_outside_boxes(self):
        for box in ([0, 0, 0, 2], [1, 2, 0, 4], [np.nan, 0, 2, 3], [110, 1, 120, 10]):
            self.assertIsNone(crop_rectangle(box, 100, 100))


if __name__ == "__main__":
    unittest.main()
