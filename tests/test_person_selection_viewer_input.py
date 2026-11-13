"""Check the viewer reconstructs the experiment's GT-assisted model input."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool/person_selection_viewer"))
sys.path.insert(0, str(ROOT / "src/gt_baselines"))
from compare_person_selection import resample  # noqa: E402
from person_selection_viewer import contributing_tokens, make_model_input  # noqa: E402


class ViewerInputTest(unittest.TestCase):
    def test_missing_selected_and_zero_coordinates(self):
        xy = np.ones((1, 17, 2), np.float32) * 50
        xy[0, 5] = 0
        data = {"offsets": np.array([0, 1, 1]), "xy": xy}
        rows = [{"gt_index": "0"}, {"gt_index": "-1"}]
        coords, norm, pos, actual = make_model_input(data, rows)
        self.assertTrue(np.array_equal(coords[0, 0], [0, 0]))
        self.assertTrue(np.isnan(coords[1]).all())
        self.assertTrue((norm[1] == 0).all())
        np.testing.assert_array_equal(actual, resample(coords))
        self.assertEqual(actual.shape, (64, 24))

    def test_token_influence_and_unselected_frames(self):
        positions = np.linspace(0, 99, 64)
        self.assertEqual(contributing_tokens(0, positions), [(0, 1.0)])
        self.assertEqual(contributing_tokens(1, positions)[0][0], 1)
        self.assertEqual(contributing_tokens(99, positions), [(63, 1.0)])
        self.assertEqual(contributing_tokens(100, positions), [])


if __name__ == "__main__":
    unittest.main()
