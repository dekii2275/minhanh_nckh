import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool/yolo_tracking_viewer"))
from motion_selector import choose_most_moving, movement_scores, pair_motion


def frame(t, include_mover=True, confidence=1):
    ids = [1, 2] if include_mover else [1]
    boxes = np.tile(np.array([1.0, 1.0, 101.0, 201.0]), (len(ids), 1))
    points = np.tile(np.array([50.0, 80.0]), (len(ids), 17, 1))
    if include_mover:
        points[1, 9:11, 0] += 2 * t  # Only the wrists move.
    return {
        "ids": np.array(ids),
        "boxes": boxes,
        "xy": points,
        "conf": np.full((len(ids), 17), confidence),
    }


class MotionSelectorTest(unittest.TestCase):
    def test_partial_body_motion_and_minimum_history(self):
        cache = [frame(t) for t in range(8)]
        self.assertIsNone(choose_most_moving(movement_scores(cache, 4)))
        scores = movement_scores(cache, 5)
        self.assertGreater(scores[2]["score"], scores[1]["score"])
        self.assertEqual(choose_most_moving(scores), 2)

    def test_missing_frame_not_bridged_and_current_ids_only(self):
        cache = [frame(t, include_mover=t != 3) for t in range(7)]
        scores = movement_scores(cache, 6, min_pairs=5)
        self.assertEqual(scores[2]["pairs"], 4)
        self.assertIsNone(scores[2]["score"])
        self.assertNotIn(2, movement_scores(cache, 3))

    def test_low_confidence_and_future_frames(self):
        cache = [frame(t, confidence=0.1) for t in range(7)]
        self.assertIsNone(choose_most_moving(movement_scores(cache, 6)))
        valid = [frame(t) for t in range(10)]
        self.assertEqual(movement_scores(valid, 6), movement_scores(valid[:7], 6))

    def test_box_translation_and_tie_break(self):
        f = frame(0)
        prev = (f["boxes"][0], f["xy"][0], f["conf"][0])
        current = (f["boxes"][0] + [3, 0, 3, 0], f["xy"][0] + [3, 0], f["conf"][0])
        result = pair_motion(prev, current)
        self.assertGreater(result["translation"], 0)
        self.assertAlmostEqual(result["articulation"], 0)
        self.assertEqual(choose_most_moving({2: {"score": 0.1}, 1: {"score": 0.1}}), 1)


if __name__ == "__main__":
    unittest.main()
