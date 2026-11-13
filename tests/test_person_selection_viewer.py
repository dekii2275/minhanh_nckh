"""Check the viewer against saved, real experiment decisions and edge cases."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool/person_selection_viewer"))
import person_selection_viewer as viewer  # noqa: E402

sys.path.insert(0, str(ROOT / "src/gt_baselines"))
from compare_person_selection import ious  # noqa: E402


class PersonSelectionViewerTests(unittest.TestCase):
    def test_missing_and_invalid_are_distinct(self):
        valid = np.array([0, 0, 10, 10])
        invalid = np.array([0, 0, 0, 0])
        self.assertEqual(viewer.category({"detections": "0"}, valid), "no_detection")
        self.assertEqual(viewer.category({"detections": "1"}, invalid), "invalid_gt")
        self.assertEqual(viewer.category({"detections": "1", "gt_index": "-1"}, valid), "no_match")
        self.assertEqual(
            viewer.category({"detections": "2", "gt_index": "1", "max_index": "0"}, valid),
            "changed",
        )

    def test_real_saved_frames_render_and_match_rules(self):
        required = [
            viewer.OUT / "videos_iou_0_3.csv",
            viewer.OUT / "frame_matches_iou_0_3.csv",
        ]
        for sid in ("0018", "0399"):
            required.extend(
                [
                    viewer.OUT / "detections" / f"{sid}.npz",
                    viewer.DATA / "labels" / f"{sid}.mat",
                    viewer.DATA / "frames" / sid,
                ]
            )
        if not all(path.exists() for path in required):
            self.skipTest("Penn Action and saved detection artifacts are not installed")
        videos = {r["id"]: r for r in viewer.read_csv(viewer.OUT / "videos_iou_0_3.csv")}
        frames = viewer.read_csv(viewer.OUT / "frame_matches_iou_0_3.csv")
        examples = [("0018", 13, "changed"), ("0018", 52, "no_match"), ("0399", 1, "no_detection")]
        for sid, frame, expected in examples:
            with self.subTest(sid=sid, frame=frame):
                rows = [r for r in frames if r["id"] == sid]
                seq, data, statuses = viewer.load_video(sid, videos[sid], rows, [])
                t = frame - 1
                self.assertEqual(statuses[t], expected)
                lo, hi = data["offsets"][t : t + 2]
                boxes = data["boxes"][lo:hi]
                scores = ious(boxes[:, :4], data["gt_bbox"][t])
                chosen = (
                    int(np.argmax(scores))
                    if len(scores) and np.isfinite(scores).all() and np.max(scores) >= 0.3
                    else -1
                )
                self.assertEqual(chosen, int(rows[t]["gt_index"]))
                image = viewer.render(seq, data, rows, statuses, t, "problem", True, 0.5)
                self.assertEqual(image.shape[1], 4 * seq.frame(t, False).shape[1])
                self.assertGreater(image.shape[0], seq.frame(t, False).shape[0])
                seq.close()


if __name__ == "__main__":
    unittest.main()
