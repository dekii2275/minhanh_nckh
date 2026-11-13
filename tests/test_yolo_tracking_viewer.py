"""Behavior checks for the tracking viewer and SciPy assignment fallback."""

import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool/yolo_tracking_viewer"))
import yolo_tracking_viewer as viewer
from assignment_fallback import lapjv_fallback


class TrackingViewerTest(unittest.TestCase):
    def test_assignment_respects_cost_limit_and_unmatched(self):
        _, x, y = lapjv_fallback(np.array([[0.1, 0.9], [0.8, 0.2], [0.9, 0.9]]), cost_limit=0.5)
        self.assertEqual(x.tolist(), [0, 1, -1])
        self.assertEqual(y.tolist(), [0, 1])
        _, x, y = lapjv_fallback(np.array([[0.51]]), cost_limit=0.5)
        self.assertEqual(x.tolist(), [-1])
        self.assertEqual(y.tolist(), [-1])

    def test_assignment_minimizes_cost_with_unmatched_penalty(self):
        rng = np.random.default_rng(7)
        limit = 0.5
        for n, m in ((1, 3), (3, 1), (2, 2), (2, 3), (3, 3)):
            for _ in range(10):
                cost = rng.random((n, m))
                _, x, y = lapjv_fallback(cost, cost_limit=limit)
                selected = [(i, int(j)) for i, j in enumerate(x) if j >= 0]
                self.assertTrue(all(cost[i, j] <= limit and y[j] == i for i, j in selected))
                actual = sum(cost[i, j] for i, j in selected) + limit / 2 * (
                    n + m - 2 * len(selected)
                )
                candidates = []
                for k in range(min(n, m) + 1):
                    for ii in itertools.combinations(range(n), k):
                        for jj in itertools.permutations(range(m), k):
                            if all(cost[i, j] <= limit for i, j in zip(ii, jj)):
                                candidates.append(
                                    sum(cost[i, j] for i, j in zip(ii, jj))
                                    + limit / 2 * (n + m - 2 * k)
                                )
                self.assertAlmostEqual(actual, min(candidates))

    def test_backward_navigation_does_not_advance_tracker(self):
        read = []
        updated = []
        source = SimpleNamespace(count=5, read=lambda i: (read.append(i), i)[1])

        def fake_track(model, tracker, image, conf, device):
            updated.append(image)
            return {"frame": image}

        with patch.object(viewer, "track_frame", side_effect=fake_track):
            cache = viewer.CachedPlayback(source, None, None, 0.1, "cpu")
            self.assertEqual(cache.get(3)["frame"], 3)
            self.assertEqual(cache.get(1)["frame"], 1)
            self.assertEqual(cache.get(4)["frame"], 4)
        self.assertEqual(read, [0, 1, 2, 3, 4])
        self.assertEqual(updated, [0, 1, 2, 3, 4])

    def test_click_selects_smallest_tracked_box(self):
        items = {
            "boxes": np.array([[0, 0, 100, 100], [20, 20, 40, 40], [15, 15, 25, 25]]),
            "ids": np.array([5, 8, -1]),
        }
        self.assertEqual(viewer.target_at(items, 22, 22), 8)
        self.assertEqual(viewer.target_at(items, 80, 80), 5)
        self.assertIsNone(viewer.target_at(items, 120, 120))

    def test_click_keeps_selected_id_and_reports_when_missing(self):
        image = np.zeros((100, 100, 3), np.uint8)
        closed = []
        source = SimpleNamespace(
            count=2,
            fps=15,
            label=None,
            read=lambda _: image.copy(),
            ground_truth=lambda _: np.full_like(image, 255),
            close=lambda: closed.append(True),
        )

        def fake_track(model, tracker, frame, conf, device):
            # The displayed GT image is white; the tracker must see raw black frames.
            self.assertEqual(int(frame.sum()), 0)
            index = len(calls)
            calls.append(index)
            if index == 0:
                return {
                    "boxes": np.array([[10, 10, 40, 70]]),
                    "scores": np.array([0.9]),
                    "xy": np.zeros((1, 17, 2)),
                    "conf": np.zeros((1, 17)),
                    "ids": np.array([4]),
                }
            return {
                "boxes": np.empty((0, 4)),
                "scores": np.empty(0),
                "xy": np.empty((0, 17, 2)),
                "conf": np.empty((0, 17)),
                "ids": np.empty(0, dtype=int),
            }

        calls = []
        callback = {}
        selections = []
        keys = iter(["click", "next", "quit"])
        original_render = viewer.render

        def render_spy(*args, **kwargs):
            selections.append(args[4])
            return original_render(*args, **kwargs)

        def wait(_):
            key = next(keys)
            if key == "click":
                callback["fn"](viewer.cv2.EVENT_LBUTTONDOWN, 120, 116, 0, None)
                return 255
            return ord("d") if key == "next" else ord("q")

        def set_mouse(_, fn):
            callback["fn"] = fn

        with (
            patch.object(viewer, "make_tracker", return_value=object()),
            patch.object(viewer, "track_frame", side_effect=fake_track),
            patch.object(viewer, "render", side_effect=render_spy),
            patch.object(viewer.cv2, "namedWindow"),
            patch.object(viewer.cv2, "resizeWindow"),
            patch.object(viewer.cv2, "setMouseCallback", side_effect=set_mouse),
            patch.object(viewer.cv2, "imshow"),
            patch.object(viewer.cv2, "waitKey", side_effect=wait),
            patch.object(viewer.cv2, "getWindowProperty", return_value=1),
            patch.object(viewer.cv2, "destroyWindow"),
        ):
            viewer.play(source, None, "bytetrack", 0.1, "cpu", 0.5, 1, None)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(selections, [None, 4, 4])
        self.assertEqual(closed, [True])

    def test_empty_yolo_frame_is_accepted_by_actual_tracker(self):
        import torch
        from ultralytics.engine.results import Boxes

        empty = Boxes(torch.empty((0, 6)), (20, 20))
        model = SimpleNamespace(predict=lambda **kw: [SimpleNamespace(boxes=empty, keypoints=None)])
        tracker = viewer.make_tracker("bytetrack")
        result = viewer.track_frame(model, tracker, np.zeros((20, 20, 3), np.uint8), 0.1, "cpu")
        self.assertEqual(result["boxes"].shape, (0, 4))
        self.assertEqual(result["ids"].size, 0)


if __name__ == "__main__":
    unittest.main()
