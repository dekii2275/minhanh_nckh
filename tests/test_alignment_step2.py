"""Protect the experimental controls, including the production-input swap choice."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_alignment_step2 import SWAP, as_df, calibration_inputs


def skeleton(t):
    xy = np.zeros((t, 17, 2), dtype=float)
    for j in range(17):
        xy[:, j] = [150 + j * 3, 80 + j * 11]
    return xy


@pytest.mark.parametrize(
    "delta,t,expected_swap", [(40, 12, True), (-40, 12, False), (40, 2, False)]
)
def test_raw_and_production_permutation_controls(delta, t, expected_swap):
    xy = skeleton(t)
    xy[:, 5, 0] = 200 + delta / 2
    xy[:, 6, 0] = 200 - delta / 2
    original = xy[:, 5:].astype(np.float32)
    inputs, swap, _ = calibration_inputs(as_df(xy, np.ones((t, 17))))
    assert swap == expected_swap
    np.testing.assert_array_equal(inputs["C1"], original)
    np.testing.assert_array_equal(inputs["C2"], original[:, SWAP])
    np.testing.assert_array_equal(inputs["C3"], original[:, SWAP] if swap else original)


def test_c3_uses_production_shoulders_instead_of_raw_shoulders():
    xy = skeleton(10)
    xy[:, 5, 0] = 100
    xy[:, 6, 0] = 200
    xy[2:8, 5, 0] = 300
    conf = np.ones((10, 17))
    conf[2:8, 5] = 0.0
    assert np.median(xy[:, 5, 0] - xy[:, 6, 0]) > 0
    inputs, swap, production_delta = calibration_inputs(as_df(xy, conf))
    assert production_delta < 0
    assert not swap
    # C3 must retain even the low-confidence raw outlier coordinates: it is alignment-only.
    np.testing.assert_array_equal(inputs["C3"], inputs["C1"])
    assert not np.allclose(inputs["C4"], inputs["C3"])
