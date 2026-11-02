"""Ablation must use saved alignment, irrespective of current shoulder orientation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_calibration_step3 import CONFIGS, SWAP, as_df, coords, transform
from skeleton_calibrator import LeftRightContinuityCalibrator


@pytest.mark.parametrize("swap", [False, True])
def test_no_configuration_calls_heuristic_or_changes_saved_decision(monkeypatch, swap):
    def forbidden(*args, **kwargs):
        raise AssertionError("Heuristic must never run during the ablation")

    monkeypatch.setattr(LeftRightContinuityCalibrator, "calibrate_sequence", forbidden)
    xy = np.zeros((8, 17, 2), float)
    for j in range(17):
        xy[:, j] = [100 + j * 3, 50 + j * 10]
    # Positive raw shoulder delta would normally trigger the heuristic, even when swap=False.
    xy[:, 5, 0] = 200
    xy[:, 6, 0] = 100
    raw = as_df(xy, np.ones((8, 17)))
    original = raw.copy(deep=True)
    for enabled in CONFIGS.values():
        off = transform(raw, enabled, False)
        result = transform(raw, enabled, swap)
        np.testing.assert_array_equal(result, off[:, SWAP] if swap else off)
        pd.testing.assert_frame_equal(raw, original)
    np.testing.assert_array_equal(
        transform(raw, CONFIGS["A"], swap), coords(raw)[:, SWAP] if swap else coords(raw)
    )
