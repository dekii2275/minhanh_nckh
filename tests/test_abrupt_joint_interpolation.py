"""Regression tests for the short, high-confidence jump interpolation rule."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/gt_baselines"))
from evaluate_abrupt_joint_interpolation_20261113 import correct  # noqa: E402


def sample():
    pose = np.tile(np.array([[[10.0, 10.0]] * 6 + [[10.0, 30.0]] * 6], dtype=np.float32), (7, 1, 1))
    return pose, np.ones((7, 12), dtype=np.float32)


def test_spike_and_return_are_interpolated_without_touching_other_joints():
    pose, confidence = sample()
    pose[2, 4] = [100.0, 100.0]

    corrected, changed = correct(pose, confidence)

    assert np.flatnonzero(changed[:, 4]).tolist() == [2, 3]
    np.testing.assert_allclose(corrected[2:4, 4], [[10, 10], [10, 10]])
    np.testing.assert_array_equal(corrected[:, :4], pose[:, :4])
    assert not changed[:, :4].any()


def test_low_confidence_or_missing_points_are_not_treated_as_jumps():
    pose, confidence = sample()
    pose[2, 4] = [100.0, 100.0]
    confidence[2, 4] = 0.1
    corrected, changed = correct(pose, confidence)
    assert not changed.any()
    np.testing.assert_array_equal(corrected, pose)

    pose[2, 4] = np.nan
    corrected, changed = correct(pose, confidence)
    assert not changed.any()
    np.testing.assert_array_equal(corrected, pose)


def test_jump_at_end_is_not_extrapolated():
    pose, confidence = sample()
    pose[-1, 4] = [100.0, 100.0]
    corrected, changed = correct(pose, confidence)
    assert not changed.any()
    np.testing.assert_array_equal(corrected, pose)


def test_unreliable_or_far_anchors_leave_flagged_segment_unchanged():
    pose, confidence = sample()
    pose[2, 4] = [100.0, 100.0]
    confidence[4, 4] = 0.1
    corrected, changed = correct(pose, confidence)
    assert not changed.any()
    np.testing.assert_array_equal(corrected, pose)

    pose, confidence = sample()
    pose[2, 4] = [100.0, 100.0]
    pose[3:, 4] = [55.0, 10.0]
    corrected, changed = correct(pose, confidence)
    assert not changed.any()
    np.testing.assert_array_equal(corrected, pose)
