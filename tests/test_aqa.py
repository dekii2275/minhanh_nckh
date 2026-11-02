"""Unit tests for Movement Quality Profiler and AQA metrics."""

import numpy as np

from movement_profiling import MovementQualityProfiler


def test_profiler_smooth_sequence():
    profiler = MovementQualityProfiler(fps=30.0)
    T = 30
    # Create smooth synthetic sinusoidal motion
    t = np.linspace(0, 2 * np.pi, T)
    coords = np.zeros((T, 12, 2), dtype=np.float32)
    for j in range(12):
        coords[:, j, 0] = 100.0 + 20.0 * np.sin(t + j * 0.1)
        coords[:, j, 1] = 200.0 + 15.0 * np.cos(t + j * 0.1)

    report = profiler.profile_sequence(coords, action_name="squat")

    assert "total_score" in report
    assert 0.0 <= report["total_score"] <= 100.0
    assert "sub_scores" in report
    assert "stability_score" in report["sub_scores"]
    assert "smoothness_score" in report["sub_scores"]
    assert "anatomical_integrity_score" in report["sub_scores"]
    assert "bilateral_symmetry_score" in report["sub_scores"]
    assert len(report["formative_feedback"]) > 0


def test_profiler_compare_raw_vs_calibrated():
    profiler = MovementQualityProfiler(fps=30.0)
    T = 25
    coords_clean = np.zeros((T, 12, 2), dtype=np.float32)
    coords_clean[:, :, 0] = 50.0
    coords_clean[:, :, 1] = 100.0

    # Add artificial noise to raw
    coords_raw = coords_clean + np.random.normal(0, 15.0, coords_clean.shape).astype(np.float32)

    comp = profiler.compare_raw_vs_calibrated(coords_raw, coords_clean, action_name="squat")
    assert "calibration_improvements" in comp
    assert "total_aqa_score_gain" in comp["calibration_improvements"]
    assert comp["calibration_improvements"]["total_aqa_score_gain"] > 0
