"""Optional in-process cost-limited assignment fallback for Ultralytics trackers.

Uses SciPy when the optional `lap` package is unavailable.  The augmented
assignment gives each unmatched track/detection a half-cost-limit penalty.
Tie-breaking and numeric details can differ from LAPJV; this is for viewing,
not a bit-identical benchmark reproduction.
"""

import importlib.util
import sys
import types

import numpy as np
from scipy.optimize import linear_sum_assignment


def lapjv_fallback(cost, extend_cost=True, cost_limit=np.inf, **kwargs):
    if not extend_cost or kwargs:
        raise NotImplementedError("Fallback only supports extended cost-limited assignment")
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(cost_limit) or cost_limit < 0:
        raise ValueError("Expected a 2-D matrix and finite nonnegative cost_limit")
    n, m = matrix.shape
    x = np.full(n, -1, dtype=np.int32)
    y = np.full(m, -1, dtype=np.int32)
    if not n or not m:
        return 0.0, x, y
    finite = np.isfinite(matrix) & (matrix <= cost_limit)
    safe = max(
        1.0,
        abs(float(cost_limit)),
        float(np.max(np.abs(matrix[np.isfinite(matrix)]))) if np.isfinite(matrix).any() else 0.0,
    )
    big = safe * (n + m + 1) * 1000
    augmented = np.full((n + m, n + m), big, dtype=np.float64)
    augmented[:n, :m] = np.where(finite, matrix, big)
    augmented[np.arange(n), m + np.arange(n)] = cost_limit / 2
    augmented[n + np.arange(m), np.arange(m)] = cost_limit / 2
    augmented[n:, m:] = 0.0
    ii, jj = linear_sum_assignment(augmented)
    selected = (ii < n) & (jj < m)
    for i, j in zip(ii[selected], jj[selected]):
        if finite[i, j]:
            x[i] = j
            y[j] = i
    total = float(sum(matrix[i, j] for i, j in enumerate(x) if j >= 0))
    return total, x, y


def configure():
    """Use installed lap if available; otherwise supply a process-local shim."""
    if "lap" in sys.modules or importlib.util.find_spec("lap") is not None:
        return "lap"
    module = types.ModuleType("lap")
    module.__version__ = "scipy-viewer-fallback"
    module.lapjv = lapjv_fallback
    sys.modules["lap"] = module
    return "scipy-fallback"
