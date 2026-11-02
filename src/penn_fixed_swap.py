"""Optional Penn Action adapter: fixed YOLO left/right swap, without GT.

Use once at the YOLO-to-Penn boundary. Existing checkpoints and other datasets
retain their original behaviour unless the caller explicitly uses this adapter.
"""

import numpy as np

from skeleton_calibrator import FullCalibrationPipeline

SWAP = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])


def penn_fixed_swap(xy, confidence):
    """Return swapped (T,12,2) coordinates and (T,12) confidences."""
    xy = np.asarray(xy)
    confidence = np.asarray(confidence)
    if xy.ndim != 3 or xy.shape[1:] != (12, 2) or confidence.shape != xy.shape[:2]:
        raise ValueError("Expected xy (T,12,2) and confidence (T,12)")
    return xy[:, SWAP, :].copy(), confidence[:, SWAP].copy()


def penn_fixed_swap_calibrated(xy, confidence):
    """Run stages 1–3, then swap once. No stage 4 heuristic or GT."""
    xy = np.asarray(xy)
    confidence = np.asarray(confidence)
    if xy.ndim != 3 or xy.shape[1:] != (12, 2) or confidence.shape != xy.shape[:2]:
        raise ValueError("Expected xy (T,12,2) and confidence (T,12)")
    pipeline = FullCalibrationPipeline(lr_mode=None)
    return penn_fixed_swap(pipeline.calibrate_array(xy, confidence), confidence)
