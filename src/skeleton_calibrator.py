"""
Module: skeleton_calibrator.py
Purpose: Modular 2D Skeleton Post-processing & Calibration Pipeline for Pose Estimation.
         Contains individual modules:
           - Module 1: One-Euro Filter (Temporal Motion Smoothing)
           - Module 2: Confidence Thresholding & Kinematic PCHIP Interpolation
           - Module 3: Anthropometric Bone-Length Constrained Projection
           - Module 4: Unsupervised Left-Right Trajectory Continuity
           - Module 5: Full Combined Calibration Pipeline
"""

import math
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# 17 COCO Keypoints
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 12 Paired Keypoints for Left-Right symmetry
SWAP_COCO = {
    'left_shoulder': 'right_shoulder', 'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow', 'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist', 'right_wrist': 'left_wrist',
    'left_hip': 'right_hip', 'right_hip': 'left_hip',
    'left_knee': 'right_knee', 'right_knee': 'left_knee',
    'left_ankle': 'right_ankle', 'right_ankle': 'left_ankle'
}

# Anatomical Bone Segments (Parent -> Child)
BONE_SEGMENTS = [
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle")
]


class LowPassFilter:
    """Standard 1st-order Low-Pass Filter."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.s = None

    def reset(self):
        self.s = None

    def filter(self, val, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if self.s is None or np.isnan(self.s):
            self.s = val
        else:
            self.s = self.alpha * val + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter1D:
    """
    1D One-Euro Filter (Casiez et al., CHI 2012).
    Adaptive low-pass filter that eliminates jitter at low speeds
    and minimizes lag at high speeds.
    """
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None
        self.last_x = None

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
        self.last_x = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, timestamp=None):
        if np.isnan(x):
            return x

        if self.last_time is None:
            dt = 1.0 / self.freq
        else:
            dt = (timestamp - self.last_time) if timestamp is not None else (1.0 / self.freq)
            if dt <= 1e-4:
                dt = 1.0 / self.freq

        self.last_time = timestamp

        # Estimate derivative
        if self.last_x is None:
            dx = 0.0
        else:
            dx = (x - self.last_x) / dt

        self.last_x = x

        # Filter derivative
        a_d = self._alpha(self.d_cutoff, dt)
        edx = self.dx_filter.filter(dx, a_d)

        # Adaptive cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff, dt)

        # Filter position
        return self.x_filter.filter(x, a)


# ==============================================================================
# MODULE 1: One-Euro Filter (Temporal Motion Smoothing)
# ==============================================================================
class OneEuroSmoothingCalibrator:
    """
    Applies independent One-Euro filters on all 2D keypoint coordinates
    along the temporal dimension for each sequence.
    """
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def calibrate_sequence(self, df_seq):
        """
        Calibrate keypoints for a single sequence DataFrame sorted by frame_index.
        Returns a new DataFrame with smoothed keypoints.
        """
        df_out = df_seq.copy()
        n_frames = len(df_out)
        if n_frames <= 2:
            return df_out

        for kpt in COCO_KEYPOINTS:
            col_x = f"pred_{kpt}_x"
            col_y = f"pred_{kpt}_y"
            col_conf = f"pred_{kpt}_conf"

            if col_x not in df_out.columns:
                continue

            filter_x = OneEuroFilter1D(self.freq, self.min_cutoff, self.beta, self.d_cutoff)
            filter_y = OneEuroFilter1D(self.freq, self.min_cutoff, self.beta, self.d_cutoff)

            xs = df_out[col_x].values.copy()
            ys = df_out[col_y].values.copy()
            smoothed_x = np.zeros(n_frames, dtype=float)
            smoothed_y = np.zeros(n_frames, dtype=float)

            for t in range(n_frames):
                smoothed_x[t] = filter_x.filter(xs[t], timestamp=t / self.freq)
                smoothed_y[t] = filter_y.filter(ys[t], timestamp=t / self.freq)

            df_out[col_x] = smoothed_x
            df_out[col_y] = smoothed_y

        return df_out


# ==============================================================================
# MODULE 2: Confidence Thresholding & Kinematic PCHIP Interpolation
# ==============================================================================
class KinematicInterpolationCalibrator:
    """
    Identifies low-confidence keypoints (conf < threshold) as occluded/missing,
    and reconstructs trajectories smoothly using Piecewise Cubic Hermite Interpolating
    Polynomial (PCHIP) along time to prevent overshoot.
    """
    def __init__(self, conf_threshold=0.35, max_gap=15):
        self.conf_threshold = conf_threshold
        self.max_gap = max_gap

    def calibrate_sequence(self, df_seq):
        df_out = df_seq.copy()
        n_frames = len(df_out)
        if n_frames <= 3:
            return df_out

        t_steps = np.arange(n_frames)

        for kpt in COCO_KEYPOINTS:
            col_x = f"pred_{kpt}_x"
            col_y = f"pred_{kpt}_y"
            col_conf = f"pred_{kpt}_conf"

            if col_x not in df_out.columns:
                continue

            xs = df_out[col_x].values.copy()
            ys = df_out[col_y].values.copy()
            confs = df_out[col_conf].values.copy()

            # Mark low-confidence or NaN as invalid
            invalid_mask = np.isnan(xs) | np.isnan(ys) | (confs < self.conf_threshold)
            valid_mask = ~invalid_mask

            # If not enough valid points to interpolate, keep as is
            if np.sum(valid_mask) < 2:
                continue

            # Check gap lengths
            valid_indices = t_steps[valid_mask]
            valid_xs = xs[valid_mask]
            valid_ys = ys[valid_mask]

            try:
                # PCHIP interpolation along temporal axis
                pchip_x = PchipInterpolator(valid_indices, valid_xs, extrapolate=False)
                pchip_y = PchipInterpolator(valid_indices, valid_ys, extrapolate=False)

                for idx in t_steps[invalid_mask]:
                    # Find closest valid points before and after
                    prev_valid = valid_indices[valid_indices < idx]
                    next_valid = valid_indices[valid_indices > idx]

                    if len(prev_valid) > 0 and len(next_valid) > 0:
                        gap_size = next_valid[0] - prev_valid[-1]
                        if gap_size <= self.max_gap:
                            xs[idx] = float(pchip_x(idx))
                            ys[idx] = float(pchip_y(idx))
                            confs[idx] = self.conf_threshold # restore baseline confidence
            except Exception:
                pass

            df_out[col_x] = xs
            df_out[col_y] = ys
            df_out[col_conf] = confs

        return df_out


# ==============================================================================
# MODULE 3: Anthropometric Bone-Length Constrained Projection
# ==============================================================================
class AnthropometricBoneConstraintCalibrator:
    """
    Constrains 2D bone lengths within biologically feasible ranges [min_ratio * L_ref, max_ratio * L_ref]
    around the sequence median reference length L_ref.
    Projects anomalous child joints back along the anatomical bone vector.
    """
    def __init__(self, min_ratio=0.40, max_ratio=1.20):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def calibrate_sequence(self, df_seq):
        df_out = df_seq.copy()
        n_frames = len(df_out)
        if n_frames <= 3:
            return df_out

        for parent_kpt, child_kpt in BONE_SEGMENTS:
            px_col, py_col = f"pred_{parent_kpt}_x", f"pred_{parent_kpt}_y"
            cx_col, cy_col = f"pred_{child_kpt}_x", f"pred_{child_kpt}_y"

            if px_col not in df_out.columns or cx_col not in df_out.columns:
                continue

            px = df_out[px_col].values.copy()
            py = df_out[py_col].values.copy()
            cx = df_out[cx_col].values.copy()
            cy = df_out[cy_col].values.copy()

            # Calculate bone lengths across sequence
            dx = cx - px
            dy = cy - py
            lengths = np.sqrt(dx**2 + dy**2)

            valid_lengths = lengths[~np.isnan(lengths) & (lengths > 1e-2)]
            if len(valid_lengths) < 3:
                continue

            l_ref = float(np.median(valid_lengths))
            min_len = self.min_ratio * l_ref
            max_len = self.max_ratio * l_ref

            # Project child joint if length is out of bound
            for t in range(n_frames):
                cur_len = lengths[t]
                if np.isnan(cur_len) or cur_len < 1e-2:
                    continue

                if cur_len > max_len or cur_len < min_len:
                    target_len = np.clip(cur_len, min_len, max_len)
                    unit_x = dx[t] / cur_len
                    unit_y = dy[t] / cur_len
                    cx[t] = px[t] + target_len * unit_x
                    cy[t] = py[t] + target_len * unit_y

            df_out[cx_col] = cx
            df_out[cy_col] = cy

        return df_out


# ==============================================================================
# MODULE 4: Unsupervised Viewpoint / Reference Frame Alignment
# ==============================================================================
class LeftRightContinuityCalibrator:
    """
    Self-supervised Reference Frame Alignment:
    In computer vision and datasets like Penn Action, annotators often label
    Left/Right according to Viewer-Centric perspective (Image-Left is Left).
    Meanwhile, COCO/YOLO models follow Subject-Centric anatomical definitions
    (where facing front places the subject's left arm on Image-Right: x_left > x_right).

    This module detects sequence-level body orientation in 2D space:
    If median(x_left_shoulder) > median(x_right_shoulder), the subject is facing front
    under viewer perspective. The module harmonizes the keypoint tracks to
    Viewer-Centric coordinate convention consistently across the sequence.
    """
    def __init__(self, mode="auto_viewer_centric"):
        self.mode = mode

    def calibrate_sequence(self, df_seq):
        df_out = df_seq.copy()
        n_frames = len(df_out)
        if n_frames <= 2:
            return df_out

        paired_joints = [
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle")
        ]

        lx_col, rx_col = "pred_left_shoulder_x", "pred_right_shoulder_x"
        if lx_col not in df_out.columns or rx_col not in df_out.columns:
            return df_out

        valid_mask = ~df_out[lx_col].isna() & ~df_out[rx_col].isna()
        if valid_mask.sum() < 2:
            return df_out

        # Compute median delta: x_left - x_right
        delta_x = df_out.loc[valid_mask, lx_col] - df_out.loc[valid_mask, rx_col]
        median_delta = float(np.median(delta_x))

        # If median_delta > 0, subject is facing front (subject's left is on image-right)
        # Harmonize to viewer-centric reference frame
        if median_delta > 0.0:
            for left_kpt, right_kpt in paired_joints:
                for suffix in ["x", "y", "conf"]:
                    col_l = f"pred_{left_kpt}_{suffix}"
                    col_r = f"pred_{right_kpt}_{suffix}"
                    if col_l in df_out.columns and col_r in df_out.columns:
                        tmp = df_out[col_l].values.copy()
                        df_out[col_l] = df_out[col_r].values.copy()
                        df_out[col_r] = tmp

        return df_out


# ==============================================================================
# MODULE 5: Full Combined Calibration Pipeline
# ==============================================================================
class FullCalibrationPipeline:
    """
    Chains all calibration stages in biologically and mathematically optimal order:
    1. Confidence Thresholding & Kinematic PCHIP Interpolation
    2. Anthropometric Bone-Length Constraint
    3. One-Euro Adaptive Temporal Smoothing
    4. Unsupervised Left-Right Continuity Harmonization
    """
    def __init__(self,
                 conf_threshold=0.35, max_gap=15,
                 min_bone_ratio=0.40, max_bone_ratio=1.20,
                 freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0,
                 lr_mode="auto_viewer_centric"):
        self.stage1_interp = KinematicInterpolationCalibrator(conf_threshold, max_gap)
        self.stage2_bone = AnthropometricBoneConstraintCalibrator(min_bone_ratio, max_bone_ratio)
        self.stage3_smooth = OneEuroSmoothingCalibrator(freq, min_cutoff, beta, d_cutoff)
        self.stage4_lr = LeftRightContinuityCalibrator(mode=lr_mode) if lr_mode else None

    def calibrate_sequence(self, df_seq):
        df_cur = df_seq.copy()
        df_cur = self.stage1_interp.calibrate_sequence(df_cur)
        df_cur = self.stage2_bone.calibrate_sequence(df_cur)
        df_cur = self.stage3_smooth.calibrate_sequence(df_cur)
        if self.stage4_lr is not None:
            df_cur = self.stage4_lr.calibrate_sequence(df_cur)
        return df_cur

    def calibrate_array(self, raw_kps, raw_confs):
        """
        Calibrate directly from numpy arrays:
          raw_kps: (T, 12, 2)
          raw_confs: (T, 12)
        Returns:
          calib_kps: (T, 12, 2)
        """
        T = raw_kps.shape[0]
        if T <= 2:
            return raw_kps.copy()

        kpt_list = [
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ] if raw_kps.shape[1] == 12 else COCO_KEYPOINTS

        import pandas as pd
        data = {"frame_index": np.arange(T)}
        for i, kpt in enumerate(kpt_list):
            data[f"pred_{kpt}_x"] = raw_kps[:, i, 0]
            data[f"pred_{kpt}_y"] = raw_kps[:, i, 1]
            data[f"pred_{kpt}_conf"] = raw_confs[:, i]

        df = pd.DataFrame(data)
        df_calib = self.calibrate_sequence(df)

        calib_kps = np.zeros_like(raw_kps)
        for i, kpt in enumerate(kpt_list):
            calib_kps[:, i, 0] = df_calib[f"pred_{kpt}_x"].values
            calib_kps[:, i, 1] = df_calib[f"pred_{kpt}_y"].values
        return calib_kps
