import numpy as np

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
            ("left_ankle", "right_ankle"),
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
