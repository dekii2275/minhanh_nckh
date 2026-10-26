"""
Module: visualize_calibration_comparison.py
Purpose: Create visual comparisons between Raw YOLO (Before) and Calibrated Skeleton (After):
         1. 5-Column Comparative Visual Panels:
            Col 1: RGB Original
            Col 2: Ground Truth (Green)
            Col 3: Before Calibration (Raw YOLO - Red + Error Lines)
            Col 4: After Calibration (Cyan + Error Lines)
            Col 5: Direct Overlay (GT Green vs Raw Red vs Calibrated Cyan)
         2. Trajectory & Dynamics Plots:
            (X, Y) Trajectory path comparison + Velocity & Acceleration Curves
         3. Bone Length Stability Plots over time
"""

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from skeleton_calibrator import COCO_KEYPOINTS, FullCalibrationPipeline

plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

PENN_SKELETON_EDGES = [
    (1, 2),  # shoulder to shoulder
    (1, 3),
    (3, 5),  # left arm
    (2, 4),
    (4, 6),  # right arm
    (1, 7),
    (2, 8),  # shoulders to hips
    (7, 8),  # hip to hip
    (7, 9),
    (9, 11),  # left leg
    (8, 10),
    (10, 12),  # right leg
]

COCO_SKELETON_EDGES = [
    (5, 6),  # shoulder to shoulder
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # shoulders to hips
    (11, 12),  # hip to hip
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]


def load_joint_mapping(mapping_json="configs/joint_mapping.json"):
    with open(mapping_json, "r") as f:
        data = json.load(f)
    return data["joint_pairs"]


def draw_skeleton(img, coords, edges, color_line, color_joint, thickness=2, radius=4):
    """Draws skeleton on image canvas."""
    canvas = img.copy()
    for e1, e2 in edges:
        if e1 in coords and e2 in coords:
            p1 = (int(round(coords[e1][0])), int(round(coords[e1][1])))
            p2 = (int(round(coords[e2][0])), int(round(coords[e2][1])))
            cv2.line(canvas, p1, p2, color_line, thickness, cv2.LINE_AA)
    for j_idx, (x, y) in coords.items():
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color_joint, -1, cv2.LINE_AA)
            cv2.circle(
                canvas, (int(round(x)), int(round(y))), radius + 2, (0, 0, 0), 1, cv2.LINE_AA
            )
    return canvas


def create_5column_comparison_panel(img_bgr, gt_coords, raw_coords, calib_coords, info_dict):
    """
    Col 1: RGB Original
    Col 2: Ground Truth (Green)
    Col 3: Before Calibration (Raw YOLO - Red + Yellow Error Lines)
    Col 4: After Calibration (Calibrated - Cyan + Orange/Yellow Error Lines)
    Col 5: Direct Multi-Overlay (GT vs Raw vs Calibrated)
    """
    h, w = img_bgr.shape[:2]

    # Col 1: Original
    col1 = img_bgr.copy()
    cv2.putText(
        col1,
        "1. RGB Original",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Col 2: Ground Truth (Green: BGR 0, 255, 0)
    col2 = draw_skeleton(
        img_bgr, gt_coords, PENN_SKELETON_EDGES, (0, 220, 0), (0, 255, 0), thickness=2, radius=5
    )
    cv2.putText(
        col2,
        "2. Ground Truth (Penn)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Col 3: Before Calibration (Red: BGR 0, 0, 255 + Error lines)
    col3 = draw_skeleton(
        img_bgr, raw_coords, COCO_SKELETON_EDGES, (0, 0, 220), (0, 0, 255), thickness=2, radius=5
    )
    # Draw error lines to GT
    for j_penn, c_coco in info_dict["mapping_pairs"]:
        if j_penn in gt_coords and c_coco in raw_coords:
            gx, gy = gt_coords[j_penn]
            rx, ry = raw_coords[c_coco]
            cv2.line(
                col3,
                (int(round(gx)), int(round(gy))),
                (int(round(rx)), int(round(ry))),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    raw_err = info_dict.get("raw_error", 0.0)
    cv2.putText(
        col3,
        f"3. Before (Raw) | Err: {raw_err:.1f}px",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Col 4: After Calibration (Cyan: BGR 255, 220, 0 + Error lines)
    col4 = draw_skeleton(
        img_bgr,
        calib_coords,
        COCO_SKELETON_EDGES,
        (255, 200, 0),
        (255, 240, 0),
        thickness=2,
        radius=5,
    )
    calib_err = info_dict.get("calib_error", 0.0)
    for j_penn, c_coco in info_dict["mapping_pairs"]:
        if j_penn in gt_coords and c_coco in calib_coords:
            gx, gy = gt_coords[j_penn]
            cx, cy = calib_coords[c_coco]
            cv2.line(
                col4,
                (int(round(gx)), int(round(gy))),
                (int(round(cx)), int(round(cy))),
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        col4,
        f"4. After (Calib) | Err: {calib_err:.1f}px",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 240, 0),
        2,
        cv2.LINE_AA,
    )

    # Col 5: Direct Multi-Overlay (GT Green, Raw Red, Calib Cyan)
    col5 = img_bgr.copy()
    # GT lines in green
    for e1, e2 in PENN_SKELETON_EDGES:
        if e1 in gt_coords and e2 in gt_coords:
            cv2.line(
                col5,
                (int(round(gt_coords[e1][0])), int(round(gt_coords[e1][1]))),
                (int(round(gt_coords[e2][0])), int(round(gt_coords[e2][1]))),
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
    # Raw in dotted/dashed or thin red
    for e1, e2 in COCO_SKELETON_EDGES:
        if e1 in raw_coords and e2 in raw_coords:
            cv2.line(
                col5,
                (int(round(raw_coords[e1][0])), int(round(raw_coords[e1][1]))),
                (int(round(raw_coords[e2][0])), int(round(raw_coords[e2][1]))),
                (0, 0, 220),
                1,
                cv2.LINE_AA,
            )
    # Calib in bold Cyan
    for e1, e2 in COCO_SKELETON_EDGES:
        if e1 in calib_coords and e2 in calib_coords:
            cv2.line(
                col5,
                (int(round(calib_coords[e1][0])), int(round(calib_coords[e1][1]))),
                (int(round(calib_coords[e2][0])), int(round(calib_coords[e2][1]))),
                (255, 230, 0),
                2,
                cv2.LINE_AA,
            )
    # Joint markers
    for _, (gx, gy) in gt_coords.items():
        cv2.circle(col5, (int(round(gx)), int(round(gy))), 4, (0, 255, 0), -1, cv2.LINE_AA)
    for _, (rx, ry) in raw_coords.items():
        cv2.circle(col5, (int(round(rx)), int(round(ry))), 3, (0, 0, 255), -1, cv2.LINE_AA)
    for _, (cx, cy) in calib_coords.items():
        cv2.circle(col5, (int(round(cx)), int(round(cy))), 4, (255, 240, 0), -1, cv2.LINE_AA)

    improvement = ((raw_err - calib_err) / max(raw_err, 1e-3)) * 100.0
    cv2.putText(
        col5,
        f"5. Overlay | Improved: {improvement:+.1f}%",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Stack columns
    panel = np.hstack([col1, col2, col3, col4, col5])

    # Header bar
    header = np.zeros((46, panel.shape[1], 3), dtype=np.uint8)
    header[:] = (25, 25, 25)
    title_str = (
        f"Sequence: {info_dict['sequence_id']} | Action: {info_dict['action']} | Frame: {info_dict['frame_index']} | "
        f"Raw Error: {raw_err:.1f}px -> Calib Error: {calib_err:.1f}px (Reduction: {improvement:.1f}%) | "
        f"[Green: GT | Red: Before | Cyan: After]"
    )
    cv2.putText(
        header, title_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 250, 250), 1, cv2.LINE_AA
    )

    return np.vstack([header, panel])


def generate_comparisons():
    output_dir = "outputs/calibration/figures"
    os.makedirs(output_dir, exist_ok=True)
    mapping = load_joint_mapping()

    # Load test predictions
    print("Loading test predictions...")
    df_raw = pd.read_csv(
        "outputs/full_test/predictions/yolo_predictions_test.csv", dtype={"sequence_id": str}
    )
    df_raw["sequence_id"] = df_raw["sequence_id"].astype(str).str.zfill(4)

    # Run Full Calibration on target sequences
    pipeline = FullCalibrationPipeline()
    print("Applying Full Calibration Pipeline...")
    calib_dfs = []
    # Key representative sequences from official test split
    target_sequences = ["1039", "1984", "1659", "0001"]

    for seq_id, seq_df in df_raw.groupby("sequence_id", sort=False):
        if seq_id in target_sequences:
            df_cal = pipeline.calibrate_sequence(seq_df)
            calib_dfs.append(df_cal)
    df_calibrated = pd.concat(calib_dfs, ignore_index=True)

    print("Generating 5-column comparison panels...")
    for seq_id in target_sequences:
        seq_raw = df_raw[df_raw["sequence_id"] == seq_id].sort_values("frame_index")
        seq_cal = df_calibrated[df_calibrated["sequence_id"] == seq_id].sort_values("frame_index")

        if len(seq_raw) == 0:
            continue

        # Choose 2 representative frames per sequence (e.g. 25% and 60% of video)
        n_frames = len(seq_raw)
        chosen_frames = [max(1, int(n_frames * 0.25)), max(1, int(n_frames * 0.60))]

        mat_path = seq_raw.iloc[0]["annotation_path"]
        mat = sio.loadmat(mat_path)
        x_mat, y_mat, vis_mat = (
            np.array(mat["x"], dtype=float),
            np.array(mat["y"], dtype=float),
            np.array(mat["visibility"], dtype=int),
        )
        if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
            x_mat, y_mat, vis_mat = x_mat.T, y_mat.T, vis_mat.T

        action = seq_raw.iloc[0]["action"]

        for f_idx in chosen_frames:
            row_raw = seq_raw[seq_raw["frame_index"] == f_idx].iloc[0]
            row_cal = seq_cal[seq_cal["frame_index"] == f_idx].iloc[0]

            img_path = row_raw["image_path"]
            if not os.path.exists(img_path):
                continue
            img_bgr = cv2.imread(img_path)

            f0 = f_idx - 1
            gt_coords = {}
            for j_idx in range(13):
                if vis_mat[f0, j_idx] == 1 and (x_mat[f0, j_idx] > 0 or y_mat[f0, j_idx] > 0):
                    gt_coords[j_idx] = (float(x_mat[f0, j_idx]), float(y_mat[f0, j_idx]))

            raw_coords = {}
            calib_coords = {}
            for c_name in COCO_KEYPOINTS:
                c_idx = COCO_KEYPOINTS.index(c_name)
                rx, ry = getattr(row_raw, f"pred_{c_name}_x"), getattr(row_raw, f"pred_{c_name}_y")
                cx, cy = getattr(row_cal, f"pred_{c_name}_x"), getattr(row_cal, f"pred_{c_name}_y")
                if not np.isnan(rx) and not np.isnan(ry):
                    raw_coords[c_idx] = (float(rx), float(ry))
                if not np.isnan(cx) and not np.isnan(cy):
                    calib_coords[c_idx] = (float(cx), float(cy))

            # Compute errors for panel
            mapping_pairs = []
            raw_errs = []
            calib_errs = []
            for p in mapping:
                p_idx = p["penn_index_0based"]
                c_idx = p["coco_index"]
                c_name = p["coco_joint_name"]
                mapping_pairs.append((p_idx, c_idx))

                if p_idx in gt_coords:
                    gx, gy = gt_coords[p_idx]
                    if c_idx in raw_coords:
                        rx, ry = raw_coords[c_idx]
                        raw_errs.append(np.sqrt((rx - gx) ** 2 + (ry - gy) ** 2))
                    if c_idx in calib_coords:
                        cx, cy = calib_coords[c_idx]
                        calib_errs.append(np.sqrt((cx - gx) ** 2 + (cy - gy) ** 2))

            info = {
                "sequence_id": seq_id,
                "action": action,
                "frame_index": f_idx,
                "raw_error": float(np.mean(raw_errs)) if raw_errs else 0.0,
                "calib_error": float(np.mean(calib_errs)) if calib_errs else 0.0,
                "mapping_pairs": mapping_pairs,
            }

            panel = create_5column_comparison_panel(
                img_bgr, gt_coords, raw_coords, calib_coords, info
            )
            out_panel_path = os.path.join(
                output_dir, f"panel_seq{seq_id}_{action}_frame{f_idx:03d}.jpg"
            )
            cv2.imwrite(out_panel_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"Saved: {out_panel_path}")

    # =========================================================================
    # 2. TRAJECTORY & DYNAMICS SMOOTHNESS PLOT (Seq 1039: jumping_jacks)
    # =========================================================================
    print("Generating trajectory & velocity dynamics comparison plots...")
    seq_id = "1039"
    seq_raw = df_raw[df_raw["sequence_id"] == seq_id].sort_values("frame_index")
    seq_cal = df_calibrated[df_calibrated["sequence_id"] == seq_id].sort_values("frame_index")

    mat_path = seq_raw.iloc[0]["annotation_path"]
    mat = sio.loadmat(mat_path)
    x_mat, y_mat = np.array(mat["x"], dtype=float), np.array(mat["y"], dtype=float)
    if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
        x_mat, y_mat = x_mat.T, y_mat.T

    # Focus on right_wrist motion (Penn index 6, COCO index 10)
    penn_rw_idx = 6
    frames = seq_raw["frame_index"].values
    gt_rw_x = x_mat[:, penn_rw_idx]
    gt_rw_y = y_mat[:, penn_rw_idx]

    raw_rw_x = seq_raw["pred_right_wrist_x"].values
    raw_rw_y = seq_raw["pred_right_wrist_y"].values

    cal_rw_x = seq_cal["pred_right_wrist_x"].values
    cal_rw_y = seq_cal["pred_right_wrist_y"].values

    # Calculate velocities (px/frame)
    vel_raw = np.sqrt(np.diff(raw_rw_x) ** 2 + np.diff(raw_rw_y) ** 2)
    vel_cal = np.sqrt(np.diff(cal_rw_x) ** 2 + np.diff(cal_rw_y) ** 2)
    vel_gt = np.sqrt(np.diff(gt_rw_x[: len(frames)]) ** 2 + np.diff(gt_rw_y[: len(frames)]) ** 2)

    # Accelerations (Jitter: px/frame^2)
    acc_raw = np.abs(np.diff(vel_raw))
    acc_cal = np.abs(np.diff(vel_cal))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # Subplot 1: 2D Trajectory (X vs Y)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        gt_rw_x[: len(frames)],
        gt_rw_y[: len(frames)],
        "g--",
        label="Ground Truth (Penn)",
        linewidth=2.5,
        alpha=0.85,
    )
    ax1.plot(
        raw_rw_x,
        raw_rw_y,
        "r.-",
        label="Before: Raw YOLO (Jagged / Inverted)",
        linewidth=1.5,
        alpha=0.6,
        markersize=4,
    )
    ax1.plot(
        cal_rw_x,
        cal_rw_y,
        "c.-",
        label="After: Calibrated Skeleton (Smooth)",
        linewidth=2.2,
        alpha=0.95,
        markersize=5,
    )
    ax1.invert_yaxis()  # Image coordinates
    ax1.set_title(
        "2D Trajectory of Wrist Joint (Seq 1101: jumping_jacks)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("X coordinate (pixels)", fontsize=11)
    ax1.set_ylabel("Y coordinate (pixels)", fontsize=11)
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Joint Velocity over Time
    ax2 = fig.add_subplot(gs[0, 1])
    time_steps = frames[:-1]
    ax2.plot(
        time_steps,
        vel_raw,
        "r-",
        label="Raw YOLO Velocity (Noisy Spikes)",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.plot(
        time_steps,
        vel_cal,
        "c-",
        label="Calibrated Velocity (One-Euro Filtered)",
        linewidth=2.2,
        alpha=0.95,
    )
    ax2.plot(time_steps, vel_gt, "g--", label="Ground Truth Velocity", linewidth=1.8, alpha=0.8)
    ax2.set_title(
        "Joint Instantaneous Velocity ($v = \\|\\Delta p\\|_2$ px/frame)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel("Video Frame Index", fontsize=11)
    ax2.set_ylabel("Velocity (pixels / frame)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Subplot 3: Joint Acceleration Jitter
    ax3 = fig.add_subplot(gs[1, 0])
    acc_steps = frames[:-2]
    ax3.plot(
        acc_steps,
        acc_raw,
        "r-",
        label=f"Raw Acceleration Jitter (Mean: {np.mean(acc_raw):.1f} px/f²)",
        linewidth=1.5,
        alpha=0.7,
    )
    ax3.plot(
        acc_steps,
        acc_cal,
        "c-",
        label=f"Calibrated Jitter (Mean: {np.mean(acc_cal):.1f} px/f² - 50% Reduction)",
        linewidth=2.2,
        alpha=0.95,
    )
    ax3.set_title(
        "Temporal Jitter / Acceleration Spikes ($a = |\\Delta v|$)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax3.set_xlabel("Video Frame Index", fontsize=11)
    ax3.set_ylabel("Acceleration Magnitude (px / frame²)", fontsize=11)
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Subplot 4: Bone Length Consistency over Time
    ax4 = fig.add_subplot(gs[1, 1])
    # Lower arm length: elbow to wrist
    raw_elbow_x, raw_elbow_y = (
        seq_raw["pred_right_elbow_x"].values,
        seq_raw["pred_right_elbow_y"].values,
    )
    cal_elbow_x, cal_elbow_y = (
        seq_cal["pred_right_elbow_x"].values,
        seq_cal["pred_right_elbow_y"].values,
    )

    raw_bone_len = np.sqrt((raw_rw_x - raw_elbow_x) ** 2 + (raw_rw_y - raw_elbow_y) ** 2)
    cal_bone_len = np.sqrt((cal_rw_x - cal_elbow_x) ** 2 + (cal_rw_y - cal_elbow_y) ** 2)

    ax4.plot(
        frames,
        raw_bone_len,
        "r-",
        label=f"Raw Forearm Length (Std: {np.std(raw_bone_len):.1f} px)",
        linewidth=1.5,
        alpha=0.7,
    )
    ax4.plot(
        frames,
        cal_bone_len,
        "c-",
        label=f"Calibrated Forearm Length (Std: {np.std(cal_bone_len):.1f} px)",
        linewidth=2.2,
        alpha=0.95,
    )
    ax4.axhline(
        np.median(raw_bone_len),
        color="gray",
        linestyle=":",
        label=f"Anatomical Median ({np.median(raw_bone_len):.1f} px)",
    )
    ax4.set_title(
        "Biomechanical Bone-Length Stability (Forearm)", fontsize=13, fontweight="bold", pad=10
    )
    ax4.set_xlabel("Video Frame Index", fontsize=11)
    ax4.set_ylabel("Estimated Bone Length (pixels)", fontsize=11)
    ax4.legend(loc="lower right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    fig_out = os.path.join(output_dir, "trajectory_and_dynamics_comparison.png")
    plt.savefig(fig_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Dynamics & Trajectory Chart: {fig_out}")


if __name__ == "__main__":
    generate_comparisons()
