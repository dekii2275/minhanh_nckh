"""
Module: visualize.py
Purpose: Create visual artifacts:
         1. 4-column comparison panels (RGB | GT | YOLO | GT+YOLO+Errors)
         2. Analytical charts (per-joint error bar chart, error distribution, confidence vs error)
         3. Dynamic video / GIF demo of a complete evaluated sequence.
"""

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

# Set matplotlib style
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"

# Skeleton definitions
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


def create_4column_panel(img_bgr, gt_coords, yolo_coords, info_dict):
    """
    Generate a 4-column panel:
    Col 1: Original RGB
    Col 2: RGB + GT (Green)
    Col 3: RGB + YOLO (Red)
    Col 4: RGB + GT + YOLO + Error Lines (Yellow)
    """
    h, w = img_bgr.shape[:2]

    # Col 1: Original RGB
    col1 = img_bgr.copy()
    cv2.putText(
        col1,
        "1. RGB Original",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Col 2: RGB + Penn Action GT (Green)
    col2 = img_bgr.copy()
    # Draw GT edges
    for j1, j2 in PENN_SKELETON_EDGES:
        if j1 in gt_coords and j2 in gt_coords:
            p1 = (int(round(gt_coords[j1][0])), int(round(gt_coords[j1][1])))
            p2 = (int(round(gt_coords[j2][0])), int(round(gt_coords[j2][1])))
            cv2.line(col2, p1, p2, (0, 220, 0), 2, cv2.LINE_AA)
    for j_idx, (gx, gy) in gt_coords.items():
        cv2.circle(col2, (int(round(gx)), int(round(gy))), 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(col2, (int(round(gx)), int(round(gy))), 7, (0, 100, 0), 1, cv2.LINE_AA)
    cv2.putText(
        col2,
        "2. Ground Truth (Penn)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Col 3: RGB + YOLO Pose (Red)
    col3 = img_bgr.copy()
    if yolo_coords:
        for c1, c2 in COCO_SKELETON_EDGES:
            if c1 in yolo_coords and c2 in yolo_coords:
                p1 = (int(round(yolo_coords[c1][0])), int(round(yolo_coords[c1][1])))
                p2 = (int(round(yolo_coords[c2][0])), int(round(yolo_coords[c2][1])))
                cv2.line(col3, p1, p2, (0, 0, 220), 2, cv2.LINE_AA)
        for c_idx, (yx, yy) in yolo_coords.items():
            cv2.circle(col3, (int(round(yx)), int(round(yy))), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(col3, (int(round(yx)), int(round(yy))), 7, (0, 0, 150), 1, cv2.LINE_AA)
    cv2.putText(
        col3,
        "3. YOLO11n-Pose",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Col 4: RGB + GT + YOLO + Error Lines (Yellow)
    col4 = img_bgr.copy()
    # Draw GT in green
    for j1, j2 in PENN_SKELETON_EDGES:
        if j1 in gt_coords and j2 in gt_coords:
            p1 = (int(round(gt_coords[j1][0])), int(round(gt_coords[j1][1])))
            p2 = (int(round(gt_coords[j2][0])), int(round(gt_coords[j2][1])))
            cv2.line(col4, p1, p2, (0, 180, 0), 1, cv2.LINE_AA)
    for j_idx, (gx, gy) in gt_coords.items():
        cv2.circle(col4, (int(round(gx)), int(round(gy))), 4, (0, 255, 0), -1, cv2.LINE_AA)

    # Draw YOLO in red
    if yolo_coords:
        for c1, c2 in COCO_SKELETON_EDGES:
            if c1 in yolo_coords and c2 in yolo_coords:
                p1 = (int(round(yolo_coords[c1][0])), int(round(yolo_coords[c1][1])))
                p2 = (int(round(yolo_coords[c2][0])), int(round(yolo_coords[c2][1])))
                cv2.line(col4, p1, p2, (0, 0, 180), 1, cv2.LINE_AA)
        for c_idx, (yx, yy) in yolo_coords.items():
            cv2.circle(col4, (int(round(yx)), int(round(yy))), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # Draw Error Lines in yellow (BGR: 0, 255, 255)
    if "matched_pairs" in info_dict:
        for (gx, gy), (yx, yy) in info_dict["matched_pairs"]:
            p_gt = (int(round(gx)), int(round(gy)))
            p_pred = (int(round(yx)), int(round(yy)))
            cv2.line(col4, p_gt, p_pred, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(
        col4,
        "4. GT + YOLO + Error",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Stack 4 columns horizontally
    panel = np.hstack([col1, col2, col3, col4])

    # Add header banner
    header = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    title_text = (
        f"Sequence: {info_dict['sequence_id']} | Action: {info_dict['action']} | "
        f"Frame: {info_dict['frame_index']} | Mean Frame Error: {info_dict['frame_mean_error']:.1f} px "
        f"[GT: Green | YOLO: Red | Error: Yellow]"
    )
    cv2.putText(
        header, title_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA
    )

    full_image = np.vstack([header, panel])
    return full_image


def generate_visualizations(
    predictions_csv="outputs/predictions/yolo_predictions.csv",
    kpts_csv="outputs/metrics/evaluation_per_keypoint.csv",
    by_joint_csv="outputs/metrics/evaluation_by_joint.csv",
    mapping_json="configs/joint_mapping.json",
    figures_dir="outputs/figures",
    demo_dir="outputs/demo",
):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)

    df_preds = pd.read_csv(predictions_csv)
    df_kpts = pd.read_csv(kpts_csv)
    df_by_joint = pd.read_csv(by_joint_csv)
    joint_pairs = load_joint_mapping(mapping_json)

    # --- 1. CHARTS ---
    # A. Per-joint Mean Error Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted_joint = df_by_joint.sort_values(by="mean_error_px", ascending=True)
    colors = [
        "#2b5c8f"
        if "shoulder" in j or "hip" in j
        else "#d95f02"
        if "wrist" in j or "ankle" in j
        else "#7570b3"
        for j in df_sorted_joint["joint_name"]
    ]
    bars = ax.barh(
        df_sorted_joint["joint_name"],
        df_sorted_joint["mean_error_px"],
        xerr=df_sorted_joint["std_error_px"],
        color=colors,
        alpha=0.85,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean Euclidean Error (pixels)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Mean Error per Body Joint (Penn Action vs YOLO11n-Pose)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f} px",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "chart_mean_error_per_joint.png")
    fig.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved: {chart1_path}")

    # B. Error Distribution Histogram & KDE
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_errors = df_kpts[df_kpts["has_prediction"]]["error_px"]
    n, bins, patches = ax.hist(
        valid_errors, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black", density=True
    )
    med = valid_errors.median()
    p90 = np.percentile(valid_errors, 90)
    ax.axvline(med, color="green", linestyle="--", linewidth=2, label=f"Median: {med:.1f} px")
    ax.axvline(
        p90, color="red", linestyle="--", linewidth=2, label=f"90th Percentile: {p90:.1f} px"
    )
    ax.axvline(20.0, color="orange", linestyle=":", linewidth=2, label="PCK@20px Threshold")
    ax.set_xlabel("Euclidean Error (pixels)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Keypoint Errors (Penn Action vs YOLO11n-Pose)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "chart_error_distribution.png")
    fig.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved: {chart2_path}")

    # C. Keypoint Confidence vs Error Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df_pred_kpts = df_kpts[df_kpts["has_prediction"]]
    ax.scatter(
        df_pred_kpts["pred_conf"],
        df_pred_kpts["error_px"],
        alpha=0.35,
        color="#4575b4",
        edgecolors="none",
        s=35,
    )
    # Highlight high confidence (>0.7) but large error (>20px)
    high_conf_bad = df_pred_kpts[
        (df_pred_kpts["pred_conf"] >= 0.7) & (df_pred_kpts["error_px"] > 20.0)
    ]
    ax.scatter(
        high_conf_bad["pred_conf"],
        high_conf_bad["error_px"],
        color="#d73027",
        alpha=0.7,
        s=45,
        label=f"Conf >= 0.7 & Err > 20px ({len(high_conf_bad)} kpts)",
    )
    ax.axhline(20.0, color="orange", linestyle="--", linewidth=1.5, label="Error = 20 px")
    ax.axvline(0.7, color="purple", linestyle=":", linewidth=1.5, label="Conf = 0.7")
    ax.set_xlabel("YOLO Keypoint Confidence", fontsize=12, fontweight="bold")
    ax.set_ylabel("Euclidean Error (pixels)", fontsize=12, fontweight="bold")
    ax.set_title("Keypoint Confidence vs Prediction Error", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "chart_confidence_vs_error.png")
    fig.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved: {chart3_path}")

    # --- 2. REPRESENTATIVE 4-COLUMN PANELS ---
    # Calculate per-frame mean error
    frame_errors = (
        df_kpts.groupby(["sequence_id", "action", "frame_index"])["error_px"]
        .agg(["mean", "count"])
        .reset_index()
    )
    frame_errors = frame_errors.sort_values(by="mean").reset_index(drop=True)
    N = len(frame_errors)

    # Pick 2 low, 2 medium, 2 high
    selected_indices = [
        int(N * 0.05),
        int(N * 0.15),  # Low
        int(N * 0.45),
        int(N * 0.55),  # Medium
        int(N * 0.85),
        int(N * 0.95),  # High
    ]
    labels_category = ["low_1", "low_2", "med_1", "med_2", "high_1", "high_2"]

    for cat, idx in zip(labels_category, selected_indices):
        target_row = frame_errors.iloc[idx]
        seq_id = str(target_row["sequence_id"])
        act = target_row["action"]
        f_idx = int(target_row["frame_index"])
        m_err = float(target_row["mean"])

        pred_frame = df_preds[
            (df_preds["sequence_id"].astype(str) == seq_id) & (df_preds["frame_index"] == f_idx)
        ].iloc[0]
        img = cv2.imread(pred_frame["image_path"])
        if img is None:
            continue

        # Load GT for this frame
        mat_data = sio.loadmat(pred_frame["annotation_path"])
        x_mat = np.array(mat_data["x"], dtype=float)
        y_mat = np.array(mat_data["y"], dtype=float)
        vis_mat = np.array(mat_data["visibility"], dtype=int)
        if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
            x_mat, y_mat, vis_mat = x_mat.T, y_mat.T, vis_mat.T

        f_row = f_idx - 1
        gt_coords = {}
        for j_i in range(13):
            if vis_mat[f_row, j_i] == 1:
                gt_coords[j_i] = (x_mat[f_row, j_i], y_mat[f_row, j_i])

        # YOLO coords
        yolo_coords = {}
        matched_pairs = []
        if pred_frame["has_detection"]:
            for p in joint_pairs:
                c_idx = p["coco_index"]
                c_name = p["coco_joint_name"]
                p_idx = p["penn_index_0based"]
                px = pred_frame[f"pred_{c_name}_x"]
                py = pred_frame[f"pred_{c_name}_y"]
                if not pd.isna(px) and not pd.isna(py):
                    yolo_coords[c_idx] = (px, py)
                    if p_idx in gt_coords:
                        matched_pairs.append((gt_coords[p_idx], (px, py)))

        info = {
            "sequence_id": seq_id,
            "action": act,
            "frame_index": f_idx,
            "frame_mean_error": m_err,
            "matched_pairs": matched_pairs,
        }
        panel_img = create_4column_panel(img, gt_coords, yolo_coords, info)
        out_panel_path = os.path.join(figures_dir, f"panel_error_{cat}.jpg")
        cv2.imwrite(out_panel_path, panel_img)
        print(f"Saved representative panel: {out_panel_path} (mean err: {m_err:.1f} px)")

    # --- 3. VIDEO / GIF DEMO ---
    # Create demo from the first sequence (50 frames)
    first_seq_id = str(df_preds["sequence_id"].iloc[0])
    seq_frames = df_preds[df_preds["sequence_id"].astype(str) == first_seq_id].sort_values(
        by="frame_index"
    )

    demo_panels = []
    for _, p_row in seq_frames.iterrows():
        f_idx = int(p_row["frame_index"])
        img = cv2.imread(p_row["image_path"])
        if img is None:
            continue

        # GT
        mat_data = sio.loadmat(p_row["annotation_path"])
        x_mat = np.array(mat_data["x"], dtype=float)
        y_mat = np.array(mat_data["y"], dtype=float)
        vis_mat = np.array(mat_data["visibility"], dtype=int)
        if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
            x_mat, y_mat, vis_mat = x_mat.T, y_mat.T, vis_mat.T
        f_row = f_idx - 1
        gt_coords = {
            j_i: (x_mat[f_row, j_i], y_mat[f_row, j_i])
            for j_i in range(13)
            if vis_mat[f_row, j_i] == 1
        }

        # YOLO
        yolo_coords = {}
        matched_pairs = []
        errs = []
        if p_row["has_detection"]:
            for p in joint_pairs:
                c_idx = p["coco_index"]
                c_name = p["coco_joint_name"]
                p_idx = p["penn_index_0based"]
                px = p_row[f"pred_{c_name}_x"]
                py = p_row[f"pred_{c_name}_y"]
                if not pd.isna(px) and not pd.isna(py):
                    yolo_coords[c_idx] = (px, py)
                    if p_idx in gt_coords:
                        matched_pairs.append((gt_coords[p_idx], (px, py)))
                        errs.append(
                            np.sqrt(
                                (px - gt_coords[p_idx][0]) ** 2 + (py - gt_coords[p_idx][1]) ** 2
                            )
                        )

        f_mean_err = float(np.mean(errs)) if errs else 0.0
        info = {
            "sequence_id": first_seq_id,
            "action": p_row["action"],
            "frame_index": f_idx,
            "frame_mean_error": f_mean_err,
            "matched_pairs": matched_pairs,
        }
        panel_img = create_4column_panel(img, gt_coords, yolo_coords, info)
        # Resize slightly for video/gif efficiency if needed
        demo_panels.append(panel_img)

    if demo_panels:
        # Write MP4
        demo_video_path = os.path.join(demo_dir, "pilot_demo.mp4")
        h, w = demo_panels[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(demo_video_path, fourcc, 6.0, (w, h))
        for p_img in demo_panels:
            out_video.write(p_img)
        out_video.release()
        print(f"Saved demo video to {demo_video_path}")

        # Also write GIF (downsampled to 800px width for fast viewing)
        import matplotlib.animation as animation

        demo_gif_path = os.path.join(demo_dir, "pilot_demo.gif")

        # Save every 2nd frame or all 50 frames to gif
        fig_gif, ax_gif = plt.subplots(figsize=(w / 150, h / 150), dpi=100)
        ax_gif.axis("off")
        fig_gif.subplots_adjust(left=0, right=1, bottom=0, top=1)

        gif_frames = []
        for p_img in demo_panels:
            rgb_panel = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
            im = ax_gif.imshow(rgb_panel, animated=True)
            gif_frames.append([im])

        ani = animation.ArtistAnimation(
            fig_gif, gif_frames, interval=150, blit=True, repeat_delay=1000
        )
        ani.save(demo_gif_path, writer="pillow")
        plt.close(fig_gif)
        print(f"Saved demo GIF to {demo_gif_path}")


if __name__ == "__main__":
    generate_visualizations()
