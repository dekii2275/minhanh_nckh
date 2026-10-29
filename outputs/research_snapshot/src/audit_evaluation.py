"""
Module: audit_evaluation.py
Purpose: Comprehensive audit of Task 7 evaluation on 15 frames evenly spaced across
         sequences 1101, 1665, and 1986.
         Generates detailed visual evidence, verifies resolutions/coordinates,
         analyzes left/right conventions, tests diagnostic swaps, and computes normalized errors.
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio


def run_audit(
    predictions_csv="outputs/predictions/yolo_predictions.csv",
    mapping_json="configs/joint_mapping.json",
    audit_fig_dir="outputs/figures/audit",
    audit_table_csv="outputs/metrics/audit_15_frames.csv",
):
    os.makedirs(audit_fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(audit_table_csv), exist_ok=True)

    df_preds = pd.read_csv(predictions_csv)
    with open(mapping_json, "r") as f:
        joint_pairs = json.load(f)["joint_pairs"]

    swap_coco = {
        "left_shoulder": "right_shoulder",
        "right_shoulder": "left_shoulder",
        "left_elbow": "right_elbow",
        "right_elbow": "left_elbow",
        "left_wrist": "right_wrist",
        "right_wrist": "left_wrist",
        "left_hip": "right_hip",
        "right_hip": "left_hip",
        "left_knee": "right_knee",
        "right_knee": "left_knee",
        "left_ankle": "right_ankle",
        "right_ankle": "left_ankle",
    }

    # Penn skeleton connections
    penn_edges = [
        (1, 2),
        (1, 3),
        (3, 5),
        (2, 4),
        (4, 6),
        (1, 7),
        (2, 8),
        (7, 8),
        (7, 9),
        (9, 11),
        (8, 10),
        (10, 12),
    ]

    # COCO skeleton connections (0-indexed indices for 12 matching body joints)
    coco_edges = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    # Select 5 frames per sequence
    audit_records = []

    for seq_id, g in df_preds.groupby("sequence_id"):
        g_sorted = g.sort_values(by="frame_index").reset_index(drop=True)
        N = len(g_sorted)
        idxs = [0, N // 4, N // 2, 3 * N // 4, N - 1]
        for idx in idxs:
            row = g_sorted.iloc[idx]
            audit_records.append(row)

    df_audit_sample = pd.DataFrame(audit_records).reset_index(drop=True)
    print(f"Auditing {len(df_audit_sample)} frames...")

    audit_summary_rows = []

    for idx, row in df_audit_sample.iterrows():
        seq_id = str(row["sequence_id"])
        action = row["action"]
        frame_idx = int(row["frame_index"])
        img_path = row["image_path"]
        mat_path = row["annotation_path"]

        img_bgr = cv2.imread(img_path)
        img_h, img_w = img_bgr.shape[:2]

        # Load mat
        mat = sio.loadmat(mat_path)
        mat_dims = mat["dimensions"].squeeze().tolist()  # [H, W, N]
        pose = str(mat["pose"][0])
        x_mat = np.array(mat["x"], dtype=float)
        y_mat = np.array(mat["y"], dtype=float)
        vis_mat = np.array(mat["visibility"], dtype=int)
        if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
            x_mat, y_mat, vis_mat = x_mat.T, y_mat.T, vis_mat.T

        f0 = frame_idx - 1
        x_gt = x_mat[f0]
        y_gt = y_mat[f0]
        vis_gt = vis_mat[f0]

        # Check resolution match
        res_match = img_h == mat_dims[0] and img_w == mat_dims[1]

        # Torso scale
        torso_s = None
        if vis_gt[1] == 1 and vis_gt[2] == 1 and vis_gt[7] == 1 and vis_gt[8] == 1:
            if x_gt[1] > 0 and x_gt[2] > 0 and x_gt[7] > 0 and x_gt[8] > 0:
                sx, sy = (x_gt[1] + x_gt[2]) / 2.0, (y_gt[1] + y_gt[2]) / 2.0
                hx, hy = (x_gt[7] + x_gt[8]) / 2.0, (y_gt[7] + y_gt[8]) / 2.0
                torso_s = float(np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2))

        # Person bounding box diagonal
        bx1, by1, bx2, by2 = row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]
        bbox_diag = (
            float(np.sqrt((bx2 - bx1) ** 2 + (by2 - by1) ** 2)) if not pd.isna(bx1) else None
        )

        # Compare direct vs swapped
        direct_errors = []
        swapped_errors = []
        joint_audit_details = []

        gt_pts = {}
        yolo_pts = {}

        for p in joint_pairs:
            p_idx = p["penn_index_0based"]
            p_name = p["penn_joint_name"]
            c_idx = p["coco_index"]
            c_name = p["coco_joint_name"]
            c_swap = swap_coco[c_name]

            if vis_gt[p_idx] == 1 and (x_gt[p_idx] > 0 or y_gt[p_idx] > 0):
                gx, gy = x_gt[p_idx], y_gt[p_idx]
                gt_pts[p_idx] = (gx, gy, p_name)

                px, py = row[f"pred_{c_name}_x"], row[f"pred_{c_name}_y"]
                px_s, py_s = row[f"pred_{c_swap}_x"], row[f"pred_{c_swap}_y"]

                if not pd.isna(px) and not pd.isna(py):
                    yolo_pts[c_idx] = (px, py, c_name)
                    d_err = float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
                    s_err = float(np.sqrt((px_s - gx) ** 2 + (py_s - gy) ** 2))
                    direct_errors.append(d_err)
                    swapped_errors.append(s_err)
                    joint_audit_details.append(
                        {"joint": p_name, "direct_err": d_err, "swapped_err": s_err}
                    )

        mean_d_err = float(np.mean(direct_errors)) if direct_errors else np.nan
        mean_s_err = float(np.mean(swapped_errors)) if swapped_errors else np.nan
        pck20_d = (np.array(direct_errors) <= 20.0).mean() * 100.0 if direct_errors else 0.0
        pck20_s = (np.array(swapped_errors) <= 20.0).mean() * 100.0 if swapped_errors else 0.0

        # Create dual-panel audit figure:
        # Left Panel: Direct comparison with labels & yellow error lines
        # Right Panel: Diagnostic Swapped comparison with cyan lines
        canvas_direct = img_bgr.copy()
        canvas_swap = img_bgr.copy()

        # Draw GT skeleton & labeled keypoints on both
        for c_img in [canvas_direct, canvas_swap]:
            # GT edges
            for j1, j2 in penn_edges:
                if j1 in gt_pts and j2 in gt_pts:
                    p1 = (int(round(gt_pts[j1][0])), int(round(gt_pts[j1][1])))
                    p2 = (int(round(gt_pts[j2][0])), int(round(gt_pts[j2][1])))
                    cv2.line(c_img, p1, p2, (0, 200, 0), 2, cv2.LINE_AA)
            # GT dots & text
            for j_idx, (gx, gy, name) in gt_pts.items():
                pt = (int(round(gx)), int(round(gy)))
                cv2.circle(c_img, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
                # Short name
                short_name = (
                    name.replace("shoulder", "sh")
                    .replace("elbow", "elb")
                    .replace("wrist", "wri")
                    .replace("knee", "kne")
                    .replace("ankle", "ank")
                )
                cv2.putText(
                    c_img,
                    f"GT:{short_name}",
                    (pt[0] + 4, pt[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Draw YOLO skeleton & dots on both
        for c_img in [canvas_direct, canvas_swap]:
            # YOLO edges
            for c1, c2 in coco_edges:
                if c1 in yolo_pts and c2 in yolo_pts:
                    p1 = (int(round(yolo_pts[c1][0])), int(round(yolo_pts[c1][1])))
                    p2 = (int(round(yolo_pts[c2][0])), int(round(yolo_pts[c2][1])))
                    cv2.line(c_img, p1, p2, (0, 0, 220), 2, cv2.LINE_AA)
            # YOLO dots & text
            for c_idx, (yx, yy, name) in yolo_pts.items():
                pt = (int(round(yx)), int(round(yy)))
                cv2.circle(c_img, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
                short_name = (
                    name.replace("shoulder", "sh")
                    .replace("elbow", "elb")
                    .replace("wrist", "wri")
                    .replace("knee", "kne")
                    .replace("ankle", "ank")
                )
                cv2.putText(
                    c_img,
                    f"Y:{short_name}",
                    (pt[0] + 4, pt[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        # On canvas_direct: Draw Yellow lines connecting DIRECT pairs
        for p in joint_pairs:
            p_idx = p["penn_index_0based"]
            c_idx = p["coco_index"]
            if p_idx in gt_pts and c_idx in yolo_pts:
                p_gt = (int(round(gt_pts[p_idx][0])), int(round(gt_pts[p_idx][1])))
                p_yolo = (int(round(yolo_pts[c_idx][0])), int(round(yolo_pts[c_idx][1])))
                cv2.line(canvas_direct, p_gt, p_yolo, (0, 255, 255), 2, cv2.LINE_AA)

        # On canvas_swap: Draw Cyan lines connecting SWAPPED pairs
        for p in joint_pairs:
            p_idx = p["penn_index_0based"]
            c_name = p["coco_joint_name"]
            c_swap_name = swap_coco[c_name]
            # find c_swap index
            c_swap_idx = [
                x["coco_index"] for x in joint_pairs if x["coco_joint_name"] == c_swap_name
            ][0]
            if p_idx in gt_pts and c_swap_idx in yolo_pts:
                p_gt = (int(round(gt_pts[p_idx][0])), int(round(gt_pts[p_idx][1])))
                p_swap = (int(round(yolo_pts[c_swap_idx][0])), int(round(yolo_pts[c_swap_idx][1])))
                cv2.line(canvas_swap, p_gt, p_swap, (255, 255, 0), 2, cv2.LINE_AA)

        # Header for each panel
        cv2.putText(
            canvas_direct,
            f"DIRECT MAPPING | Mean Err: {mean_d_err:.1f}px | PCK20: {pck20_d:.0f}%",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas_swap,
            f"DIAGNOSTIC SWAP | Mean Err: {mean_s_err:.1f}px | PCK20: {pck20_s:.0f}%",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        combined_view = np.hstack([canvas_direct, canvas_swap])

        diag_bar = np.zeros((36, combined_view.shape[1], 3), dtype=np.uint8)
        diag_bar[:] = (25, 25, 25)

        torso_str = f"{torso_s:.1f}px" if torso_s is not None else "N/A"
        bbox_str = f"{bbox_diag:.1f}px" if bbox_diag is not None else "N/A"
        bar_text = (
            f"Seq: {seq_id} | Action: {action} | Pose: {pose} | Frame: {frame_idx} | "
            f"Res: {img_w}x{img_h} | Torso: {torso_str} | BBox Diag: {bbox_str} "
            f"[GT: Green, YOLO: Red, Direct Err: Yellow, Swap Err: Cyan]"
        )
        cv2.putText(
            diag_bar,
            bar_text,
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        audit_fig = np.vstack([diag_bar, combined_view])
        out_fig_name = f"audit_seq{seq_id}_f{frame_idx:04d}.jpg"
        out_fig_path = os.path.join(audit_fig_dir, out_fig_name)
        cv2.imwrite(out_fig_path, audit_fig)

        audit_summary_rows.append(
            {
                "sequence_id": seq_id,
                "action": action,
                "pose": pose,
                "frame_index": frame_idx,
                "img_width": img_w,
                "img_height": img_h,
                "mat_dim_match": res_match,
                "torso_scale_px": torso_s if torso_s is not None else np.nan,
                "bbox_diag_px": bbox_diag if bbox_diag is not None else np.nan,
                "valid_gt_kpts": len(gt_pts),
                "direct_mean_err_px": mean_d_err,
                "direct_median_err_px": float(np.median(direct_errors))
                if direct_errors
                else np.nan,
                "direct_pck20": pck20_d,
                "swapped_mean_err_px": mean_s_err,
                "swapped_median_err_px": float(np.median(swapped_errors))
                if swapped_errors
                else np.nan,
                "swapped_pck20": pck20_s,
                "error_reduction_pct": ((mean_d_err - mean_s_err) / mean_d_err * 100.0)
                if mean_d_err > 0
                else 0.0,
                "norm_torso_direct": (mean_d_err / torso_s) if torso_s else np.nan,
                "norm_torso_swapped": (mean_s_err / torso_s) if torso_s else np.nan,
                "figure_path": out_fig_path,
            }
        )

    df_audit_table = pd.DataFrame(audit_summary_rows)
    df_audit_table.to_csv(audit_table_csv, index=False)
    print(f"Saved audit table of 15 frames to {audit_table_csv}")
    print(
        df_audit_table[
            [
                "sequence_id",
                "action",
                "frame_index",
                "direct_mean_err_px",
                "swapped_mean_err_px",
                "error_reduction_pct",
            ]
        ]
    )
    return df_audit_table


if __name__ == "__main__":
    run_audit()
