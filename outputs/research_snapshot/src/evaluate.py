"""
Module: evaluate.py
Purpose: Rigorous quantitative evaluation of YOLO11n-Pose predictions against Penn Action GT.
         Computes Euclidean pixel errors, PCK@20px, PCK-torso@0.10, PCK-torso@0.20,
         breakdowns per joint, sequence, and action, confidence analysis, coverage, and error cases.
"""

import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio


def load_joint_mapping(mapping_json="configs/joint_mapping.json"):
    with open(mapping_json, "r") as f:
        data = json.load(f)
    return data["joint_pairs"]


def compute_torso_scale(x_penn, y_penn, vis_penn):
    """
    Torso scale: distance from midpoint of shoulders to midpoint of hips in GT.
    Penn Action 0-indexed indices:
    1: left_shoulder, 2: right_shoulder, 7: left_hip, 8: right_hip
    """
    if vis_penn[1] == 1 and vis_penn[2] == 1 and vis_penn[7] == 1 and vis_penn[8] == 1:
        # Check non-zero
        if x_penn[1] > 0 and x_penn[2] > 0 and x_penn[7] > 0 and x_penn[8] > 0:
            mid_shoulder_x = (x_penn[1] + x_penn[2]) / 2.0
            mid_shoulder_y = (y_penn[1] + y_penn[2]) / 2.0
            mid_hip_x = (x_penn[7] + x_penn[8]) / 2.0
            mid_hip_y = (y_penn[7] + y_penn[8]) / 2.0
            scale = np.sqrt((mid_shoulder_x - mid_hip_x) ** 2 + (mid_shoulder_y - mid_hip_y) ** 2)
            if scale > 1e-3:
                return float(scale)
    return None


def evaluate_pipeline(
    predictions_csv="outputs/predictions/yolo_predictions.csv",
    mapping_json="configs/joint_mapping.json",
    metrics_dir="outputs/metrics",
):
    os.makedirs(metrics_dir, exist_ok=True)
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv}")

    df_preds = pd.read_csv(predictions_csv)
    joint_pairs = load_joint_mapping(mapping_json)

    # Cache loaded .mat annotations by path
    annotations_cache = {}

    per_keypoint_records = []

    total_frames = len(df_preds)
    frames_with_det = int(df_preds["has_detection"].sum())
    frames_without_det = total_frames - frames_with_det

    total_valid_gt_kpts = 0
    total_invalid_gt_kpts = 0
    total_predicted_kpts = 0
    total_missing_predictions = 0

    torso_scales = []
    invalid_torso_scale_count = 0

    for _, row in df_preds.iterrows():
        seq_id = str(row["sequence_id"])
        action = str(row["action"])
        frame_idx = int(row["frame_index"])
        mat_path = row["annotation_path"]
        has_det = bool(row["has_detection"])

        if mat_path not in annotations_cache:
            mat_data = sio.loadmat(mat_path)
            x_arr = np.array(mat_data["x"], dtype=float)
            y_arr = np.array(mat_data["y"], dtype=float)
            vis_arr = np.array(mat_data["visibility"], dtype=int)
            if x_arr.shape[1] != 13 and x_arr.shape[0] == 13:
                x_arr, y_arr, vis_arr = x_arr.T, y_arr.T, vis_arr.T
            annotations_cache[mat_path] = (x_arr, y_arr, vis_arr)

        x_arr, y_arr, vis_arr = annotations_cache[mat_path]
        f_idx_0 = frame_idx - 1  # 0-indexed frame row in mat

        x_gt = x_arr[f_idx_0]
        y_gt = y_arr[f_idx_0]
        vis_gt = vis_arr[f_idx_0]

        torso_s = compute_torso_scale(x_gt, y_gt, vis_gt)
        if torso_s is not None:
            torso_scales.append(torso_s)
        else:
            invalid_torso_scale_count += 1

        for pair in joint_pairs:
            p_idx = pair["penn_index_0based"]
            c_name = pair["coco_joint_name"]
            joint_name = pair["penn_joint_name"]

            # Check if GT is valid (visibility == 1 and non-zero coords)
            is_valid_gt = (vis_gt[p_idx] == 1) and (x_gt[p_idx] > 0 or y_gt[p_idx] > 0)

            if not is_valid_gt:
                total_invalid_gt_kpts += 1
                continue

            total_valid_gt_kpts += 1

            # Check if prediction exists
            pred_x_val = row[f"pred_{c_name}_x"]
            pred_y_val = row[f"pred_{c_name}_y"]
            pred_conf_val = row[f"pred_{c_name}_conf"]

            has_pred = has_det and (not pd.isna(pred_x_val)) and (not pd.isna(pred_y_val))

            if has_pred:
                total_predicted_kpts += 1
                err_px = float(
                    np.sqrt((pred_x_val - x_gt[p_idx]) ** 2 + (pred_y_val - y_gt[p_idx]) ** 2)
                )

                # PCK criteria
                pck20 = 1 if err_px <= 20.0 else 0
                pck_torso01 = 1 if (torso_s is not None and err_px <= 0.10 * torso_s) else 0
                pck_torso02 = 1 if (torso_s is not None and err_px <= 0.20 * torso_s) else 0

                rec = {
                    "sequence_id": seq_id,
                    "action": action,
                    "frame_index": frame_idx,
                    "joint_name": joint_name,
                    "gt_x": float(x_gt[p_idx]),
                    "gt_y": float(y_gt[p_idx]),
                    "pred_x": float(pred_x_val),
                    "pred_y": float(pred_y_val),
                    "pred_conf": float(pred_conf_val),
                    "error_px": err_px,
                    "has_prediction": True,
                    "torso_scale": torso_s if torso_s is not None else np.nan,
                    "pck20_correct": pck20,
                    "pck_torso01_correct": pck_torso01,
                    "pck_torso02_correct": pck_torso02,
                }
            else:
                total_missing_predictions += 1
                # Missing prediction counts as incorrect for PCK
                rec = {
                    "sequence_id": seq_id,
                    "action": action,
                    "frame_index": frame_idx,
                    "joint_name": joint_name,
                    "gt_x": float(x_gt[p_idx]),
                    "gt_y": float(y_gt[p_idx]),
                    "pred_x": np.nan,
                    "pred_y": np.nan,
                    "pred_conf": np.nan,
                    "error_px": np.nan,
                    "has_prediction": False,
                    "torso_scale": torso_s if torso_s is not None else np.nan,
                    "pck20_correct": 0,
                    "pck_torso01_correct": 0,
                    "pck_torso02_correct": 0,
                }
            per_keypoint_records.append(rec)

    df_kpts = pd.DataFrame(per_keypoint_records)
    kpts_csv_path = os.path.join(metrics_dir, "evaluation_per_keypoint.csv")
    df_kpts.to_csv(kpts_csv_path, index=False)
    print(f"Saved evaluated keypoints to {kpts_csv_path}")

    # Valid predicted pairs only for Euclidean distance statistics
    df_valid_pairs = df_kpts[df_kpts["has_prediction"]]
    errors = df_valid_pairs["error_px"].to_numpy()

    mean_err = float(np.mean(errors)) if len(errors) > 0 else np.nan
    median_err = float(np.median(errors)) if len(errors) > 0 else np.nan
    std_err = float(np.std(errors)) if len(errors) > 0 else np.nan
    p90_err = float(np.percentile(errors, 90)) if len(errors) > 0 else np.nan

    # PCK metrics: Denominator is ALL valid GT keypoints (missing predictions are incorrect)
    pck20_rate = (
        float(df_kpts["pck20_correct"].sum()) / total_valid_gt_kpts
        if total_valid_gt_kpts > 0
        else np.nan
    )
    pck_torso01_rate = (
        float(df_kpts["pck_torso01_correct"].sum()) / total_valid_gt_kpts
        if total_valid_gt_kpts > 0
        else np.nan
    )
    pck_torso02_rate = (
        float(df_kpts["pck_torso02_correct"].sum()) / total_valid_gt_kpts
        if total_valid_gt_kpts > 0
        else np.nan
    )

    coverage_rate = (
        float(total_predicted_kpts) / total_valid_gt_kpts if total_valid_gt_kpts > 0 else 0.0
    )

    # Grouped breakdowns
    by_joint_list = []
    for j_name, g in df_kpts.groupby("joint_name"):
        g_pred = g[g["has_prediction"]]
        by_joint_list.append(
            {
                "joint_name": j_name,
                "valid_gt_count": len(g),
                "predicted_count": len(g_pred),
                "missing_count": len(g) - len(g_pred),
                "mean_error_px": float(g_pred["error_px"].mean()) if len(g_pred) > 0 else np.nan,
                "median_error_px": float(g_pred["error_px"].median())
                if len(g_pred) > 0
                else np.nan,
                "std_error_px": float(g_pred["error_px"].std()) if len(g_pred) > 0 else np.nan,
                "p90_error_px": float(np.percentile(g_pred["error_px"], 90))
                if len(g_pred) > 0
                else np.nan,
                "pck20": float(g["pck20_correct"].sum()) / len(g) if len(g) > 0 else np.nan,
                "pck_torso01": float(g["pck_torso01_correct"].sum()) / len(g)
                if len(g) > 0
                else np.nan,
                "pck_torso02": float(g["pck_torso02_correct"].sum()) / len(g)
                if len(g) > 0
                else np.nan,
            }
        )
    df_by_joint = pd.DataFrame(by_joint_list).sort_values(by="mean_error_px", ascending=False)
    df_by_joint.to_csv(os.path.join(metrics_dir, "evaluation_by_joint.csv"), index=False)

    by_seq_list = []
    for (s_id, act), g in df_kpts.groupby(["sequence_id", "action"]):
        g_pred = g[g["has_prediction"]]
        by_seq_list.append(
            {
                "sequence_id": s_id,
                "action": act,
                "valid_gt_count": len(g),
                "predicted_count": len(g_pred),
                "missing_count": len(g) - len(g_pred),
                "mean_error_px": float(g_pred["error_px"].mean()) if len(g_pred) > 0 else np.nan,
                "median_error_px": float(g_pred["error_px"].median())
                if len(g_pred) > 0
                else np.nan,
                "pck20": float(g["pck20_correct"].sum()) / len(g) if len(g) > 0 else np.nan,
                "pck_torso01": float(g["pck_torso01_correct"].sum()) / len(g)
                if len(g) > 0
                else np.nan,
                "pck_torso02": float(g["pck_torso02_correct"].sum()) / len(g)
                if len(g) > 0
                else np.nan,
            }
        )
    df_by_seq = pd.DataFrame(by_seq_list)
    df_by_seq.to_csv(os.path.join(metrics_dir, "evaluation_by_sequence.csv"), index=False)

    # Confidence Bins Analysis
    # [0.0, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 0.9), [0.9, 1.0]
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
    conf_analysis = []
    for low, high in bins:
        sub = df_valid_pairs[
            (df_valid_pairs["pred_conf"] >= low) & (df_valid_pairs["pred_conf"] < high)
        ]
        c_count = len(sub)
        if c_count > 0:
            c_mean = float(sub["error_px"].mean())
            c_med = float(sub["error_px"].median())
            c_gt20 = float((sub["error_px"] > 20.0).sum()) / c_count * 100.0
        else:
            c_mean, c_med, c_gt20 = np.nan, np.nan, np.nan
        conf_analysis.append(
            {
                "bin": f"[{low:.1f}, {min(high, 1.0):.1f})",
                "count": c_count,
                "mean_error_px": c_mean,
                "median_error_px": c_med,
                "pct_error_gt_20px": c_gt20,
            }
        )

    # High confidence (>0.7) but large error (>20px) check
    high_conf_large_err = df_valid_pairs[
        (df_valid_pairs["pred_conf"] >= 0.7) & (df_valid_pairs["error_px"] > 20.0)
    ]
    high_conf_large_err_count = len(high_conf_large_err)

    # Identify joint with maximum error
    max_err_joint = df_by_joint.iloc[0]["joint_name"] if len(df_by_joint) > 0 else "N/A"
    max_err_val = df_by_joint.iloc[0]["mean_error_px"] if len(df_by_joint) > 0 else 0.0

    summary = {
        "total_sequences": int(df_preds["sequence_id"].nunique()),
        "total_frames": total_frames,
        "frames_with_detection": frames_with_det,
        "frames_without_detection": frames_without_det,
        "valid_gt_keypoints": total_valid_gt_kpts,
        "invalid_gt_keypoints": total_invalid_gt_kpts,
        "predicted_keypoints": total_predicted_kpts,
        "missing_predictions": total_missing_predictions,
        "coverage_rate": coverage_rate,
        "mean_error_px": mean_err,
        "median_error_px": median_err,
        "std_error_px": std_err,
        "p90_error_px": p90_err,
        "pck_20px": pck20_rate,
        "pck_torso_010": pck_torso01_rate,
        "pck_torso_020": pck_torso02_rate,
        "mean_torso_scale_px": float(np.mean(torso_scales)) if len(torso_scales) > 0 else np.nan,
        "invalid_torso_scale_frames": invalid_torso_scale_count,
        "max_error_joint": max_err_joint,
        "max_error_joint_val_px": max_err_val,
        "high_conf_large_error_count": high_conf_large_err_count,
        "confidence_analysis": conf_analysis,
    }

    summary_json_path = os.path.join(metrics_dir, "evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {summary_json_path}")

    print("\n" + "=" * 50)
    print("PILOT EVALUATION SUMMARY:")
    print(f"Sequences evaluated: {summary['total_sequences']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Valid GT keypoints: {summary['valid_gt_keypoints']}")
    print(f"Predicted keypoints: {summary['predicted_keypoints']}")
    print(f"Missing predictions: {summary['missing_predictions']}")
    print(f"Coverage rate: {summary['coverage_rate'] * 100:.1f}%")
    print(f"Mean error: {summary['mean_error_px']:.2f} px")
    print(f"Median error: {summary['median_error_px']:.2f} px")
    print(f"P90 error: {summary['p90_error_px']:.2f} px")
    print(f"PCK@20px: {summary['pck_20px'] * 100:.1f}%")
    print(f"PCK-torso@0.10: {summary['pck_torso_010'] * 100:.1f}%")
    print(f"PCK-torso@0.20: {summary['pck_torso_020'] * 100:.1f}%")
    print(
        f"Joint with highest error: {summary['max_error_joint']} ({summary['max_error_joint_val_px']:.2f} px)"
    )
    print(f"High confidence (>0.7) but error >20px cases: {summary['high_conf_large_error_count']}")
    print("=" * 50 + "\n")
    return summary


if __name__ == "__main__":
    evaluate_pipeline()
