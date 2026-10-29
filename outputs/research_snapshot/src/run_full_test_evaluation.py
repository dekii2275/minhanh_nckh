"""
Module: run_full_test_evaluation.py
Purpose: Complete benchmark of YOLO11n-Pose on the OFFICIAL FULL TEST SPLIT of Penn Action:
         1,068 test sequences (train == -1) totaling 76,298 frames across all 15 actions.
"""

import glob
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from ultralytics import YOLO

# COCO keypoint names
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SWAP_COCO = {
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


def load_joint_mapping(mapping_json="configs/joint_mapping.json"):
    with open(mapping_json, "r") as f:
        data = json.load(f)
    return data["joint_pairs"]


def get_torso_scale(x_penn, y_penn, vis_penn):
    """Torso scale: distance from mid-shoulder to mid-hip in GT."""
    if vis_penn[1] == 1 and vis_penn[2] == 1 and vis_penn[7] == 1 and vis_penn[8] == 1:
        if x_penn[1] > 0 and x_penn[2] > 0 and x_penn[7] > 0 and x_penn[8] > 0:
            sx, sy = (x_penn[1] + x_penn[2]) / 2.0, (y_penn[1] + y_penn[2]) / 2.0
            hx, hy = (x_penn[7] + x_penn[8]) / 2.0, (y_penn[7] + y_penn[8]) / 2.0
            scale = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
            if scale > 1e-3:
                return float(scale)
    return None


def step1_prepare_test_frames(
    penn_dir="data/raw/Penn_Action", output_csv="data/processed/full_test_frames.csv"
):
    if os.path.exists(output_csv):
        print(f"Test frames CSV already exists at {output_csv}, loading...")
        df = pd.read_csv(output_csv)
        print(f"Total test frames: {len(df)} across {df['sequence_id'].nunique()} sequences.")
        return df

    print(f"Scanning labels in {penn_dir}/labels to identify official test split (train == -1)...")
    labels = sorted(glob.glob(os.path.join(penn_dir, "labels", "*.mat")))
    records = []

    for mat_path in tqdm(labels, desc="Scanning test sequences"):
        mat = sio.loadmat(mat_path)
        is_train = int(mat["train"][0, 0])
        if is_train == -1:  # Official Test Split
            seq_id = os.path.splitext(os.path.basename(mat_path))[0]
            action = str(mat["action"][0])
            pose = str(mat["pose"][0])
            nframes = int(mat["nframes"][0, 0])
            seq_dir = os.path.join(penn_dir, "frames", seq_id)

            for f_idx in range(1, nframes + 1):
                img_path = os.path.join(seq_dir, f"{f_idx:06d}.jpg")
                records.append(
                    {
                        "sequence_id": seq_id,
                        "action": action,
                        "pose": pose,
                        "frame_index": f_idx,
                        "image_path": img_path,
                        "annotation_path": mat_path,
                    }
                )

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(
        f"Saved {len(df)} test frames from {df['sequence_id'].nunique()} test sequences to {output_csv}"
    )
    return df


def step2_run_batch_inference(
    df_test,
    model_path="models/yolo11n-pose.pt",
    output_csv="outputs/full_test/predictions/yolo_predictions_test.csv",
    batch_size=64,
    imgsz=640,
    conf=0.25,
    device="cuda:0",
):
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 1024 * 1024:
        print(f"Inference output file already exists at {output_csv}. Loading...")
        return pd.read_csv(output_csv)

    print(f"Initializing YOLO model from {model_path} on {device}...")
    model = YOLO(model_path)
    # Warmup
    _ = model(df_test["image_path"].iloc[:5].tolist(), imgsz=imgsz, device=device, verbose=False)

    total_frames = len(df_test)
    print(f"Starting batch inference on {total_frames} frames (batch_size={batch_size})...")
    start_time = time.time()

    header_written = False
    buffer_records = []
    chunk_save_size = 5000

    for i in tqdm(range(0, total_frames, batch_size), desc="Full Test Inference"):
        batch_df = df_test.iloc[i : i + batch_size]
        img_paths = batch_df["image_path"].tolist()

        results = model(
            source=img_paths, imgsz=imgsz, conf=conf, device=device, batch=batch_size, verbose=False
        )

        for row_tuple, res in zip(batch_df.itertuples(index=False), results):
            boxes = res.boxes
            kpts = res.keypoints
            num_persons = len(boxes) if boxes is not None else 0

            rec = {
                "sequence_id": row_tuple.sequence_id,
                "action": row_tuple.action,
                "pose": row_tuple.pose,
                "frame_index": row_tuple.frame_index,
                "image_path": row_tuple.image_path,
                "annotation_path": row_tuple.annotation_path,
                "num_persons_detected": num_persons,
                "has_detection": (num_persons > 0),
            }

            if num_persons > 0:
                # Select highest confidence detection
                confs = boxes.conf.cpu().numpy()
                best_i = int(np.argmax(confs))
                b_xyxy = boxes.xyxy[best_i].cpu().numpy().tolist()
                k_xy = kpts.xy[best_i].cpu().numpy()
                k_conf = kpts.conf[best_i].cpu().numpy() if kpts.conf is not None else np.ones(17)

                rec["det_confidence"] = float(confs[best_i])
                rec["bbox_x1"] = float(b_xyxy[0])
                rec["bbox_y1"] = float(b_xyxy[1])
                rec["bbox_x2"] = float(b_xyxy[2])
                rec["bbox_y2"] = float(b_xyxy[3])

                for c_idx, c_name in enumerate(COCO_KEYPOINTS):
                    rec[f"pred_{c_name}_x"] = float(k_xy[c_idx, 0])
                    rec[f"pred_{c_name}_y"] = float(k_xy[c_idx, 1])
                    rec[f"pred_{c_name}_conf"] = float(k_conf[c_idx])
            else:
                rec["det_confidence"] = np.nan
                rec["bbox_x1"] = np.nan
                rec["bbox_y1"] = np.nan
                rec["bbox_x2"] = np.nan
                rec["bbox_y2"] = np.nan
                for c_idx, c_name in enumerate(COCO_KEYPOINTS):
                    rec[f"pred_{c_name}_x"] = np.nan
                    rec[f"pred_{c_name}_y"] = np.nan
                    rec[f"pred_{c_name}_conf"] = np.nan

            buffer_records.append(rec)

        # Periodic flush to disk
        if len(buffer_records) >= chunk_save_size:
            df_chunk = pd.DataFrame(buffer_records)
            mode = "w" if not header_written else "a"
            df_chunk.to_csv(output_csv, mode=mode, header=not header_written, index=False)
            header_written = True
            buffer_records = []

    if buffer_records:
        df_chunk = pd.DataFrame(buffer_records)
        mode = "w" if not header_written else "a"
        df_chunk.to_csv(output_csv, mode=mode, header=not header_written, index=False)

    total_time = time.time() - start_time
    fps = total_frames / total_time
    print(
        f"\nInference completed in {total_time:.1f}s ({total_time / 60:.2f} mins) at {fps:.1f} FPS."
    )
    print(f"Predictions saved to {output_csv}")
    return pd.read_csv(output_csv)


def step3_evaluate_full_test(
    predictions_csv="outputs/full_test/predictions/yolo_predictions_test.csv",
    mapping_json="configs/joint_mapping.json",
    metrics_dir="outputs/full_test/metrics",
):
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"Loading predictions from {predictions_csv}...")
    df_preds = pd.read_csv(predictions_csv)
    joint_pairs = load_joint_mapping(mapping_json)

    print("Caching .mat annotations for 1,068 test sequences...")
    annotations_cache = {}
    for mat_path in tqdm(df_preds["annotation_path"].unique(), desc="Caching annotations"):
        mat = sio.loadmat(mat_path)
        x_mat = np.array(mat["x"], dtype=float)
        y_mat = np.array(mat["y"], dtype=float)
        vis_mat = np.array(mat["visibility"], dtype=int)
        if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
            x_mat, y_mat, vis_mat = x_mat.T, y_mat.T, vis_mat.T
        annotations_cache[mat_path] = (x_mat, y_mat, vis_mat)

    print(f"Evaluating {len(df_preds)} frames across 12 joints...")
    total_valid_gt = 0
    total_missing_pred = 0

    direct_errors = []
    swapped_errors = []
    direct_pck20 = []
    swapped_pck20 = []
    direct_pck_t01 = []
    direct_pck_t02 = []
    swapped_pck_t01 = []
    swapped_pck_t02 = []

    # Per-joint lists
    by_joint_data = {
        p["penn_joint_name"]: {"direct": [], "swapped": [], "valid": 0, "missing": 0}
        for p in joint_pairs
    }
    # Per-action lists
    actions_list = df_preds["action"].unique().tolist()
    by_action_data = {
        act: {"direct": [], "swapped": [], "valid": 0, "missing": 0} for act in actions_list
    }
    # Per-pose lists
    poses_list = df_preds["pose"].unique().tolist()
    by_pose_data = {
        pose: {"direct": [], "swapped": [], "valid": 0, "missing": 0} for pose in poses_list
    }

    # Confidence bins
    conf_bins_keys = ["[0.0, 0.3)", "[0.3, 0.5)", "[0.5, 0.7)", "[0.7, 0.9)", "[0.9, 1.0]"]
    conf_bins_data = {k: {"direct": [], "swapped": [], "count": 0} for k in conf_bins_keys}

    torso_scales = []

    for row in tqdm(
        df_preds.itertuples(index=False), total=len(df_preds), desc="Calculating errors"
    ):
        mat_path = row.annotation_path
        f_idx = row.frame_index
        action = row.action
        pose = row.pose
        has_det = row.has_detection

        x_mat, y_mat, vis_mat = annotations_cache[mat_path]
        f0 = f_idx - 1

        x_gt = x_mat[f0]
        y_gt = y_mat[f0]
        vis_gt = vis_mat[f0]

        t_scale = get_torso_scale(x_gt, y_gt, vis_gt)
        if t_scale is not None:
            torso_scales.append(t_scale)

        for p in joint_pairs:
            p_idx = p["penn_index_0based"]
            p_name = p["penn_joint_name"]
            c_name = p["coco_joint_name"]
            c_swap_name = SWAP_COCO[c_name]

            if vis_gt[p_idx] == 1 and (x_gt[p_idx] > 0 or y_gt[p_idx] > 0):
                total_valid_gt += 1
                by_joint_data[p_name]["valid"] += 1
                by_action_data[action]["valid"] += 1
                by_pose_data[pose]["valid"] += 1

                gx, gy = x_gt[p_idx], y_gt[p_idx]
                px = getattr(row, f"pred_{c_name}_x")
                py = getattr(row, f"pred_{c_name}_y")
                pconf = getattr(row, f"pred_{c_name}_conf")

                px_s = getattr(row, f"pred_{c_swap_name}_x")
                py_s = getattr(row, f"pred_{c_swap_name}_y")

                if has_det and not np.isnan(px) and not np.isnan(py):
                    d_err = float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
                    s_err = float(np.sqrt((px_s - gx) ** 2 + (py_s - gy) ** 2))

                    direct_errors.append(d_err)
                    swapped_errors.append(s_err)

                    direct_pck20.append(1 if d_err <= 20.0 else 0)
                    swapped_pck20.append(1 if s_err <= 20.0 else 0)

                    if t_scale is not None:
                        direct_pck_t01.append(1 if d_err <= 0.10 * t_scale else 0)
                        direct_pck_t02.append(1 if d_err <= 0.20 * t_scale else 0)
                        swapped_pck_t01.append(1 if s_err <= 0.10 * t_scale else 0)
                        swapped_pck_t02.append(1 if s_err <= 0.20 * t_scale else 0)
                    else:
                        direct_pck_t01.append(0)
                        direct_pck_t02.append(0)
                        swapped_pck_t01.append(0)
                        swapped_pck_t02.append(0)

                    by_joint_data[p_name]["direct"].append(d_err)
                    by_joint_data[p_name]["swapped"].append(s_err)

                    by_action_data[action]["direct"].append(d_err)
                    by_action_data[action]["swapped"].append(s_err)

                    by_pose_data[pose]["direct"].append(d_err)
                    by_pose_data[pose]["swapped"].append(s_err)

                    # Confidence bin
                    if pconf < 0.3:
                        ck = "[0.0, 0.3)"
                    elif pconf < 0.5:
                        ck = "[0.3, 0.5)"
                    elif pconf < 0.7:
                        ck = "[0.5, 0.7)"
                    elif pconf < 0.9:
                        ck = "[0.7, 0.9)"
                    else:
                        ck = "[0.9, 1.0]"
                    conf_bins_data[ck]["count"] += 1
                    conf_bins_data[ck]["direct"].append(d_err)
                    conf_bins_data[ck]["swapped"].append(s_err)
                else:
                    total_missing_pred += 1
                    by_joint_data[p_name]["missing"] += 1
                    by_action_data[action]["missing"] += 1
                    by_pose_data[pose]["missing"] += 1

                    direct_pck20.append(0)
                    swapped_pck20.append(0)
                    direct_pck_t01.append(0)
                    direct_pck_t02.append(0)
                    swapped_pck_t01.append(0)
                    swapped_pck_t02.append(0)

    # Compile Overall Summary
    total_frames = len(df_preds)
    frames_with_det = int(df_preds["has_detection"].sum())

    summary = {
        "total_test_sequences": int(df_preds["sequence_id"].nunique()),
        "total_test_frames": total_frames,
        "frames_with_detection": frames_with_det,
        "frames_without_detection": total_frames - frames_with_det,
        "frame_detection_rate": float(frames_with_det) / total_frames,
        "total_valid_gt_keypoints": total_valid_gt,
        "total_missing_predictions": total_missing_pred,
        "keypoint_coverage_rate": float(total_valid_gt - total_missing_pred) / total_valid_gt,
        "mean_torso_scale_px": float(np.mean(torso_scales)) if torso_scales else np.nan,
        # DIRECT NOMINAL MAPPING (Raw Penn GT vs Raw COCO)
        "direct_mapping": {
            "mean_error_px": float(np.mean(direct_errors)),
            "median_error_px": float(np.median(direct_errors)),
            "std_error_px": float(np.std(direct_errors)),
            "p90_error_px": float(np.percentile(direct_errors, 90)),
            "pck_20px": float(np.mean(direct_pck20)),
            "pck_torso_010": float(np.mean(direct_pck_t01)),
            "pck_torso_020": float(np.mean(direct_pck_t02)),
        },
        # DIAGNOSTIC SWAPPED MAPPING (Anatomically-aligned Left/Right)
        "diagnostic_swapped_mapping": {
            "mean_error_px": float(np.mean(swapped_errors)),
            "median_error_px": float(np.median(swapped_errors)),
            "std_error_px": float(np.std(swapped_errors)),
            "p90_error_px": float(np.percentile(swapped_errors, 90)),
            "pck_20px": float(np.mean(swapped_pck20)),
            "pck_torso_010": float(np.mean(swapped_pck_t01)),
            "pck_torso_020": float(np.mean(swapped_pck_t02)),
        },
    }

    with open(os.path.join(metrics_dir, "full_test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Compile Table By Action
    by_act_rows = []
    for act, data in by_action_data.items():
        d_errs = data["direct"]
        s_errs = data["swapped"]
        val = data["valid"]
        by_act_rows.append(
            {
                "action": act,
                "valid_gt_kpts": val,
                "direct_mean_err_px": float(np.mean(d_errs)) if d_errs else np.nan,
                "direct_median_err_px": float(np.median(d_errs)) if d_errs else np.nan,
                "direct_pck20": float((np.array(d_errs) <= 20.0).sum()) / val if val else 0.0,
                "swapped_mean_err_px": float(np.mean(s_errs)) if s_errs else np.nan,
                "swapped_median_err_px": float(np.median(s_errs)) if s_errs else np.nan,
                "swapped_pck20": float((np.array(s_errs) <= 20.0).sum()) / val if val else 0.0,
                "error_reduction_pct": float(
                    (np.mean(d_errs) - np.mean(s_errs)) / np.mean(d_errs) * 100.0
                )
                if d_errs
                else 0.0,
            }
        )
    df_by_act = pd.DataFrame(by_act_rows).sort_values(by="valid_gt_kpts", ascending=False)
    df_by_act.to_csv(os.path.join(metrics_dir, "evaluation_by_action.csv"), index=False)

    # Compile Table By Joint
    by_jnt_rows = []
    for j_name, data in by_joint_data.items():
        d_errs = data["direct"]
        s_errs = data["swapped"]
        val = data["valid"]
        by_jnt_rows.append(
            {
                "joint_name": j_name,
                "valid_gt_kpts": val,
                "direct_mean_err_px": float(np.mean(d_errs)) if d_errs else np.nan,
                "direct_median_err_px": float(np.median(d_errs)) if d_errs else np.nan,
                "direct_pck20": float((np.array(d_errs) <= 20.0).sum()) / val if val else 0.0,
                "swapped_mean_err_px": float(np.mean(s_errs)) if s_errs else np.nan,
                "swapped_median_err_px": float(np.median(s_errs)) if s_errs else np.nan,
                "swapped_pck20": float((np.array(s_errs) <= 20.0).sum()) / val if val else 0.0,
                "error_reduction_pct": float(
                    (np.mean(d_errs) - np.mean(s_errs)) / np.mean(d_errs) * 100.0
                )
                if d_errs
                else 0.0,
            }
        )
    df_by_jnt = pd.DataFrame(by_jnt_rows).sort_values(by="direct_mean_err_px", ascending=False)
    df_by_jnt.to_csv(os.path.join(metrics_dir, "evaluation_by_joint.csv"), index=False)

    # Compile Table By Pose Viewpoint
    by_pose_rows = []
    for p_name, data in by_pose_data.items():
        d_errs = data["direct"]
        s_errs = data["swapped"]
        val = data["valid"]
        by_pose_rows.append(
            {
                "pose_viewpoint": p_name,
                "valid_gt_kpts": val,
                "direct_mean_err_px": float(np.mean(d_errs)) if d_errs else np.nan,
                "direct_median_err_px": float(np.median(d_errs)) if d_errs else np.nan,
                "direct_pck20": float((np.array(d_errs) <= 20.0).sum()) / val if val else 0.0,
                "swapped_mean_err_px": float(np.mean(s_errs)) if s_errs else np.nan,
                "swapped_median_err_px": float(np.median(s_errs)) if s_errs else np.nan,
                "swapped_pck20": float((np.array(s_errs) <= 20.0).sum()) / val if val else 0.0,
                "error_reduction_pct": float(
                    (np.mean(d_errs) - np.mean(s_errs)) / np.mean(d_errs) * 100.0
                )
                if d_errs
                else 0.0,
            }
        )
    df_by_pose = pd.DataFrame(by_pose_rows).sort_values(by="valid_gt_kpts", ascending=False)
    df_by_pose.to_csv(os.path.join(metrics_dir, "evaluation_by_pose.csv"), index=False)

    # Confidence Bins
    conf_rows = []
    for ck in conf_bins_keys:
        d = conf_bins_data[ck]
        c = d["count"]
        d_errs = d["direct"]
        s_errs = d["swapped"]
        conf_rows.append(
            {
                "confidence_bin": ck,
                "keypoint_count": c,
                "direct_mean_err_px": float(np.mean(d_errs)) if d_errs else np.nan,
                "direct_median_err_px": float(np.median(d_errs)) if d_errs else np.nan,
                "swapped_mean_err_px": float(np.mean(s_errs)) if s_errs else np.nan,
                "swapped_median_err_px": float(np.median(s_errs)) if s_errs else np.nan,
                "pct_err_gt_20px_direct": float((np.array(d_errs) > 20.0).sum()) / c * 100.0
                if c
                else 0.0,
                "pct_err_gt_20px_swapped": float((np.array(s_errs) > 20.0).sum()) / c * 100.0
                if c
                else 0.0,
            }
        )
    df_conf = pd.DataFrame(conf_rows)
    df_conf.to_csv(os.path.join(metrics_dir, "evaluation_by_confidence.csv"), index=False)

    print("\n" + "=" * 60)
    print("FULL TEST BENCHMARK RESULTS (76,298 FRAMES):")
    print(f"Total Test Sequences: {summary['total_test_sequences']}")
    print(f"Total Test Frames: {summary['total_test_frames']}")
    print(f"Frame Detection Rate: {summary['frame_detection_rate'] * 100:.2f}%")
    print(f"Total Valid GT Keypoints: {summary['total_valid_gt_keypoints']}")
    print(f"Keypoint Coverage Rate: {summary['keypoint_coverage_rate'] * 100:.2f}%")
    print("\n[DIRECT NOMINAL MAPPING]:")
    print(f"  Mean Error: {summary['direct_mapping']['mean_error_px']:.2f} px")
    print(f"  Median Error: {summary['direct_mapping']['median_error_px']:.2f} px")
    print(f"  PCK@20px: {summary['direct_mapping']['pck_20px'] * 100:.2f}%")
    print(f"  PCK-torso@0.10: {summary['direct_mapping']['pck_torso_010'] * 100:.2f}%")
    print(f"  PCK-torso@0.20: {summary['direct_mapping']['pck_torso_020'] * 100:.2f}%")
    print("\n[DIAGNOSTIC SWAPPED MAPPING]:")
    print(f"  Mean Error: {summary['diagnostic_swapped_mapping']['mean_error_px']:.2f} px")
    print(f"  Median Error: {summary['diagnostic_swapped_mapping']['median_error_px']:.2f} px")
    print(f"  PCK@20px: {summary['diagnostic_swapped_mapping']['pck_20px'] * 100:.2f}%")
    print(f"  PCK-torso@0.10: {summary['diagnostic_swapped_mapping']['pck_torso_010'] * 100:.2f}%")
    print(f"  PCK-torso@0.20: {summary['diagnostic_swapped_mapping']['pck_torso_020'] * 100:.2f}%")
    print("=" * 60 + "\n")
    return summary


def step4_generate_full_test_charts(
    metrics_dir="outputs/full_test/metrics", figures_dir="outputs/full_test/figures"
):
    os.makedirs(figures_dir, exist_ok=True)
    df_act = pd.read_csv(os.path.join(metrics_dir, "evaluation_by_action.csv"))
    df_jnt = pd.read_csv(os.path.join(metrics_dir, "evaluation_by_joint.csv"))
    df_conf = pd.read_csv(os.path.join(metrics_dir, "evaluation_by_confidence.csv"))

    # Chart 1: Mean Error by Action (Direct vs Swapped)
    fig, ax = plt.subplots(figsize=(12, 7))
    df_act_sorted = df_act.sort_values(by="direct_mean_err_px", ascending=True)
    y = np.arange(len(df_act_sorted))
    h = 0.38
    ax.barh(
        y - h / 2,
        df_act_sorted["direct_mean_err_px"],
        height=h,
        label="Direct Nominal Mapping",
        color="#d95f02",
        alpha=0.85,
    )
    ax.barh(
        y + h / 2,
        df_act_sorted["swapped_mean_err_px"],
        height=h,
        label="Diagnostic Swapped Mapping",
        color="#2b5c8f",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df_act_sorted["action"], fontsize=10)
    ax.set_xlabel("Mean Euclidean Error (pixels)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Full Test Benchmark (76,298 Frames): Mean Error by Action across 15 Actions",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart1 = os.path.join(figures_dir, "chart_full_test_error_by_action.png")
    fig.savefig(chart1, dpi=300)
    plt.close()
    print(f"Saved: {chart1}")

    # Chart 2: Mean Error by Joint (Direct vs Swapped)
    fig, ax = plt.subplots(figsize=(11, 6))
    df_jnt_sorted = df_jnt.sort_values(by="direct_mean_err_px", ascending=True)
    y = np.arange(len(df_jnt_sorted))
    ax.barh(
        y - h / 2,
        df_jnt_sorted["direct_mean_err_px"],
        height=h,
        label="Direct Nominal Mapping",
        color="#e7298a",
        alpha=0.85,
    )
    ax.barh(
        y + h / 2,
        df_jnt_sorted["swapped_mean_err_px"],
        height=h,
        label="Diagnostic Swapped Mapping",
        color="#1b9e77",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df_jnt_sorted["joint_name"], fontsize=10)
    ax.set_xlabel("Mean Euclidean Error (pixels)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Full Test Benchmark: Mean Error per Joint (12 Body Joints)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart2 = os.path.join(figures_dir, "chart_full_test_error_by_joint.png")
    fig.savefig(chart2, dpi=300)
    plt.close()
    print(f"Saved: {chart2}")

    # Chart 3: PCK@20px by Action
    fig, ax = plt.subplots(figsize=(12, 7))
    df_act_pck = df_act.sort_values(by="swapped_pck20", ascending=True)
    y = np.arange(len(df_act_pck))
    ax.barh(
        y - h / 2,
        df_act_pck["direct_pck20"] * 100,
        height=h,
        label="Direct Nominal Mapping",
        color="#fc8d62",
        alpha=0.85,
    )
    ax.barh(
        y + h / 2,
        df_act_pck["swapped_pck20"] * 100,
        height=h,
        label="Diagnostic Swapped Mapping",
        color="#66c2a5",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df_act_pck["action"], fontsize=10)
    ax.set_xlabel("PCK@20px (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Full Test Benchmark: PCK@20px across 15 Actions", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    chart3 = os.path.join(figures_dir, "chart_full_test_pck_by_action.png")
    fig.savefig(chart3, dpi=300)
    plt.close()
    print(f"Saved: {chart3}")

    # Chart 4: Error by Pose Viewpoint
    df_pose = pd.read_csv(os.path.join(metrics_dir, "evaluation_by_pose.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df_pose))
    ax.barh(
        y - h / 2,
        df_pose["direct_mean_err_px"],
        height=h,
        label="Direct Nominal Mapping",
        color="#e41a1c",
        alpha=0.85,
    )
    ax.barh(
        y + h / 2,
        df_pose["swapped_mean_err_px"],
        height=h,
        label="Diagnostic Swapped Mapping",
        color="#377eb8",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df_pose["pose_viewpoint"], fontsize=11)
    ax.set_xlabel("Mean Euclidean Error (pixels)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Full Test Benchmark: Error by Camera Viewpoint (Pose)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart4 = os.path.join(figures_dir, "chart_full_test_error_by_pose.png")
    fig.savefig(chart4, dpi=300)
    plt.close()
    print(f"Saved: {chart4}")

    # Chart 5: Error and Count by Confidence Bin
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_conf))
    ax1.bar(
        x, df_conf["keypoint_count"], width=0.4, color="#cbd5e1", label="Keypoint Count", alpha=0.7
    )
    ax1.set_ylabel("Number of Keypoints", fontsize=11, fontweight="bold", color="#475569")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_conf["confidence_bin"], fontsize=10)
    ax1.set_xlabel("YOLO Confidence Bin", fontsize=11, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df_conf["swapped_mean_err_px"],
        color="#0284c7",
        marker="o",
        linewidth=2.5,
        label="Swapped Mean Error (px)",
    )
    ax2.plot(
        x,
        df_conf["pct_err_gt_20px_swapped"],
        color="#ef4444",
        marker="s",
        linewidth=2.5,
        linestyle="--",
        label="% Error > 20px (Swapped)",
    )
    ax2.set_ylabel("Error (px) / Outlier Rate (%)", fontsize=11, fontweight="bold", color="#0284c7")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    ax1.set_title(
        "Full Test: Error vs Keypoint Confidence Score", fontsize=13, fontweight="bold", pad=12
    )
    plt.tight_layout()
    chart5 = os.path.join(figures_dir, "chart_full_test_confidence_analysis.png")
    fig.savefig(chart5, dpi=300)
    plt.close()
    print(f"Saved: {chart5}")


def main():
    print("=" * 70)
    print("STARTING FULL TEST SET EVALUATION (76,298 FRAMES / 1,068 SEQUENCES)")
    print("=" * 70)

    # 1. Prepare frame list
    df_test = step1_prepare_test_frames()

    # 2. Batch inference
    step2_run_batch_inference(df_test, batch_size=64)

    # 3. Evaluate metrics
    step3_evaluate_full_test()

    # 4. Generate summary charts
    step4_generate_full_test_charts()

    print("=" * 70)
    print("FULL TEST EVALUATION FINISHED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
