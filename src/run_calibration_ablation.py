"""
Module: run_calibration_ablation.py
Purpose: Step-by-step ablation study of skeleton calibration modules on Penn Action Test Split (76,298 frames).
Usage:
    python src/run_calibration_ablation.py --module module1_one_euro
    python src/run_calibration_ablation.py --module module2_interpolation
    python src/run_calibration_ablation.py --module module3_bone_constraint
    python src/run_calibration_ablation.py --module module4_left_right
    python src/run_calibration_ablation.py --module module5_full_combined
    python src/run_calibration_ablation.py --all
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from skeleton_calibrator import (
    OneEuroSmoothingCalibrator,
    KinematicInterpolationCalibrator,
    AnthropometricBoneConstraintCalibrator,
    LeftRightContinuityCalibrator,
    FullCalibrationPipeline,
    COCO_KEYPOINTS,
    BONE_SEGMENTS
)

def load_joint_mapping(mapping_json="configs/joint_mapping.json"):
    with open(mapping_json, "r") as f:
        data = json.load(f)
    return data["joint_pairs"]

def get_torso_scale(x_penn, y_penn, vis_penn):
    if vis_penn[1] == 1 and vis_penn[2] == 1 and vis_penn[7] == 1 and vis_penn[8] == 1:
        if x_penn[1] > 0 and x_penn[2] > 0 and x_penn[7] > 0 and x_penn[8] > 0:
            sx, sy = (x_penn[1] + x_penn[2]) / 2.0, (y_penn[1] + y_penn[2]) / 2.0
            hx, hy = (x_penn[7] + x_penn[8]) / 2.0, (y_penn[7] + y_penn[8]) / 2.0
            scale = np.sqrt((sx - hx)**2 + (sy - hy)**2)
            if scale > 1e-3:
                return float(scale)
    return None

def compute_motion_jitter(df_preds):
    """
    Computes mean acceleration norm (jitter) across all sequences for tracked joints.
    Jitter = mean ||x_{t+1} - 2*x_t + x_{t-1}||_2
    """
    jitters = []
    for seq_id, seq_df in df_preds.groupby('sequence_id'):
        if len(seq_df) < 3:
            continue
        for kpt in ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_ankle", "right_ankle"]:
            col_x, col_y = f"pred_{kpt}_x", f"pred_{kpt}_y"
            if col_x in seq_df.columns:
                xs = seq_df[col_x].values
                ys = seq_df[col_y].values
                valid = ~np.isnan(xs) & ~np.isnan(ys)
                if np.sum(valid) >= 3:
                    # Discrete 2nd derivative
                    acc_x = np.diff(xs, n=2)
                    acc_y = np.diff(ys, n=2)
                    acc_norm = np.sqrt(acc_x**2 + acc_y**2)
                    valid_acc = acc_norm[~np.isnan(acc_norm)]
                    if len(valid_acc) > 0:
                        jitters.extend(valid_acc.tolist())
    return float(np.mean(jitters)) if jitters else 0.0

def compute_bone_length_dispersion(df_preds):
    """
    Computes mean standard deviation of bone lengths across sequences.
    """
    stds = []
    for seq_id, seq_df in df_preds.groupby('sequence_id'):
        if len(seq_df) < 3:
            continue
        for parent_kpt, child_kpt in BONE_SEGMENTS:
            px_col, py_col = f"pred_{parent_kpt}_x", f"pred_{parent_kpt}_y"
            cx_col, cy_col = f"pred_{child_kpt}_x", f"pred_{child_kpt}_y"
            if px_col in seq_df.columns and cx_col in seq_df.columns:
                dx = seq_df[cx_col].values - seq_df[px_col].values
                dy = seq_df[cy_col].values - seq_df[py_col].values
                lens = np.sqrt(dx**2 + dy**2)
                valid_lens = lens[~np.isnan(lens) & (lens > 1e-2)]
                if len(valid_lens) >= 3:
                    stds.append(float(np.std(valid_lens)))
    return float(np.mean(stds)) if stds else 0.0

def evaluate_dataframe(df_preds, annotations_cache, joint_pairs):
    """
    Calculates quantitative metrics against Penn Action ground truth.
    """
    errors = []
    pck20 = []
    pck_t01 = []
    pck_t02 = []

    for row in df_preds.itertuples(index=False):
        mat_path = row.annotation_path
        f_idx = row.frame_index
        has_det = row.has_detection

        x_mat, y_mat, vis_mat = annotations_cache[mat_path]
        f0 = f_idx - 1
        x_gt = x_mat[f0]
        y_gt = y_mat[f0]
        vis_gt = vis_mat[f0]

        t_scale = get_torso_scale(x_gt, y_gt, vis_gt)

        for p in joint_pairs:
            p_idx = p['penn_index_0based']
            c_name = p['coco_joint_name']

            if vis_gt[p_idx] == 1 and (x_gt[p_idx] > 0 or y_gt[p_idx] > 0):
                gx, gy = x_gt[p_idx], y_gt[p_idx]
                px = getattr(row, f"pred_{c_name}_x")
                py = getattr(row, f"pred_{c_name}_y")

                if has_det and not np.isnan(px) and not np.isnan(py):
                    err = float(np.sqrt((px - gx)**2 + (py - gy)**2))
                    errors.append(err)
                    pck20.append(1 if err <= 20.0 else 0)

                    if t_scale is not None:
                        pck_t01.append(1 if err <= 0.10 * t_scale else 0)
                        pck_t02.append(1 if err <= 0.20 * t_scale else 0)
                    else:
                        pck_t01.append(0)
                        pck_t02.append(0)

    errs = np.array(errors, dtype=float)
    return {
        "valid_points": len(errors),
        "mean_error_px": float(np.mean(errs)),
        "median_error_px": float(np.median(errs)),
        "std_error_px": float(np.std(errs)),
        "p90_error_px": float(np.percentile(errs, 90)),
        "pck_20px": float(np.mean(pck20)),
        "pck_torso_010": float(np.mean(pck_t01)),
        "pck_torso_020": float(np.mean(pck_t02))
    }

def run_calibration_ablation(
    module_name="module1_one_euro",
    predictions_csv="outputs/full_test/predictions/yolo_predictions_test.csv",
    mapping_json="configs/joint_mapping.json",
    output_dir="outputs/calibration"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==================================================================")
    print(f"RUNNING ABLATION EXPERIMENT: {module_name}")
    print(f"==================================================================")

    print(f"Loading predictions from {predictions_csv}...")
    df_raw = pd.read_csv(predictions_csv)
    joint_pairs = load_joint_mapping(mapping_json)

    # 1. Apply calibration module if not baseline
    t0 = time.time()
    if module_name == "baseline":
        calibrator = None
        df_calib = df_raw
    elif module_name == "module1_one_euro":
        calibrator = OneEuroSmoothingCalibrator(freq=30.0, min_cutoff=1.0, beta=0.05)
    elif module_name == "module2_interpolation":
        calibrator = KinematicInterpolationCalibrator(conf_threshold=0.35, max_gap=15)
    elif module_name == "module3_bone_constraint":
        calibrator = AnthropometricBoneConstraintCalibrator(min_ratio=0.40, max_ratio=1.20)
    elif module_name == "module4_left_right":
        calibrator = LeftRightContinuityCalibrator(mode="auto_viewer_centric")
    elif module_name == "module5_full_combined":
        calibrator = FullCalibrationPipeline()
    else:
        raise ValueError(f"Unknown module name: {module_name}")

    if calibrator is not None:
        print(f"Applying {calibrator.__class__.__name__} to 1,068 test sequences...")
        calib_chunks = []
        for seq_id, seq_df in tqdm(df_raw.groupby('sequence_id', sort=False), desc=f"Calibrating with {module_name}"):
            df_res = calibrator.calibrate_sequence(seq_df)
            calib_chunks.append(df_res)
        df_calib = pd.concat(calib_chunks, ignore_index=True)
        calib_time = time.time() - t0
        print(f"Calibration completed in {calib_time:.2f} seconds ({len(df_calib)} frames).")
    else:
        df_calib = df_raw
        calib_time = 0.0

    # 2. Compute Smoothness & Biomechanical Consistency
    print("Computing temporal jitter & bone length dispersion...")
    mean_jitter = compute_motion_jitter(df_calib)
    bone_std = compute_bone_length_dispersion(df_calib)

    # 3. Cache annotations & Compute accuracy against GT
    print("Loading Penn Action annotations cache...")
    annotations_cache = {}
    for mat_path in tqdm(df_calib['annotation_path'].unique(), desc="Caching annotations"):
        mat = sio.loadmat(mat_path)
        x_mat = np.array(mat['x'], dtype=float)
        y_mat = np.array(mat['y'], dtype=float)
        vis_mat = np.array(mat['visibility'], dtype=int)
        if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
            x_mat, y_mat, vis_mat = x_mat.T, y_mat.T, vis_mat.T
        annotations_cache[mat_path] = (x_mat, y_mat, vis_mat)

    print("Evaluating pose accuracy metrics...")
    acc_metrics = evaluate_dataframe(df_calib, annotations_cache, joint_pairs)

    result = {
        "module": module_name,
        "calib_time_seconds": calib_time,
        "temporal_jitter_px": mean_jitter,
        "bone_length_std_px": bone_std,
        "mean_error_px": acc_metrics["mean_error_px"],
        "median_error_px": acc_metrics["median_error_px"],
        "std_error_px": acc_metrics["std_error_px"],
        "p90_error_px": acc_metrics["p90_error_px"],
        "pck_20px": acc_metrics["pck_20px"],
        "pck_torso_010": acc_metrics["pck_torso_010"],
        "pck_torso_020": acc_metrics["pck_torso_020"]
    }

    print("\n--- RESULTS FOR", module_name.upper(), "---")
    print(f"  * Temporal Jitter (Mean Accel): {mean_jitter:.3f} px (lower is smoother)")
    print(f"  * Bone Length Std (Dispersion): {bone_std:.3f} px (lower is more consistent)")
    print(f"  * Mean Euclidean Error:         {acc_metrics['mean_error_px']:.2f} px")
    print(f"  * Median Error:                 {acc_metrics['median_error_px']:.2f} px")
    print(f"  * PCK@20px:                     {acc_metrics['pck_20px']*100:.2f}%")
    print(f"  * PCK-torso@0.20:               {acc_metrics['pck_torso_020']*100:.2f}%")

    out_file = os.path.join(output_dir, f"{module_name}_metrics.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved module result to {out_file}\n")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="module1_one_euro",
                        choices=["baseline", "module1_one_euro", "module2_interpolation",
                                 "module3_bone_constraint", "module4_left_right", "module5_full_combined"])
    parser.add_argument("--all", action="store_true", help="Run all modules in sequence")
    args = parser.parse_args()

    if args.all:
        modules = ["baseline", "module1_one_euro", "module2_interpolation",
                   "module3_bone_constraint", "module4_left_right", "module5_full_combined"]
        all_results = []
        for mod in modules:
            res = run_calibration_ablation(mod)
            all_results.append(res)
        df_all = pd.DataFrame(all_results)
        df_all.to_csv("outputs/calibration/ablation_summary.csv", index=False)
        print("Completed all modules! Saved summary to outputs/calibration/ablation_summary.csv")
    else:
        run_calibration_ablation(args.module)
