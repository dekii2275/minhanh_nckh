"""
Module: run_aqa_benchmark.py
Purpose: Benchmark and generate Action Quality Assessment (AQA) and Movement Profiling reports
         comparing Raw YOLO vs. Calibrated YOLO (KineCalib) across Penn Action sports sequences.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.action_dataset import COCO_TO_CANONICAL
from movement_profiling import MovementQualityProfiler
from skeleton_calibrator import FullCalibrationPipeline


def run_aqa_benchmark(
    yolo_test_csv: str = "outputs/full_test/predictions/yolo_predictions_test.csv",
    out_dir: str = "outputs/aqa",
    max_sequences_per_class: int = 15,
):
    print("\n=======================================================")
    print("  RUNNING MOVEMENT PROFILING & AQA BENCHMARK")
    print(f"  Source CSV: {yolo_test_csv}")
    print(f"  Max sequences per action class: {max_sequences_per_class}")
    print("=======================================================")

    os.makedirs(out_dir, exist_ok=True)
    profiler = MovementQualityProfiler(fps=30.0)
    calibrator = FullCalibrationPipeline()

    print(f"Loading YOLO test predictions from {yolo_test_csv}...")
    df_test = pd.read_csv(yolo_test_csv, dtype={"sequence_id": str})
    df_test["sequence_id"] = df_test["sequence_id"].astype(str).str.zfill(4)

    # Group by sequence
    grouped = df_test.groupby("sequence_id", sort=False)
    action_counts = {}

    benchmark_records = []
    case_studies = []

    print("Analyzing sports sequences and generating AQA movement profiles...")
    for seq_id, seq_df in tqdm(grouped, desc="AQA Profiling"):
        action = seq_df.iloc[0]["action"]
        count = action_counts.get(action, 0)
        if count >= max_sequences_per_class:
            continue
        action_counts[action] = count + 1

        seq_df = seq_df.sort_values("frame_index")
        T = len(seq_df)
        if T < 5:
            continue

        # Extract Raw YOLO 2D coordinates [T, 12, 2]
        raw_coords = np.zeros((T, 12, 2), dtype=np.float32)
        for c_idx, c_name in enumerate(COCO_TO_CANONICAL):
            raw_coords[:, c_idx, 0] = seq_df[f"pred_{c_name}_x"].values
            raw_coords[:, c_idx, 1] = seq_df[f"pred_{c_name}_y"].values

        # Calibrate using KineCalib
        df_cal = calibrator.calibrate_sequence(seq_df)
        calib_coords = np.zeros((T, 12, 2), dtype=np.float32)
        for c_idx, c_name in enumerate(COCO_TO_CANONICAL):
            calib_coords[:, c_idx, 0] = df_cal[f"pred_{c_name}_x"].values
            calib_coords[:, c_idx, 1] = df_cal[f"pred_{c_name}_y"].values

        # Compare AQA profiles
        comparison = profiler.compare_raw_vs_calibrated(
            raw_coords, calib_coords, action_name=action
        )
        raw_p = comparison["raw_profile"]
        calib_p = comparison["calibrated_profile"]
        diff = comparison["calibration_improvements"]

        record = {
            "seq_id": seq_id,
            "action": action,
            "frames": T,
            "raw_total_score": raw_p["total_score"],
            "calib_total_score": calib_p["total_score"],
            "score_gain": diff["total_aqa_score_gain"],
            "raw_stability": raw_p["sub_scores"]["stability_score"],
            "calib_stability": calib_p["sub_scores"]["stability_score"],
            "raw_smoothness": raw_p["sub_scores"]["smoothness_score"],
            "calib_smoothness": calib_p["sub_scores"]["smoothness_score"],
            "raw_integrity": raw_p["sub_scores"]["anatomical_integrity_score"],
            "calib_integrity": calib_p["sub_scores"]["anatomical_integrity_score"],
            "raw_symmetry": raw_p["sub_scores"]["bilateral_symmetry_score"],
            "calib_symmetry": calib_p["sub_scores"]["bilateral_symmetry_score"],
            "jitter_reduction_pct": diff["jitter_reduction_pct"],
        }
        benchmark_records.append(record)

        # Retain detailed case studies for representative athletic actions
        if (
            action in {"squat", "bench_press", "baseball_pitch", "golf_swing", "tennis_serve"}
            and len(case_studies) < 5
        ):
            case_studies.append(
                {
                    "seq_id": seq_id,
                    "action": action,
                    "raw_score": raw_p["total_score"],
                    "calib_score": calib_p["total_score"],
                    "gain": diff["total_aqa_score_gain"],
                    "formative_feedback": calib_p["formative_feedback"],
                    "calib_sub_scores": calib_p["sub_scores"],
                    "raw_sub_scores": raw_p["sub_scores"],
                }
            )

    df_records = pd.DataFrame(benchmark_records)
    print(f"\nCompleted AQA evaluation across {len(df_records)} athletic sequences.")

    # Compute macro and per-action aggregates
    action_summary = {}
    for act, act_df in df_records.groupby("action"):
        action_summary[act] = {
            "num_sequences": len(act_df),
            "raw_aqa_score_mean": round(float(act_df["raw_total_score"].mean()), 2),
            "calib_aqa_score_mean": round(float(act_df["calib_total_score"].mean()), 2),
            "aqa_gain_mean": round(float(act_df["score_gain"].mean()), 2),
            "stability_gain_mean": round(
                float((act_df["calib_stability"] - act_df["raw_stability"]).mean()), 2
            ),
            "smoothness_gain_mean": round(
                float((act_df["calib_smoothness"] - act_df["raw_smoothness"]).mean()), 2
            ),
            "integrity_gain_mean": round(
                float((act_df["calib_integrity"] - act_df["raw_integrity"]).mean()), 2
            ),
            "symmetry_gain_mean": round(
                float((act_df["calib_symmetry"] - act_df["raw_symmetry"]).mean()), 2
            ),
            "jitter_reduction_pct_mean": round(float(act_df["jitter_reduction_pct"].mean()), 2),
        }

    overall_summary = {
        "num_analyzed_sequences": len(df_records),
        "overall_metrics": {
            "raw_aqa_score_mean": round(float(df_records["raw_total_score"].mean()), 2),
            "calib_aqa_score_mean": round(float(df_records["calib_total_score"].mean()), 2),
            "overall_aqa_gain": round(float(df_records["score_gain"].mean()), 2),
            "stability_score_gain": round(
                float((df_records["calib_stability"] - df_records["raw_stability"]).mean()), 2
            ),
            "smoothness_score_gain": round(
                float((df_records["calib_smoothness"] - df_records["raw_smoothness"]).mean()), 2
            ),
            "integrity_score_gain": round(
                float((df_records["calib_integrity"] - df_records["raw_integrity"]).mean()), 2
            ),
            "symmetry_score_gain": round(
                float((df_records["calib_symmetry"] - df_records["raw_symmetry"]).mean()), 2
            ),
            "mean_jitter_reduction_pct": round(float(df_records["jitter_reduction_pct"].mean()), 2),
        },
        "per_action_summary": action_summary,
        "case_studies": case_studies,
    }

    summary_json_path = os.path.join(out_dir, "aqa_benchmark_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=4, ensure_ascii=False)

    print("\n=======================================================")
    print("  AQA MOVEMENT QUALITY BENCHMARK RESULTS")
    print("=======================================================")
    print(f"Total Sequences Analyzed:       {len(df_records)}")
    print(
        f"Raw YOLO Mean AQA Score:        {overall_summary['overall_metrics']['raw_aqa_score_mean']:.2f} / 100"
    )
    print(
        f"Calibrated YOLO Mean AQA Score: {overall_summary['overall_metrics']['calib_aqa_score_mean']:.2f} / 100"
    )
    print(
        f"Mean Movement Quality Gain:     +{overall_summary['overall_metrics']['overall_aqa_gain']:.2f} Points"
    )
    print(
        f"Mean Jitter Reduction:          -{overall_summary['overall_metrics']['mean_jitter_reduction_pct']:.2f}%"
    )
    print(f"Summary JSON saved to:          {summary_json_path}")
    print("=======================================================\n")

    return overall_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_csv", type=str, default="outputs/full_test/predictions/yolo_predictions_test.csv"
    )
    parser.add_argument("--out_dir", type=str, default="outputs/aqa")
    parser.add_argument("--max_seq", type=int, default=15)
    args = parser.parse_args()

    run_aqa_benchmark(
        yolo_test_csv=args.yolo_csv, out_dir=args.out_dir, max_sequences_per_class=args.max_seq
    )
