"""
Module: run_pipeline.py
Purpose: Master execution pipeline coordinating dataset inspection, sequence selection,
         YOLO11n-Pose inference, quantitative evaluation, and visualization.
"""

import os
import sys
import time

def run_full_pilot_pipeline():
    print("=" * 70)
    print("STARTING PENN ACTION - YOLO11n-POSE EVALUATION PILOT")
    print("=" * 70)

    # 1. Check extracted dataset
    penn_dir = "data/raw/Penn_Action"
    if not os.path.exists(os.path.join(penn_dir, "frames")) or not os.path.exists(os.path.join(penn_dir, "labels")):
        print(f"Dataset not extracted yet at {penn_dir}.")
        archive = "data/raw/Penn_Action.tar.gz"
        if os.path.exists(archive) and not os.path.exists(archive + ".aria2"):
            from download_dataset import verify_and_extract_dataset
            verify_and_extract_dataset(archive, "data/raw")
        else:
            print("Penn_Action.tar.gz is still downloading. Please run after download completes.")
            return False

    # 2. Inspect Dataset & Verify Ground Truth
    print("\n--- STEP 1: INSPECTING GROUND TRUTH ANNOTATIONS ---")
    from inspect_dataset import inspect_and_verify_gt
    gt_ok = inspect_and_verify_gt(penn_dir)
    if not gt_ok:
        print("ERROR: Ground Truth verification failed! Halting pipeline.")
        return False
    print("Ground Truth verified successfully.")

    # 3. Select Target Sequences
    print("\n--- STEP 2: DETERMINISTIC SEQUENCE SELECTION ---")
    from select_sequences import select_pilot_dataset
    df_selected = select_pilot_dataset(
        penn_dir=penn_dir,
        target_actions=["jumping_jacks", "squat", "tennis_forehand"],
        frames_per_sequence=50,
        output_csv="data/processed/selected_frames.csv"
    )

    # 4. Run YOLO11n-Pose Inference
    print("\n--- STEP 3: YOLO11n-POSE INFERENCE ---")
    from run_inference import run_yolo_pose_inference
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    run_yolo_pose_inference(
        frames_csv="data/processed/selected_frames.csv",
        model_path="models/yolo11n-pose.pt",
        output_selected_csv="outputs/predictions/yolo_predictions.csv",
        output_raw_json="outputs/predictions/yolo_predictions_raw.json",
        imgsz=640,
        conf=0.25,
        device=device
    )

    # 5. Evaluate Quantitative Metrics
    print("\n--- STEP 4: QUANTITATIVE EVALUATION & PCK ---")
    from evaluate import evaluate_pipeline
    summary = evaluate_pipeline(
        predictions_csv="outputs/predictions/yolo_predictions.csv",
        mapping_json="configs/joint_mapping.json",
        metrics_dir="outputs/metrics"
    )

    # 6. Generate Visualizations & Demos
    print("\n--- STEP 5: VISUALIZATIONS & ARTIFACTS ---")
    from visualize import generate_visualizations
    generate_visualizations(
        predictions_csv="outputs/predictions/yolo_predictions.csv",
        kpts_csv="outputs/metrics/evaluation_per_keypoint.csv",
        by_joint_csv="outputs/metrics/evaluation_by_joint.csv",
        mapping_json="configs/joint_mapping.json",
        figures_dir="outputs/figures",
        demo_dir="outputs/demo"
    )

    print("\n" + "=" * 70)
    print("PILOT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    # Add src to sys.path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    run_full_pilot_pipeline()
