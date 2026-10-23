"""
Module: run_inference.py
Purpose: Run official YOLO11n-Pose on selected Penn Action frames.
         Supports saving both raw detections (all persons) and selected detection
         (highest person detection confidence, strictly NO GT used).
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# COCO 17 Keypoints names
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def run_yolo_pose_inference(
    frames_csv="data/processed/selected_frames.csv",
    model_path="models/yolo11n-pose.pt",
    output_selected_csv="outputs/predictions/yolo_predictions.csv",
    output_raw_json="outputs/predictions/yolo_predictions_raw.json",
    imgsz=640,
    conf=0.25,
    device="cuda:0"
):
    """
    Run YOLO11n-Pose inference frame-by-frame.
    Rules:
    - Strictly autonomous inference (no GT used).
    - If multiple persons detected: select detection with highest person box confidence.
    - If no person detected: mark has_detection = False and fill NaNs.
    """
    os.makedirs(os.path.dirname(output_selected_csv), exist_ok=True)
    if not os.path.exists(frames_csv):
        raise FileNotFoundError(f"Frames list not found: {frames_csv}")

    df_frames = pd.read_csv(frames_csv)
    print(f"Loaded {len(df_frames)} frames from {frames_csv}")

    print(f"Loading YOLO model from {model_path} on {device}...")
    model = YOLO(model_path)
    
    raw_predictions = []
    selected_records = []

    for _, row in tqdm(df_frames.iterrows(), total=len(df_frames), desc="YOLO Pose Inference"):
        img_path = row['image_path']
        seq_id = str(row['sequence_id'])
        action = row['action']
        frame_idx = int(row['frame_index'])

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found {img_path}")
            continue

        # Run model inference
        results = model.predict(
            source=img_path,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False
        )[0]

        num_persons = 0
        boxes = results.boxes
        keypoints = results.keypoints

        frame_raw_dets = []
        if boxes is not None and len(boxes) > 0:
            num_persons = len(boxes)
            for i in range(num_persons):
                box_xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                box_conf = float(boxes.conf[i].cpu().item())
                kpts_xy = keypoints.xy[i].cpu().numpy().tolist() # (17, 2)
                kpts_conf = keypoints.conf[i].cpu().numpy().tolist() if keypoints.conf is not None else [1.0]*17
                frame_raw_dets.append({
                    'det_idx': i,
                    'box_xyxy': box_xyxy,
                    'box_conf': box_conf,
                    'keypoints_xy': kpts_xy,
                    'keypoints_conf': kpts_conf
                })

        raw_predictions.append({
            'sequence_id': seq_id,
            'action': action,
            'frame_index': frame_idx,
            'image_path': img_path,
            'num_persons_detected': num_persons,
            'detections': frame_raw_dets
        })

        # Selection rule: Highest person box confidence
        if num_persons > 0:
            best_det = max(frame_raw_dets, key=lambda d: d['box_conf'])
            b_xyxy = best_det['box_xyxy']
            b_conf = best_det['box_conf']
            k_xy = np.array(best_det['keypoints_xy']) # (17, 2)
            k_conf = np.array(best_det['keypoints_conf']) # (17,)

            rec = {
                'sequence_id': seq_id,
                'action': action,
                'frame_index': frame_idx,
                'image_path': img_path,
                'annotation_path': row['annotation_path'],
                'has_detection': True,
                'num_persons_detected': num_persons,
                'det_confidence': b_conf,
                'bbox_x1': b_xyxy[0],
                'bbox_y1': b_xyxy[1],
                'bbox_x2': b_xyxy[2],
                'bbox_y2': b_xyxy[3]
            }
            for c_idx, c_name in enumerate(COCO_KEYPOINTS):
                rec[f"pred_{c_name}_x"] = float(k_xy[c_idx, 0])
                rec[f"pred_{c_name}_y"] = float(k_xy[c_idx, 1])
                rec[f"pred_{c_name}_conf"] = float(k_conf[c_idx])
        else:
            rec = {
                'sequence_id': seq_id,
                'action': action,
                'frame_index': frame_idx,
                'image_path': img_path,
                'annotation_path': row['annotation_path'],
                'has_detection': False,
                'num_persons_detected': 0,
                'det_confidence': np.nan,
                'bbox_x1': np.nan,
                'bbox_y1': np.nan,
                'bbox_x2': np.nan,
                'bbox_y2': np.nan
            }
            for c_idx, c_name in enumerate(COCO_KEYPOINTS):
                rec[f"pred_{c_name}_x"] = np.nan
                rec[f"pred_{c_name}_y"] = np.nan
                rec[f"pred_{c_name}_conf"] = np.nan

        selected_records.append(rec)

    # Save outputs
    df_selected = pd.DataFrame(selected_records)
    df_selected.to_csv(output_selected_csv, index=False)
    print(f"Saved selected predictions to {output_selected_csv}")

    with open(output_raw_json, "w") as f:
        json.dump(raw_predictions, f, indent=2)
    print(f"Saved raw detections to {output_raw_json}")

    # Print summary
    total_f = len(df_selected)
    detected_f = df_selected['has_detection'].sum()
    print(f"Inference complete: {detected_f}/{total_f} frames have person detections ({detected_f/total_f*100:.1f}%)")
    return df_selected

if __name__ == "__main__":
    run_yolo_pose_inference()
