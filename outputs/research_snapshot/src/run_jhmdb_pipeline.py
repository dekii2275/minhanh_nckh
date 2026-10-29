"""
End-to-End J-HMDB Benchmark Pipeline:
1. Parse J-HMDB (21 classes, 928 videos, Split 1: 660 train, 268 test).
2. Extract Ground Truth skeletons (12 joints).
3. Run YOLO11n-Pose inference to extract Raw Skeletons.
4. Run our proposed FullCalibrationPipeline (KineCalib) to generate Calibrated Skeletons.
5. Compute quantitative Pose Estimation & Calibration metrics (Jitter, Bone Std, PCK, Euclidean Error).
6. Train and evaluate BiLSTM + Self-Attention Action Recognition (21 classes) on all modalities.
7. Save summary JSON and comparative outputs.
"""

import json
import os
import time

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from models.action_bilstm import ActionBiLSTMAttentionClassifier as ActionBiLSTM
from skeleton_calibrator import FullCalibrationPipeline

# Constants & Paths
JHMDB_ROOT = os.path.abspath("data/raw/jhmdb")
VIDEO_DIR = os.path.join(JHMDB_ROOT, "ReCompress_Videos")
JOINT_DIR = os.path.join(JHMDB_ROOT, "joint_positions")
SPLIT_DIR = os.path.join(JHMDB_ROOT, "splits")
OUTPUT_DIR = os.path.abspath("outputs/jhmdb")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# J-HMDB official 15-joint order:
# 0: neck, 1: belly, 2: head, 3: r_shoulder, 4: l_shoulder, 5: r_hip, 6: l_hip,
# 7: r_elbow, 8: l_elbow, 9: r_knee, 10: l_knee, 11: r_wrist, 12: l_wrist, 13: r_ankle, 14: l_ankle
# Standard 12 joints order (l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_kn, r_kn, l_an, r_an):
JHMDB_12_INDICES = [4, 3, 8, 7, 12, 11, 6, 5, 10, 9, 14, 13]

# COCO indices for YOLO11:
COCO_12_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Bones connecting pairs of indices (0 to 11):
BONES = [
    (0, 1),  # shoulders
    (0, 2),
    (2, 4),  # left arm
    (1, 3),
    (3, 5),  # right arm
    (6, 7),  # hips
    (0, 6),
    (1, 7),  # torso
    (6, 8),
    (8, 10),  # left leg
    (7, 9),
    (9, 11),  # right leg
]

ACTION_CLASSES = sorted(
    [
        "brush_hair",
        "catch",
        "clap",
        "climb_stairs",
        "golf",
        "jump",
        "kick_ball",
        "pick",
        "pour",
        "pullup",
        "push",
        "run",
        "shoot_ball",
        "shoot_bow",
        "shoot_gun",
        "sit",
        "stand",
        "swing_baseball",
        "throw",
        "walk",
        "wave",
    ]
)
CLASS_TO_IDX = {c: i for i, c in enumerate(ACTION_CLASSES)}


def parse_split(split_num=1):
    """Parse Split 1 text files into train and test video lists."""
    train_videos = []
    test_videos = []
    for action in ACTION_CLASSES:
        split_file = os.path.join(SPLIT_DIR, f"{action}_test_split{split_num}.txt")
        if not os.path.exists(split_file):
            continue
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    vname, flag = parts[0], parts[1]
                    item = {
                        "action": action,
                        "video_name": vname,
                        "video_id": vname.replace(".avi", ""),
                    }
                    if flag == "1":
                        train_videos.append(item)
                    elif flag == "2":
                        test_videos.append(item)
    return train_videos, test_videos


def extract_all_skeletons(all_videos, device="cuda"):
    """
    Extract Ground Truth, Raw YOLO, and Calibrated YOLO skeletons for all videos.
    Returns dictionary indexed by video_id.
    """
    cache_path = os.path.join(OUTPUT_DIR, "jhmdb_extracted_skeletons.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached skeletons from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return data["skeletons"].item()

    print(f"Extracting skeletons for {len(all_videos)} J-HMDB videos using YOLO11n...")
    model = YOLO("models/yolo11n-pose.pt")
    # J-HMDB is already subject-centric (puppet flow), matching COCO/YOLO.
    calibrator = FullCalibrationPipeline(lr_mode=None)
    results = {}

    start_time = time.time()
    total_frames = 0

    for item in tqdm(all_videos, desc="Extracting Skeletons"):
        action = item["action"]
        vname = item["video_name"]
        vid = item["video_id"]
        vpath = os.path.join(VIDEO_DIR, action, vname)
        mpath = os.path.join(JOINT_DIR, action, vid, "joint_positions.mat")

        if not os.path.exists(vpath) or not os.path.exists(mpath):
            continue

        mat = sio.loadmat(mpath)
        gt_pos_full = mat["pos_img"]  # (2, 15, T)
        T = gt_pos_full.shape[2]
        total_frames += T

        # GT 12 joints: (T, 12, 2)
        gt_12 = gt_pos_full[:, JHMDB_12_INDICES, :].transpose(2, 1, 0)  # (T, 12, 2)

        # Video frames
        cap = cv2.VideoCapture(vpath)
        frames = []
        for _ in range(T):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Run YOLO on frames
        raw_kps = np.zeros((T, 12, 2), dtype=np.float32)
        raw_confs = np.zeros((T, 12), dtype=np.float32)

        # Batch YOLO inference
        if len(frames) > 0:
            yolo_outs = model.predict(frames, device=device, verbose=False, imgsz=640)
            for t_idx, out in enumerate(yolo_outs):
                if out.keypoints is not None and len(out.keypoints.xy) > 0:
                    best_person_idx = 0
                    if out.boxes is not None and len(out.boxes.conf) > 0:
                        best_person_idx = int(torch.argmax(out.boxes.conf).cpu())

                    kps_xy = out.keypoints.xy[best_person_idx].cpu().numpy()  # (17, 2)
                    kps_conf = (
                        out.keypoints.conf[best_person_idx].cpu().numpy()
                        if out.keypoints.conf is not None
                        else np.ones(17)
                    )

                    raw_kps[t_idx] = kps_xy[COCO_12_INDICES]
                    raw_confs[t_idx] = kps_conf[COCO_12_INDICES]

        # Calibrate using our proposed KineCalib pipeline
        calib_kps = calibrator.calibrate_array(raw_kps, raw_confs)

        results[vid] = {
            "action": action,
            "gt": gt_12,
            "raw": raw_kps,
            "calib": calib_kps,
            "confs": raw_confs,
            "num_frames": T,
        }

    elapsed = time.time() - start_time
    fps = total_frames / max(elapsed, 1e-4)
    print(f"Extraction completed in {elapsed:.2f}s ({fps:.1f} FPS across {total_frames} frames).")

    np.savez_compressed(cache_path, skeletons=results)
    return results


def compute_pose_metrics(skeletons, test_videos):
    """Compute quantitative pose estimation and calibration metrics on Test Split 1."""
    raw_errors = []
    calib_errors = []
    raw_jitters = []
    calib_jitters = []
    raw_bone_stds = []
    calib_bone_stds = []

    pck20_raw = 0
    pck20_calib = 0
    total_pts = 0

    torso_pck10_raw = 0
    torso_pck10_calib = 0
    torso_pck20_raw = 0
    torso_pck20_calib = 0

    for item in test_videos:
        vid = item["video_id"]
        if vid not in skeletons:
            continue
        data = skeletons[vid]
        gt = data["gt"]  # (T, 12, 2)
        raw = data["raw"]  # (T, 12, 2)
        calib = data["calib"]  # (T, 12, 2)
        T = data["num_frames"]

        # Torso diameter (shoulder to opposite hip)
        l_shoulder = gt[:, 0]
        r_hip = gt[:, 7]
        r_shoulder = gt[:, 1]
        l_hip = gt[:, 6]
        d1 = np.linalg.norm(l_shoulder - r_hip, axis=1)
        d2 = np.linalg.norm(r_shoulder - l_hip, axis=1)
        torso_size = np.maximum(d1, d2)  # (T,)

        # Errors
        err_raw = np.linalg.norm(raw - gt, axis=2)  # (T, 12)
        err_calib = np.linalg.norm(calib - gt, axis=2)  # (T, 12)

        raw_errors.extend(err_raw.flatten())
        calib_errors.extend(err_calib.flatten())

        pck20_raw += np.sum(err_raw <= 20.0)
        pck20_calib += np.sum(err_calib <= 20.0)
        total_pts += err_raw.size

        # Torso normalized PCK
        torso_diag = torso_size[:, None]  # (T, 1)
        torso_pck10_raw += np.sum(err_raw <= 0.10 * torso_diag)
        torso_pck10_calib += np.sum(err_calib <= 0.10 * torso_diag)
        torso_pck20_raw += np.sum(err_raw <= 0.20 * torso_diag)
        torso_pck20_calib += np.sum(err_calib <= 0.20 * torso_diag)

        # Jitter: mean L2 frame-to-frame displacement
        if T > 1:
            j_raw = np.mean(np.linalg.norm(np.diff(raw, axis=0), axis=2))
            j_calib = np.mean(np.linalg.norm(np.diff(calib, axis=0), axis=2))
            raw_jitters.append(j_raw)
            calib_jitters.append(j_calib)

        # Bone length standard deviation across sequence
        for p1, p2 in BONES:
            l_raw = np.linalg.norm(raw[:, p1] - raw[:, p2], axis=1)
            l_calib = np.linalg.norm(calib[:, p1] - calib[:, p2], axis=1)
            raw_bone_stds.append(np.std(l_raw))
            calib_bone_stds.append(np.std(l_calib))

    metrics = {
        "raw_mean_error_px": float(np.mean(raw_errors)),
        "calib_mean_error_px": float(np.mean(calib_errors)),
        "error_reduction_pct": float(
            (np.mean(raw_errors) - np.mean(calib_errors)) / np.mean(raw_errors) * 100
        ),
        "raw_pck20": float(pck20_raw / total_pts * 100),
        "calib_pck20": float(pck20_calib / total_pts * 100),
        "pck20_gain_abs": float((pck20_calib - pck20_raw) / total_pts * 100),
        "raw_torso_pck10": float(torso_pck10_raw / total_pts * 100),
        "calib_torso_pck10": float(torso_pck10_calib / total_pts * 100),
        "raw_torso_pck20": float(torso_pck20_raw / total_pts * 100),
        "calib_torso_pck20": float(torso_pck20_calib / total_pts * 100),
        "raw_jitter_px_per_frame": float(np.mean(raw_jitters)),
        "calib_jitter_px_per_frame": float(np.mean(calib_jitters)),
        "jitter_reduction_pct": float(
            (np.mean(raw_jitters) - np.mean(calib_jitters)) / np.mean(raw_jitters) * 100
        ),
        "raw_bone_std_px": float(np.mean(raw_bone_stds)),
        "calib_bone_std_px": float(np.mean(calib_bone_stds)),
        "bone_std_reduction_pct": float(
            (np.mean(raw_bone_stds) - np.mean(calib_bone_stds)) / np.mean(raw_bone_stds) * 100
        ),
    }
    return metrics


class JHMDBActionDataset(Dataset):
    def __init__(self, video_items, skeletons, modality="gt", max_frames=45):
        self.samples = []
        self.max_frames = max_frames

        for item in video_items:
            vid = item["video_id"]
            if vid in skeletons:
                data = skeletons[vid]
                kps = data[modality].copy()  # (T, 12, 2)
                label = CLASS_TO_IDX[item["action"]]

                # Normalization: MidHip centering & torso scaling
                mid_hip = (kps[:, 6:7, :] + kps[:, 7:8, :]) / 2.0
                kps_centered = kps - mid_hip

                # Torso scale: distance between shoulders
                shoulder_dist = np.linalg.norm(kps[:, 0] - kps[:, 1], axis=1, keepdims=True)
                scale = np.mean(shoulder_dist)
                if scale < 1.0:
                    scale = 50.0
                kps_norm = (kps_centered / scale).reshape(kps.shape[0], -1)  # (T, 24)

                self.samples.append((torch.tensor(kps_norm, dtype=torch.float32), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded = torch.zeros(len(sequences), max_len, sequences[0].shape[-1])
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded, torch.tensor(lengths), torch.tensor(labels)


def train_and_eval_bilstm(
    train_items, test_items, skeletons, train_mod="gt", epochs=50, device="cuda"
):
    """Train BiLSTM on specified training modality and evaluate on all test modalities."""
    train_dataset = JHMDBActionDataset(train_items, skeletons, modality=train_mod)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 21 classes for J-HMDB
    model = ActionBiLSTM(
        input_size=24, hidden_size=128, num_classes=21, num_layers=2, attention_heads=4
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    for ep in range(epochs):
        for seqs, lens, targets in train_loader:
            seqs, lens, targets = seqs.to(device), lens.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs, lens)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate across all 3 test modalities
    model.eval()
    results = {}
    for test_mod in ["gt", "raw", "calib"]:
        test_dataset = JHMDBActionDataset(test_items, skeletons, modality=test_mod)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for seqs, lens, targets in test_loader:
                seqs, lens, targets = seqs.to(device), lens.to(device), targets.to(device)
                outputs = model(seqs, lens)

                # Top-1
                preds_top1 = outputs.argmax(dim=1)
                correct_top1 += (preds_top1 == targets).sum().item()

                # Top-5
                _, preds_top5 = outputs.topk(5, dim=1)
                targets_expanded = targets.unsqueeze(1).expand_as(preds_top5)
                correct_top5 += (preds_top5 == targets_expanded).any(dim=1).sum().item()

                total += targets.size(0)

        results[f"{test_mod}_top1_acc"] = float(correct_top1 / total * 100)
        results[f"{test_mod}_top5_acc"] = float(correct_top5 / total * 100)

    return results


def main():
    print("=" * 70)
    print("J-HMDB BENCHMARK: POSE EVALUATION, CALIBRATION & ACTION RECOGNITION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )

    # Step 1: Parse splits
    train_videos, test_videos = parse_split(split_num=1)
    all_videos = train_videos + test_videos
    print(
        f"J-HMDB Split 1: {len(train_videos)} train videos, {len(test_videos)} test videos (Total: {len(all_videos)})"
    )

    # Step 2: Extract skeletons
    skeletons = extract_all_skeletons(all_videos, device=device)

    # Step 3: Compute Pose & Calibration metrics
    print("\n--- Computing Pose Estimation & Calibration Metrics on Test Set ---")
    pose_metrics = compute_pose_metrics(skeletons, test_videos)
    for k, v in pose_metrics.items():
        print(f"  {k}: {v:.2f}")

    # Step 4: Action Recognition Experiment 1: Trained on Ground Truth
    print("\n--- Training Action Recognition Baseline on Ground Truth Train (50 Epochs) ---")
    gt_trained_results = train_and_eval_bilstm(
        train_videos, test_videos, skeletons, train_mod="gt", epochs=50, device=device
    )
    print("Results (GT-trained model):")
    for k, v in gt_trained_results.items():
        print(f"  {k}: {v:.2f}%")

    # Step 5: Action Recognition Experiment 2: Domain-Matched Training on Calibrated Skeleton
    print("\n--- Training Action Recognition Model on Calibrated Skeleton Train (50 Epochs) ---")
    calib_trained_results = train_and_eval_bilstm(
        train_videos, test_videos, skeletons, train_mod="calib", epochs=50, device=device
    )
    print("Results (Domain-Matched Calibrated model):")
    for k, v in calib_trained_results.items():
        print(f"  {k}: {v:.2f}%")

    # Save summary
    summary = {
        "dataset": "J-HMDB",
        "num_classes": 21,
        "split": 1,
        "train_videos": len(train_videos),
        "test_videos": len(test_videos),
        "pose_and_calibration_metrics": pose_metrics,
        "action_recognition_gt_trained": gt_trained_results,
        "action_recognition_domain_matched": calib_trained_results,
    }

    out_file = os.path.join(OUTPUT_DIR, "jhmdb_benchmark_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nAll results saved to {out_file} successfully!")


if __name__ == "__main__":
    main()
