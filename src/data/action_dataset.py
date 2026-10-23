"""
Module: action_dataset.py
Purpose: Dataset and DataLoader for Skeleton-based Action Recognition on Penn Action.
         Preprocesses and normalizes 2D skeletons (MidHip centering + Torso/Shoulder scaling).
         Provides Train and Test splits under Ground Truth, Raw YOLO, and Calibrated YOLO modalities.
"""

import os
import glob
import json
import pickle
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd

from skeleton_calibrator import FullCalibrationPipeline

# 15 Penn Action Classes
ACTION_CLASSES = [
    "baseball_pitch", "baseball_swing", "bench_press", "bowl",
    "clean_and_jerk", "golf_swing", "jump_rope", "jumping_jacks",
    "pullup", "pushup", "situp", "squat",
    "strum_guitar", "tennis_forehand", "tennis_serve"
]
ACTION_TO_ID = {act: i for i, act in enumerate(ACTION_CLASSES)}
ID_TO_ACTION = {i: act for i, act in enumerate(ACTION_CLASSES)}

# 12 Canonical Joints (Index mapping)
CANONICAL_JOINTS = [
    ("left_shoulder", 1),
    ("right_shoulder", 2),
    ("left_elbow", 3),
    ("right_elbow", 4),
    ("left_wrist", 5),
    ("right_wrist", 6),
    ("left_hip", 7),
    ("right_hip", 8),
    ("left_knee", 9),
    ("right_knee", 10),
    ("left_ankle", 11),
    ("right_ankle", 12)
]

COCO_TO_CANONICAL = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def normalize_skeleton_sequence(coords):
    """
    Normalizes a 2D skeleton sequence [T, 12, 2]:
    1. Centering: Center all joints at MidHip (midpoint of left_hip & right_hip).
    2. Scaling: Scale by shoulder width (distance between left_shoulder & right_shoulder),
       with fallback to torso scale when shoulders are compressed.
    Returns: normalized array of shape [T, 24] (flattened 12 * 2).
    """
    T = coords.shape[0]
    out = np.zeros((T, 12, 2), dtype=np.float32)

    # Indices in our 12 canonical joints:
    # 0: left_shoulder, 1: right_shoulder
    # 6: left_hip, 7: right_hip
    for t in range(T):
        frame = coords[t] # [12, 2]

        # 1. MidHip Centering
        if not np.isnan(frame[6]).any() and not np.isnan(frame[7]).any():
            midhip = (frame[6] + frame[7]) / 2.0
        else:
            valid_pts = frame[~np.isnan(frame).any(axis=1)]
            midhip = np.mean(valid_pts, axis=0) if len(valid_pts) > 0 else np.zeros(2)

        centered = frame - midhip

        # 2. Scale Normalization
        scale = 1.0
        if not np.isnan(centered[0]).any() and not np.isnan(centered[1]).any():
            shoulder_dist = float(np.linalg.norm(centered[0] - centered[1]))
            if shoulder_dist > 5.0:
                scale = shoulder_dist
            else:
                # Fallback to torso scale
                torso_dist = float(np.linalg.norm((centered[0] + centered[1]) / 2.0))
                scale = torso_dist if torso_dist > 5.0 else 50.0
        else:
            valid_pts = centered[~np.isnan(centered).any(axis=1)]
            scale = float(np.max(np.linalg.norm(valid_pts, axis=1))) if len(valid_pts) > 0 else 50.0

        scale = max(scale, 1e-4)
        out[t] = centered / scale

    # Replace any remaining NaNs with 0
    out[np.isnan(out)] = 0.0
    return out.reshape(T, 24).astype(np.float32)


class PennActionSkeletonDataset(Dataset):
    """PyTorch Dataset for Penn Action skeleton sequences."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return features, label, sample['seq_id']


def collate_skeleton_batch(batch):
    """
    Collate function with zero-padding and lengths extraction.
    Returns:
        padded_features: [Batch, T_max, 24]
        lengths: [Batch]
        labels: [Batch]
        seq_ids: list[str]
    """
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    seq_ids = [item[2] for item in batch]

    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels_list)

    return padded_features, lengths, labels, seq_ids


def build_or_load_penn_action_splits(
    penn_dir="data/raw/Penn_Action",
    yolo_test_csv="outputs/full_test/predictions/yolo_predictions_test.csv",
    cache_path="data/processed/penn_action_bilstm_data.pkl",
    force_rebuild=False
):
    """
    Builds or loads the preprocessed dataset splits:
    - Train (GT): 1,258 sequences
    - Test GT: 1,068 sequences
    - Test Raw YOLO: 1,068 sequences
    - Test Calibrated YOLO: 1,068 sequences
    """
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading preprocessed action recognition data from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data

    print("\n--- BUILDING PENN ACTION RECOGNITION DATASET CACHE ---")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Parse all Ground Truth sequences
    print("Parsing 2,326 Ground Truth label files...")
    mat_files = sorted(glob.glob(os.path.join(penn_dir, "labels", "*.mat")))
    
    train_gt_samples = []
    test_gt_samples = []
    gt_dict = {}

    for mat_path in tqdm(mat_files, desc="Processing Ground Truth"):
        mat = sio.loadmat(mat_path)
        seq_id = os.path.splitext(os.path.basename(mat_path))[0].zfill(4)
        is_train = int(mat['train'][0, 0])
        action = str(mat['action'][0])
        if action not in ACTION_TO_ID:
            continue
        label = ACTION_TO_ID[action]

        x_mat = np.array(mat['x'], dtype=float)
        y_mat = np.array(mat['y'], dtype=float)
        vis_mat = np.array(mat['visibility'], dtype=int)
        if x_mat.shape[1] != 13 and x_mat.shape[0] == 13:
            x_mat, y_mat, vis_mat = x_mat.T, y_mat.T, vis_mat.T

        T = x_mat.shape[0]
        coords = np.zeros((T, 12, 2), dtype=np.float32)

        for c_idx, (j_name, p_idx) in enumerate(CANONICAL_JOINTS):
            coords[:, c_idx, 0] = x_mat[:, p_idx]
            coords[:, c_idx, 1] = y_mat[:, p_idx]
            # Zero out invisible
            invis = (vis_mat[:, p_idx] == 0) | (coords[:, c_idx, 0] <= 0)
            coords[invis, c_idx, :] = np.nan

        norm_feats = normalize_skeleton_sequence(coords)
        sample = {
            'seq_id': seq_id,
            'features': norm_feats,
            'label': label,
            'action': action
        }
        gt_dict[seq_id] = sample

        if is_train == 1:
            train_gt_samples.append(sample)
        elif is_train == -1:
            test_gt_samples.append(sample)

    print(f"Parsed {len(train_gt_samples)} Train GT sequences and {len(test_gt_samples)} Test GT sequences.")

    # 2. Parse Raw YOLO Test Sequences
    print("\nParsing Raw YOLO test sequences from predictions CSV...")
    df_test = pd.read_csv(yolo_test_csv, dtype={'sequence_id': str})
    df_test['sequence_id'] = df_test['sequence_id'].astype(str).str.zfill(4)

    test_raw_samples = []
    for seq_id, seq_df in tqdm(df_test.groupby('sequence_id', sort=False), desc="Processing Raw YOLO Test"):
        seq_df = seq_df.sort_values('frame_index')
        action = seq_df.iloc[0]['action']
        if action not in ACTION_TO_ID:
            continue
        label = ACTION_TO_ID[action]

        T = len(seq_df)
        coords = np.zeros((T, 12, 2), dtype=np.float32)
        for c_idx, c_name in enumerate(COCO_TO_CANONICAL):
            col_x, col_y = f"pred_{c_name}_x", f"pred_{c_name}_y"
            coords[:, c_idx, 0] = seq_df[col_x].values
            coords[:, c_idx, 1] = seq_df[col_y].values

        norm_feats = normalize_skeleton_sequence(coords)
        test_raw_samples.append({
            'seq_id': seq_id,
            'features': norm_feats,
            'label': label,
            'action': action
        })

    # 3. Parse Calibrated YOLO Test Sequences
    print("\nCalibrating YOLO test sequences using FullCalibrationPipeline...")
    calibrator = FullCalibrationPipeline()
    test_calib_samples = []

    for seq_id, seq_df in tqdm(df_test.groupby('sequence_id', sort=False), desc="Calibrating Test Sequences"):
        seq_df = seq_df.sort_values('frame_index')
        df_cal = calibrator.calibrate_sequence(seq_df)
        action = df_cal.iloc[0]['action']
        if action not in ACTION_TO_ID:
            continue
        label = ACTION_TO_ID[action]

        T = len(df_cal)
        coords = np.zeros((T, 12, 2), dtype=np.float32)
        for c_idx, c_name in enumerate(COCO_TO_CANONICAL):
            col_x, col_y = f"pred_{c_name}_x", f"pred_{c_name}_y"
            coords[:, c_idx, 0] = df_cal[col_x].values
            coords[:, c_idx, 1] = df_cal[col_y].values

        norm_feats = normalize_skeleton_sequence(coords)
        test_calib_samples.append({
            'seq_id': seq_id,
            'features': norm_feats,
            'label': label,
            'action': action
        })

    data = {
        'train_gt': train_gt_samples,
        'test_gt': test_gt_samples,
        'test_raw_yolo': test_raw_samples,
        'test_calib_yolo': test_calib_samples
    }

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved dataset cache to {cache_path} ({len(data['train_gt'])} train, {len(data['test_gt'])} test)")
    return data
