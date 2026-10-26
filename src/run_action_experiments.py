"""
Module: run_action_experiments.py
Purpose: Independent ablation study of 4 improvement directions for Action Recognition:
         - Direction 1: Kinematic Feature Enrichment (64 dims: Pos + Vel + Bones)
         - Direction 2: Noise-Aware Pose Augmentation (Gaussian jitter, Dropout, Scale)
         - Direction 3: Spatio-Temporal Dual Attention (Spatial Joint Attention + Temporal LSTM)
         - Direction 4: Combined SOTA Pipeline (Kinematics + Augmentation + Fine-tuning)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from data.action_dataset import ACTION_CLASSES, build_or_load_penn_action_splits
from experiment_protocol import prepare_penn_splits, protocol_metadata
from models.action_bilstm import ActionBiLSTMAttentionClassifier
from models.spatiotemporal_attention import SpatioTemporalAttentionClassifier

BONE_PAIRS = [
    (0, 2),
    (1, 3),  # shoulders -> elbows
    (2, 4),
    (3, 5),  # elbows -> wrists
    (6, 8),
    (7, 9),  # hips -> knees
    (8, 10),
    (9, 11),  # knees -> ankles
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_kinematic_features_from_pos24(pos24):
    """
    Given pos24 [T, 24]:
    Extracts:
      - pos: [T, 24]
      - vel: [T, 24] (first order temporal difference)
      - bones: [T, 16] (8 bone vectors * 2)
    Returns: [T, 64]
    """
    T = pos24.shape[0]
    coords = pos24.reshape(T, 12, 2)

    # 1. Velocity [T, 24]
    vel = np.zeros_like(pos24)
    if T > 1:
        vel[1:] = pos24[1:] - pos24[:-1]

    # 2. Bone Vectors [T, 16]
    bones = np.zeros((T, len(BONE_PAIRS) * 2), dtype=np.float32)
    for b_idx, (p, c) in enumerate(BONE_PAIRS):
        bones[:, b_idx * 2 : b_idx * 2 + 2] = coords[:, c, :] - coords[:, p, :]

    return np.concatenate([pos24, vel, bones], axis=1).astype(np.float32)


class ExperimentSkeletonDataset(Dataset):
    """
    Flexible dataset supporting:
    - feature_mode: 'baseline_24' or 'kinematic_64'
    - augment: boolean (applies jitter, joint dropout, scale jitter during training)
    """

    def __init__(self, raw_samples, feature_mode="baseline_24", augment=False):
        self.feature_mode = feature_mode
        self.augment = augment
        self.samples = []

        for s in raw_samples:
            pos24 = s["features"]  # [T, 24]
            if feature_mode == "kinematic_64":
                feat = compute_kinematic_features_from_pos24(pos24)
            else:
                feat = pos24.astype(np.float32)

            self.samples.append(
                {
                    "seq_id": s["seq_id"],
                    "features": feat,
                    "pos24": pos24.astype(np.float32),
                    "label": s["label"],
                    "action": s["action"],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feat = item["features"].copy()
        T = feat.shape[0]

        # Apply augmentation if enabled (Train time only)
        if self.augment:
            if self.feature_mode == "baseline_24":
                coords = feat.reshape(T, 12, 2)
                # 1. Jitter noise: N(0, 0.015^2)
                noise = np.random.normal(0.0, 0.015, coords.shape).astype(np.float32)
                coords = coords + noise

                # 2. Scale jitter: s in [0.92, 1.08]
                scale = np.random.uniform(0.92, 1.08)
                coords = coords * scale

                # 3. Random Joint Dropout (5% chance per joint)
                if np.random.rand() < 0.5:
                    drop_j = np.random.randint(0, 12)
                    coords[:, drop_j, :] = 0.0

                feat = coords.reshape(T, 24)

            elif self.feature_mode == "kinematic_64":
                pos24 = item["pos24"].copy()
                coords = pos24.reshape(T, 12, 2)

                # Jitter & Scale on position
                noise = np.random.normal(0.0, 0.012, coords.shape).astype(np.float32)
                coords = (coords + noise) * np.random.uniform(0.94, 1.06)

                if np.random.rand() < 0.4:
                    drop_j = np.random.randint(0, 12)
                    coords[:, drop_j, :] = 0.0

                pos24_aug = coords.reshape(T, 24)
                feat = compute_kinematic_features_from_pos24(pos24_aug)

        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(item["label"], dtype=torch.long),
            item["seq_id"],
        )


def collate_fn(batch):
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    seq_ids = [item[2] for item in batch]

    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels_list)

    return padded_features, lengths, labels, seq_ids


def compute_metrics(y_true, y_pred, num_classes=15):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred)) * 100.0

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    f1s = []
    weights = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()
        weights.append(support)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) * 100.0 if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1, cm


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, lengths, targets, _ in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            logits = model(inputs, lengths=lengths)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    acc, macro_f1, cm = compute_metrics(all_targets, all_preds)
    probs_arr = np.array(all_probs)
    top3_preds = np.argsort(probs_arr, axis=1)[:, -3:]
    top3_acc = float(np.mean([t in top3_preds[i] for i, t in enumerate(all_targets)])) * 100.0

    return {"top1_acc": acc, "top3_acc": top3_acc, "macro_f1": macro_f1, "confusion_matrix": cm}


def run_experiment(
    exp_name,
    epochs=50,
    batch_size=32,
    lr=0.001,
    device="cuda:0",
    seed=42,
    validation_fraction=0.2,
    output_dir="outputs/action_recognition/experiments/protocol_v2",
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("\n==================================================================")
    print(f"RUNNING INDEPENDENT EXPERIMENT: {exp_name.upper()}")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size}")
    print("==================================================================")

    splits = prepare_penn_splits(build_or_load_penn_action_splits(), validation_fraction, seed)

    configurations = {
        "exp1_kinematics": ("kinematic_64", False),
        "exp2_augmentation": ("baseline_24", True),
        "exp3_spatiotemporal": ("baseline_24", False),
        "exp4_combined": ("kinematic_64", True),
    }
    if exp_name not in configurations:
        raise ValueError(f"Unknown experiment: {exp_name}")
    feature_mode, augment = configurations[exp_name]
    train_ds = ExperimentSkeletonDataset(splits["train_gt"], feature_mode, augment=augment)
    validation_ds = ExperimentSkeletonDataset(splits["validation_gt"], feature_mode)
    test_gt_ds = ExperimentSkeletonDataset(splits["test_gt"], feature_mode)
    test_raw_ds = ExperimentSkeletonDataset(splits["test_raw_yolo"], feature_mode)
    test_calib_ds = ExperimentSkeletonDataset(splits["test_calib_yolo"], feature_mode)

    if exp_name == "exp3_spatiotemporal":
        model = SpatioTemporalAttentionClassifier(
            num_joints=12,
            joint_in_dim=2,
            spatial_dim=64,
            spatial_heads=4,
            hidden_size=128,
            num_layers=2,
            num_classes=len(ACTION_CLASSES),
        ).to(device)
    else:
        model = ActionBiLSTMAttentionClassifier(
            input_size=64 if feature_mode == "kinematic_64" else 24,
            hidden_size=128,
            num_layers=2,
            num_classes=len(ACTION_CLASSES),
            attention_heads=4,
        ).to(device)

    validation_loader = DataLoader(
        validation_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_gt_loader = DataLoader(
        test_gt_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_raw_loader = DataLoader(
        test_raw_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_calib_loader = DataLoader(
        test_calib_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = -1.0
    best_epoch = 0
    best_weights_path = os.path.join(out_dir, f"{exp_name}_best.pt")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for inputs, lengths, targets, _ in train_loader:
            inputs, lengths, targets = inputs.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs, lengths=lengths)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Select checkpoints on validation only
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            res_val = evaluate(model, validation_loader, device)
            if res_val["top1_acc"] > best_val_acc:
                best_val_acc = res_val["top1_acc"]
                best_epoch = ep
                torch.save(model.state_dict(), best_weights_path)
            print(
                f"[{exp_name}] Epoch {ep:02d}/{epochs:02d} | Validation GT Acc: {res_val['top1_acc']:.2f}% | F1: {res_val['macro_f1']:.2f}% | Best: {best_val_acc:.2f}%"
            )

    duration = time.time() - t0

    # Load best checkpoint and evaluate on all 3 test modalities
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))

    res_gt = evaluate(model, test_gt_loader, device)
    res_raw = evaluate(model, test_raw_loader, device)
    res_calib = evaluate(model, test_calib_loader, device)

    result_dict = {
        "protocol": protocol_metadata(splits, validation_fraction, seed),
        "best_epoch": best_epoch,
        "best_validation_top1_acc": best_val_acc,
        "experiment": exp_name,
        "epochs": epochs,
        "train_time_seconds": duration,
        "ground_truth_test": {
            "top1_acc": res_gt["top1_acc"],
            "top3_acc": res_gt["top3_acc"],
            "macro_f1": res_gt["macro_f1"],
        },
        "raw_yolo_test": {
            "top1_acc": res_raw["top1_acc"],
            "top3_acc": res_raw["top3_acc"],
            "macro_f1": res_raw["macro_f1"],
        },
        "calibrated_yolo_test": {
            "top1_acc": res_calib["top1_acc"],
            "top3_acc": res_calib["top3_acc"],
            "macro_f1": res_calib["macro_f1"],
        },
    }

    out_json = os.path.join(out_dir, f"{exp_name}_summary.json")
    with open(out_json, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n--- COMPLETED {exp_name.upper()} (Time: {duration:.1f}s) ---")
    print(
        f"  * Ground Truth Test:     Top-1: {res_gt['top1_acc']:.2f}% | Macro-F1: {res_gt['macro_f1']:.2f}%"
    )
    print(
        f"  * Raw YOLO Test:         Top-1: {res_raw['top1_acc']:.2f}% | Macro-F1: {res_raw['macro_f1']:.2f}%"
    )
    print(
        f"  * Calibrated YOLO Test:  Top-1: {res_calib['top1_acc']:.2f}% | Macro-F1: {res_calib['macro_f1']:.2f}%\n"
    )
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="exp1_kinematics",
        choices=["exp1_kinematics", "exp2_augmentation", "exp3_spatiotemporal", "exp4_combined"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument(
        "--output_dir", default="outputs/action_recognition/experiments/protocol_v2"
    )
    args = parser.parse_args()

    run_experiment(
        args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        output_dir=args.output_dir,
    )
