"""
Module: run_kd_experiment.py
Purpose: Knowledge Distillation (KD) experiment for Skeleton-based Action Recognition.
         Inspired by Liu et al. (Signal Processing 2024):
         - Teacher model: Pretrained on clean Ground-Truth skeletons.
         - Student model: Trained on noisy / occluded skeletons, guided by:
           1. Ground-truth cross-entropy loss
           2. Soft teacher logits (KL Divergence with temperature)
           3. Latent spatiotemporal feature representation alignment (MSE)
         - Evaluates Student performance on Ground Truth, Raw YOLO, and Calibrated YOLO.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Add src to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.action_dataset import ACTION_CLASSES, build_or_load_penn_action_splits
from experiment_protocol import prepare_penn_splits, protocol_metadata
from models.action_bilstm import ActionBiLSTMAttentionClassifier
from models.kd_loss import KnowledgeDistillationLoss


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairedKDSkeletonDataset(Dataset):
    """
    Dataset that yields paired inputs for Knowledge Distillation:
    - Teacher input: Clean, uncorrupted Ground-Truth skeleton.
    - Student input: Noisy / occluded skeleton (simulating real-world low-quality pose detection).
    """

    def __init__(self, raw_samples, noise_level: float = 0.02, dropout_prob: float = 0.4):
        self.samples = raw_samples
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pos24_clean = item["features"].copy()  # [T, 24]
        T = pos24_clean.shape[0]

        # Student input: inject realistic pose noise (jitter + dropout + scaling)
        coords = pos24_clean.reshape(T, 12, 2).copy()

        # 1. Temporal jitter noise
        jitter = np.random.normal(0.0, self.noise_level, coords.shape).astype(np.float32)
        coords += jitter

        # 2. Scale jitter
        scale = np.random.uniform(0.92, 1.08)
        coords *= scale

        # 3. Occlusion joint dropouts (drop random joint over time)
        if np.random.rand() < self.dropout_prob:
            num_drop = np.random.randint(1, 3)
            drop_joints = np.random.choice(12, size=num_drop, replace=False)
            coords[:, drop_joints, :] = 0.0

        pos24_student = coords.reshape(T, 24).astype(np.float32)

        return (
            torch.tensor(pos24_clean, dtype=torch.float32),
            torch.tensor(pos24_student, dtype=torch.float32),
            torch.tensor(item["label"], dtype=torch.long),
            item["seq_id"],
        )


def collate_kd_batch(batch):
    teacher_feats = [b[0] for b in batch]
    student_feats = [b[1] for b in batch]
    labels = torch.stack([b[2] for b in batch])
    seq_ids = [b[3] for b in batch]

    lengths = torch.tensor([len(f) for f in teacher_feats], dtype=torch.long)
    padded_teacher = pad_sequence(teacher_feats, batch_first=True, padding_value=0.0)
    padded_student = pad_sequence(student_feats, batch_first=True, padding_value=0.0)

    return padded_teacher, padded_student, lengths, labels, seq_ids


class SingleModalSkeletonDataset(Dataset):
    """Standard evaluation dataset for student inference."""

    def __init__(self, raw_samples):
        self.samples = raw_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        features = torch.tensor(item["features"], dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.long)
        return features, label, item["seq_id"]


def collate_single_batch(batch):
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    seq_ids = [item[2] for item in batch]

    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels_list)
    return padded, lengths, labels, seq_ids


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for inputs, lengths, labels, _ in loader:
            inputs, lengths = inputs.to(device), lengths.to(device)
            logits = model(inputs, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    probs_arr = np.array(all_probs)

    acc = float(np.mean(all_preds == all_targets)) * 100.0

    # Macro F1
    num_classes = len(ACTION_CLASSES)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) * 100.0

    # Top-3 and Top-5 accuracy
    k3 = min(3, num_classes)
    top3_preds = np.argsort(probs_arr, axis=1)[:, -k3:]
    top3_acc = float(np.mean([t in top3_preds[i] for i, t in enumerate(all_targets)])) * 100.0

    k5 = min(5, num_classes)
    top5_preds = np.argsort(probs_arr, axis=1)[:, -k5:]
    top5_acc = float(np.mean([t in top5_preds[i] for i, t in enumerate(all_targets)])) * 100.0

    return {
        "top1_acc": acc,
        "top3_acc": top3_acc,
        "top5_acc": top5_acc,
        "macro_f1": macro_f1,
    }


def train_teacher_if_needed(train_samples, val_samples, device, seed=42, epochs=30):
    """Trains a baseline teacher on clean GT skeletons if no checkpoint exists."""
    print("Training clean Teacher model on Ground Truth...")
    teacher = ActionBiLSTMAttentionClassifier(
        input_size=24, hidden_size=128, num_layers=2, num_classes=15
    )
    teacher = teacher.to(device)

    train_loader = DataLoader(
        SingleModalSkeletonDataset(train_samples),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_single_batch,
    )
    val_loader = DataLoader(
        SingleModalSkeletonDataset(val_samples),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_single_batch,
    )

    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        teacher.train()
        for inputs, lengths, labels, _ in train_loader:
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = teacher(inputs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(teacher, val_loader, device)
        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            best_state = teacher.state_dict().copy()

    if best_state is not None:
        teacher.load_state_dict(best_state)
    print(f"Teacher training complete. Best Val Acc: {best_val_acc:.2f}%")
    return teacher


def run_kd_experiment(
    data=None,
    teacher_ckpt=None,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    temperature=3.0,
    alpha_ce=0.4,
    beta_kd=0.4,
    gamma_feat=0.2,
    seed=42,
    out_dir="outputs/action_recognition/kd_experiment",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print("  RUNNING KNOWLEDGE DISTILLATION (KD) EXPERIMENT")
    print(f"  Device: {device} | Seed: {seed} | Epochs: {epochs}")
    print(
        f"  KD Loss Params: T={temperature}, alpha={alpha_ce}, beta={beta_kd}, gamma={gamma_feat}"
    )
    print("=======================================================")

    os.makedirs(out_dir, exist_ok=True)

    if data is None:
        data = build_or_load_penn_action_splits()

    splits = prepare_penn_splits(data, seed=seed)
    train_samples = splits["train_gt"]
    val_samples = splits["validation_gt"]
    print(f"Protocol v2 Split: {len(train_samples)} Train | {len(val_samples)} Validation")

    # 1. Initialize Teacher
    teacher = ActionBiLSTMAttentionClassifier(
        input_size=24, hidden_size=128, num_layers=2, num_classes=15
    )
    teacher = teacher.to(device)

    # Search for candidate teacher checkpoints
    default_candidates = [
        teacher_ckpt,
        "outputs/action_recognition/protocol_v2/multi_seed/baseline/seed0/best_bilstm_model.pt",
        "outputs/action_recognition/best_bilstm_model.pt",
    ]
    loaded = False
    for cand in default_candidates:
        if cand and os.path.exists(cand):
            print(f"Loading pretrained Teacher checkpoint from: {cand}")
            state = torch.load(cand, map_location=device, weights_only=False)
            if "model_state_dict" in state:
                teacher.load_state_dict(state["model_state_dict"])
            else:
                teacher.load_state_dict(state)
            loaded = True
            break

    if not loaded:
        print("No pretrained teacher found. Training teacher on clean GT...")
        teacher = train_teacher_if_needed(train_samples, val_samples, device, seed=seed, epochs=30)

    # Freeze Teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Initialize Student
    student = ActionBiLSTMAttentionClassifier(
        input_size=24, hidden_size=128, num_layers=2, num_classes=15
    )
    student = student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    kd_criterion = KnowledgeDistillationLoss(
        temperature=temperature,
        alpha_ce=alpha_ce,
        beta_kd=beta_kd,
        gamma_feat=gamma_feat,
        feat_loss_type="mse",
    )

    # 3. DataLoaders
    train_loader = DataLoader(
        PairedKDSkeletonDataset(train_samples, noise_level=0.015, dropout_prob=0.35),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_kd_batch,
    )
    val_loader = DataLoader(
        SingleModalSkeletonDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_single_batch,
    )

    best_val_acc = 0.0
    best_student_state = None
    best_epoch = 0
    t0 = time.time()

    print("\nTraining Student via Multi-Level Knowledge Distillation...")
    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = 0.0

        for teacher_inputs, student_inputs, lengths, labels, _ in train_loader:
            teacher_inputs = teacher_inputs.to(device)
            student_inputs = student_inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits, teacher_feats = teacher(
                    teacher_inputs, lengths, return_features=True
                )

            # Student forward
            student_logits, student_feats = student(student_inputs, lengths, return_features=True)

            # Combined KD loss
            loss, _ = kd_criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=labels,
                student_features=student_feats,
                teacher_features=teacher_feats,
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation checkpoint selection
        val_res = evaluate(student, val_loader, device)
        val_acc = val_res["top1_acc"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_student_state = student.state_dict().copy()

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"  Epoch [{epoch:02d}/{epochs:02d}] Loss: {running_loss / len(train_loader):.4f} | "
                f"Val GT Top-1: {val_acc:.2f}% (Best: {best_val_acc:.2f}% @ Ep {best_epoch})"
            )

    train_duration = time.time() - t0
    print(
        f"\nStudent training finished in {train_duration:.1f}s. Restoring best checkpoint from epoch {best_epoch}."
    )
    student.load_state_dict(best_student_state)

    # Save Student checkpoint
    best_ckpt_path = os.path.join(out_dir, "best_student_kd.pt")
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_student_state,
            "best_val_acc": best_val_acc,
            "kd_params": {
                "temperature": temperature,
                "alpha_ce": alpha_ce,
                "beta_kd": beta_kd,
                "gamma_feat": gamma_feat,
            },
        },
        best_ckpt_path,
    )

    # 4. Final Independent Test Evaluation (3 modalities)
    print("\n--- EVALUATING DISTILLED STUDENT ON TEST MODALITIES (1,068 sequences) ---")
    test_gt_loader = DataLoader(
        SingleModalSkeletonDataset(data["test_gt"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_single_batch,
    )
    test_raw_loader = DataLoader(
        SingleModalSkeletonDataset(data["test_raw_yolo"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_single_batch,
    )
    test_calib_loader = DataLoader(
        SingleModalSkeletonDataset(data["test_calib_yolo"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_single_batch,
    )

    res_gt = evaluate(student, test_gt_loader, device)
    res_raw = evaluate(student, test_raw_loader, device)
    res_calib = evaluate(student, test_calib_loader, device)

    # Evaluation of Teacher for comparison
    teacher_res_gt = evaluate(teacher, test_gt_loader, device)
    teacher_res_raw = evaluate(teacher, test_raw_loader, device)
    teacher_res_calib = evaluate(teacher, test_calib_loader, device)

    summary = {
        "experiment": "knowledge_distillation_bilstm",
        "protocol": protocol_metadata(
            splits=splits,
            validation_fraction=0.2,
            seed=seed,
        ),
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "val_best_top1_acc": best_val_acc,
        "training_time_seconds": train_duration,
        "kd_hyperparameters": {
            "temperature": temperature,
            "alpha_ce": alpha_ce,
            "beta_kd": beta_kd,
            "gamma_feat": gamma_feat,
        },
        "student_results": {
            "ground_truth_test": res_gt,
            "raw_yolo_test": res_raw,
            "calibrated_yolo_test": res_calib,
            "kinecalib_gain": {
                "top1_gain": res_calib["top1_acc"] - res_raw["top1_acc"],
                "top5_gain": res_calib["top5_acc"] - res_raw["top5_acc"],
                "f1_gain": res_calib["macro_f1"] - res_raw["macro_f1"],
            },
        },
        "teacher_results": {
            "ground_truth_test": teacher_res_gt,
            "raw_yolo_test": teacher_res_raw,
            "calibrated_yolo_test": teacher_res_calib,
        },
    }

    summary_json_path = os.path.join(out_dir, "kd_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=======================================================")
    print("  KNOWLEDGE DISTILLATION BENCHMARK RESULTS")
    print("=======================================================")
    print("Teacher (Clean GT Trained):")
    print(f"  * GT Test Top-1:         {teacher_res_gt['top1_acc']:.2f}%")
    print(f"  * Raw YOLO Test Top-1:   {teacher_res_raw['top1_acc']:.2f}%")
    print(f"  * Calib YOLO Test Top-1: {teacher_res_calib['top1_acc']:.2f}%")
    print("Student (Distilled with GT Teacher Guidance):")
    print(
        f"  * GT Test Top-1:         {res_gt['top1_acc']:.2f}% | Top-5: {res_gt['top5_acc']:.2f}% | F1: {res_gt['macro_f1']:.2f}%"
    )
    print(
        f"  * Raw YOLO Test Top-1:   {res_raw['top1_acc']:.2f}% | Top-5: {res_raw['top5_acc']:.2f}% | F1: {res_raw['macro_f1']:.2f}%"
    )
    print(
        f"  * Calib YOLO Test Top-1: {res_calib['top1_acc']:.2f}% | Top-5: {res_calib['top5_acc']:.2f}% | F1: {res_calib['macro_f1']:.2f}%"
    )
    print(
        f"  * KineCalib Recovery:    +{res_calib['top1_acc'] - res_raw['top1_acc']:.2f}% Top-1 Accuracy"
    )
    print("=======================================================\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Knowledge Distillation experiment on Penn Action."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha_ce", type=float, default=0.4)
    parser.add_argument("--beta_kd", type=float, default=0.4)
    parser.add_argument("--gamma_feat", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/action_recognition/kd_experiment")
    args = parser.parse_args()

    run_kd_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
        beta_kd=args.beta_kd,
        gamma_feat=args.gamma_feat,
        seed=args.seed,
        teacher_ckpt=args.teacher_ckpt,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
