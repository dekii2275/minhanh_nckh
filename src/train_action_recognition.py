"""
Module: train_action_recognition.py
Purpose: Train ActionBiLSTMAttentionClassifier on Penn Action (15 classes) for 50 epochs.
         Evaluates downstream performance on:
           1. Ground Truth Test Split (Upper bound)
           2. Raw YOLO-Pose Test Split (Before calibration)
           3. Calibrated YOLO-Pose Test Split (After calibration)
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append("src")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from models.action_bilstm import ActionBiLSTMAttentionClassifier
from data.action_dataset import (
    PennActionSkeletonDataset,
    collate_skeleton_batch,
    build_or_load_penn_action_splits,
    ACTION_CLASSES,
    ACTION_TO_ID
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred, num_classes=15):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred)) * 100.0

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    f1s = []
    weighted_f1s = []
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
        weighted_f1s.append(f1 * support)

    macro_f1 = float(np.mean(f1s))
    total_support = sum(weights)
    weighted_f1 = float(sum(weighted_f1s) / total_support) if total_support > 0 else 0.0
    return acc, macro_f1, weighted_f1, cm

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, lengths, targets, _ in dataloader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            logits = model(inputs, lengths=lengths)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples
    acc, macro_f1, weighted_f1, cm = compute_metrics(all_targets, all_preds, num_classes=len(ACTION_CLASSES))

    # Top-3 accuracy
    probs_arr = np.array(all_probs)
    top3_preds = np.argsort(probs_arr, axis=1)[:, -3:]
    top3_acc = float(np.mean([t in top3_preds[i] for i, t in enumerate(all_targets)])) * 100.0

    return {
        "loss": avg_loss,
        "top1_acc": acc,
        "top3_acc": top3_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "preds": all_preds,
        "targets": all_targets,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, classes, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Action',
           xlabel='Predicted Action')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, format(val, 'd'),
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=8)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_training_curves(history, output_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Test (GT) Loss', linewidth=2)
    ax1.set_title('Cross-Entropy Loss over 50 Epochs', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy & F1 Curve
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Top-1 Acc (%)', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Test (GT) Top-1 Acc (%)', linewidth=2)
    ax2.plot(epochs, history['val_f1'], 'g--', label='Test (GT) Macro-F1 (%)', linewidth=2)
    ax2.set_title('Accuracy and Macro-F1 over 50 Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Metric (%)', fontsize=11)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/action_recognition")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"==================================================================")
    print(f"TRAINING ACTION RECOGNITION BASELINE (50 EPOCHS)")
    print(f"Device: {device} | Architecture: BiLSTM + Attention (2 layers, 128 hidden)")
    print(f"Classes: 15 Penn Action categories | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"==================================================================\n")

    # 1. Load Data
    splits = build_or_load_penn_action_splits()
    train_dataset = PennActionSkeletonDataset(splits['train_gt'])
    test_gt_dataset = PennActionSkeletonDataset(splits['test_gt'])
    test_raw_dataset = PennActionSkeletonDataset(splits['test_raw_yolo'])
    test_calib_dataset = PennActionSkeletonDataset(splits['test_calib_yolo'])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_skeleton_batch, pin_memory=True
    )
    test_gt_loader = DataLoader(
        test_gt_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_skeleton_batch
    )
    test_raw_loader = DataLoader(
        test_raw_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_skeleton_batch
    )
    test_calib_loader = DataLoader(
        test_calib_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_skeleton_batch
    )

    print(f"Dataset summary:")
    print(f"  * Train sequences (GT):           {len(train_dataset)}")
    print(f"  * Test sequences (GT):            {len(test_gt_dataset)}")
    print(f"  * Test sequences (Raw YOLO):      {len(test_raw_dataset)}")
    print(f"  * Test sequences (Calibrated):    {len(test_calib_dataset)}")

    # 2. Build Model
    model = ActionBiLSTMAttentionClassifier(
        input_size=24, # 12 joints * 2
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(ACTION_CLASSES),
        attention_heads=args.heads,
        dropout=args.dropout,
        bidirectional=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 3. Train Loop
    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(args.output_dir, "best_bilstm_model.pt")

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, lengths, targets, _ in train_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs, lengths=lengths)
            loss = criterion(logits, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += len(targets)

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Evaluate on GT test split
        eval_gt = evaluate_model(model, test_gt_loader, criterion, device)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(eval_gt['loss'])
        history['val_acc'].append(eval_gt['top1_acc'])
        history['val_f1'].append(eval_gt['macro_f1'])

        # Save best model
        if eval_gt['top1_acc'] > best_val_acc:
            best_val_acc = eval_gt['top1_acc']
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
                  f"Test (GT) Acc: {eval_gt['top1_acc']:.2f}% | Test F1: {eval_gt['macro_f1']:.2f}% | "
                  f"Best: {best_val_acc:.2f}% (Epoch {best_epoch})")

    train_duration = time.time() - start_train_time
    print(f"\nTraining completed in {train_duration:.2f}s ({train_duration/60:.2f} mins).")
    print(f"Best GT Test Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch}")

    # Plot training curves
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(history, curves_path)
    print(f"Saved training curves to {curves_path}")

    # 4. Final Evaluation Across All 3 Modalities
    print("\n==================================================================")
    print("FINAL EVALUATION ACROSS ALL 3 TEST MODALITIES (Best Checkpoint)")
    print("==================================================================")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    res_gt = evaluate_model(model, test_gt_loader, criterion, device)
    res_raw = evaluate_model(model, test_raw_loader, criterion, device)
    res_calib = evaluate_model(model, test_calib_loader, criterion, device)

    print(f"\n1. GROUND TRUTH SKELETON (Upper Bound):")
    print(f"   * Top-1 Accuracy:  {res_gt['top1_acc']:.2f}%")
    print(f"   * Top-3 Accuracy:  {res_gt['top3_acc']:.2f}%")
    print(f"   * Macro-F1:        {res_gt['macro_f1']:.2f}%")
    print(f"   * Weighted-F1:     {res_gt['weighted_f1']:.2f}%")

    print(f"\n2. RAW YOLO-POSE SKELETON (Before Calibration):")
    print(f"   * Top-1 Accuracy:  {res_raw['top1_acc']:.2f}%")
    print(f"   * Top-3 Accuracy:  {res_raw['top3_acc']:.2f}%")
    print(f"   * Macro-F1:        {res_raw['macro_f1']:.2f}%")
    print(f"   * Weighted-F1:     {res_raw['weighted_f1']:.2f}%")

    print(f"\n3. CALIBRATED YOLO-POSE SKELETON (After Calibration):")
    print(f"   * Top-1 Accuracy:  {res_calib['top1_acc']:.2f}%")
    print(f"   * Top-3 Accuracy:  {res_calib['top3_acc']:.2f}%")
    print(f"   * Macro-F1:        {res_calib['macro_f1']:.2f}%")
    print(f"   * Weighted-F1:     {res_calib['weighted_f1']:.2f}%")

    # Plot confusion matrices
    plot_confusion_matrix(res_gt['confusion_matrix'], ACTION_CLASSES,
                          f"Penn Action GT (Top-1: {res_gt['top1_acc']:.1f}%)",
                          os.path.join(args.output_dir, "confusion_matrix_gt.png"))

    plot_confusion_matrix(res_raw['confusion_matrix'], ACTION_CLASSES,
                          f"Raw YOLO-Pose (Top-1: {res_raw['top1_acc']:.1f}%)",
                          os.path.join(args.output_dir, "confusion_matrix_raw.png"))

    plot_confusion_matrix(res_calib['confusion_matrix'], ACTION_CLASSES,
                          f"Calibrated Skeleton (Top-1: {res_calib['top1_acc']:.1f}%)",
                          os.path.join(args.output_dir, "confusion_matrix_calib.png"))

    # Save summary json
    summary = {
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_time_seconds": train_duration,
        "ground_truth_test": {
            "top1_acc": res_gt["top1_acc"],
            "top3_acc": res_gt["top3_acc"],
            "macro_f1": res_gt["macro_f1"],
            "weighted_f1": res_gt["weighted_f1"]
        },
        "raw_yolo_test": {
            "top1_acc": res_raw["top1_acc"],
            "top3_acc": res_raw["top3_acc"],
            "macro_f1": res_raw["macro_f1"],
            "weighted_f1": res_raw["weighted_f1"]
        },
        "calibrated_yolo_test": {
            "top1_acc": res_calib["top1_acc"],
            "top3_acc": res_calib["top3_acc"],
            "macro_f1": res_calib["macro_f1"],
            "weighted_f1": res_calib["weighted_f1"]
        },
        "calibration_gain": {
            "top1_acc_gain": res_calib["top1_acc"] - res_raw["top1_acc"],
            "macro_f1_gain": res_calib["macro_f1"] - res_raw["macro_f1"]
        }
    }

    with open(os.path.join(args.output_dir, "action_recognition_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved full results to {args.output_dir}/action_recognition_summary.json")

if __name__ == "__main__":
    main()
