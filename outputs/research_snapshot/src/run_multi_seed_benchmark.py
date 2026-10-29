"""
Module: run_multi_seed_benchmark.py
Purpose: Run multi-seed (seeds 0, 1, 2, 3, 4) Protocol v2 benchmark on Penn Action.
         Covers:
           - Baseline (ActionBiLSTMAttentionClassifier, 24 dims)
           - Exp 1: Kinematic Feature Enrichment (64 dims: Pos + Vel + Bones)
           - Exp 2: Noise-Aware Pose Augmentation (Gaussian jitter, Dropout, Scale)
           - Exp 3: Spatio-Temporal Dual Attention
           - Exp 4: Combined SOTA Pipeline
         Computes Mean +/- Std for all metrics and produces a comprehensive report.
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

import run_action_experiments as exp_runner
import train_action_recognition as baseline_runner
from data.action_dataset import ACTION_CLASSES, build_or_load_penn_action_splits
from experiment_protocol import prepare_penn_splits, protocol_metadata
from models.action_bilstm import ActionBiLSTMAttentionClassifier


def run_baseline_for_seed(
    seed,
    splits_cache,
    epochs=50,
    batch_size=32,
    lr=0.001,
    hidden_size=128,
    num_layers=2,
    heads=4,
    dropout=0.3,
    validation_fraction=0.2,
    device="cuda:0",
    output_dir=None,
):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "action_recognition_summary.json")

    baseline_runner.set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    splits = prepare_penn_splits(splits_cache, validation_fraction, seed)
    train_dataset = baseline_runner.PennActionSkeletonDataset(splits["train_gt"])
    validation_dataset = baseline_runner.PennActionSkeletonDataset(splits["validation_gt"])
    test_gt_dataset = baseline_runner.PennActionSkeletonDataset(splits["test_gt"])
    test_raw_dataset = baseline_runner.PennActionSkeletonDataset(splits["test_raw_yolo"])
    test_calib_dataset = baseline_runner.PennActionSkeletonDataset(splits["test_calib_yolo"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=baseline_runner.collate_skeleton_batch,
        pin_memory=device.type == "cuda",
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=baseline_runner.collate_skeleton_batch,
    )
    test_gt_loader = torch.utils.data.DataLoader(
        test_gt_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=baseline_runner.collate_skeleton_batch,
    )
    test_raw_loader = torch.utils.data.DataLoader(
        test_raw_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=baseline_runner.collate_skeleton_batch,
    )
    test_calib_loader = torch.utils.data.DataLoader(
        test_calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=baseline_runner.collate_skeleton_batch,
    )

    model = ActionBiLSTMAttentionClassifier(
        input_size=24,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(ACTION_CLASSES),
        attention_heads=heads,
        dropout=dropout,
        bidirectional=True,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = -1.0
    best_epoch = 0
    checkpoint_path = os.path.join(output_dir, "best_bilstm_model.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += len(targets)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        eval_val = baseline_runner.evaluate_model(model, validation_loader, criterion, device)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(eval_val["loss"])
        history["val_acc"].append(eval_val["top1_acc"])
        history["val_f1"].append(eval_val["macro_f1"])

        if eval_val["top1_acc"] > best_val_acc:
            best_val_acc = eval_val["top1_acc"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[Baseline | Seed {seed}] Ep {epoch:02d}/{epochs:02d} | "
                f"Train: {epoch_train_acc:.2f}% | Val GT: {eval_val['top1_acc']:.2f}% | "
                f"Best Val: {best_val_acc:.2f}% (Ep {best_epoch})"
            )

    train_duration = time.time() - t0

    # Curves & confusion matrices
    curves_path = os.path.join(output_dir, "training_curves.png")
    baseline_runner.plot_training_curves(history, curves_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    res_gt = baseline_runner.evaluate_model(model, test_gt_loader, criterion, device)
    res_raw = baseline_runner.evaluate_model(model, test_raw_loader, criterion, device)
    res_calib = baseline_runner.evaluate_model(model, test_calib_loader, criterion, device)

    baseline_runner.plot_confusion_matrix(
        res_gt["confusion_matrix"],
        ACTION_CLASSES,
        f"Baseline GT (Seed {seed}) Top-1: {res_gt['top1_acc']:.1f}%",
        os.path.join(output_dir, "confusion_matrix_gt.png"),
    )
    baseline_runner.plot_confusion_matrix(
        res_raw["confusion_matrix"],
        ACTION_CLASSES,
        f"Baseline Raw YOLO (Seed {seed}) Top-1: {res_raw['top1_acc']:.1f}%",
        os.path.join(output_dir, "confusion_matrix_raw.png"),
    )
    baseline_runner.plot_confusion_matrix(
        res_calib["confusion_matrix"],
        ACTION_CLASSES,
        f"Baseline Calibrated (Seed {seed}) Top-1: {res_calib['top1_acc']:.1f}%",
        os.path.join(output_dir, "confusion_matrix_calib.png"),
    )

    summary = {
        "protocol": protocol_metadata(splits, validation_fraction, seed),
        "best_validation_top1_acc": best_val_acc,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "train_time_seconds": train_duration,
        "ground_truth_test": {
            "top1_acc": res_gt["top1_acc"],
            "top3_acc": res_gt["top3_acc"],
            "top5_acc": res_gt["top5_acc"],
            "macro_f1": res_gt["macro_f1"],
            "weighted_f1": res_gt["weighted_f1"],
        },
        "raw_yolo_test": {
            "top1_acc": res_raw["top1_acc"],
            "top3_acc": res_raw["top3_acc"],
            "top5_acc": res_raw["top5_acc"],
            "macro_f1": res_raw["macro_f1"],
            "weighted_f1": res_raw["weighted_f1"],
        },
        "calibrated_yolo_test": {
            "top1_acc": res_calib["top1_acc"],
            "top3_acc": res_calib["top3_acc"],
            "top5_acc": res_calib["top5_acc"],
            "macro_f1": res_calib["macro_f1"],
            "weighted_f1": res_calib["weighted_f1"],
        },
        "calibration_gain": {
            "top1_acc_gain": res_calib["top1_acc"] - res_raw["top1_acc"],
            "top5_acc_gain": res_calib["top5_acc"] - res_raw["top5_acc"],
            "macro_f1_gain": res_calib["macro_f1"] - res_raw["macro_f1"],
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_experiment_for_seed(
    mode,
    seed,
    epochs=50,
    batch_size=32,
    validation_fraction=0.2,
    device="cuda:0",
    output_dir=None,
):
    os.makedirs(output_dir, exist_ok=True)
    res = exp_runner.run_experiment(
        mode,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        validation_fraction=validation_fraction,
        device=device,
        output_dir=output_dir,
    )
    return res


def compute_statistics_for_metric_list(vals):
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [round(float(v), 2) for v in arr],
    }


def aggregate_seed_results(seed_results):
    """
    Given a list of result dicts across seeds for a specific mode:
    Computes Mean +/- Std for all evaluation fields.
    """
    aggregated = {
        "n_seeds": len(seed_results),
        "seeds": [r["protocol"]["seed"] for r in seed_results],
        "validation_gt_top1": compute_statistics_for_metric_list(
            [r["best_validation_top1_acc"] for r in seed_results]
        ),
        "train_time_seconds": compute_statistics_for_metric_list(
            [r["train_time_seconds"] for r in seed_results]
        ),
        "ground_truth_test": {
            "top1_acc": compute_statistics_for_metric_list(
                [r["ground_truth_test"]["top1_acc"] for r in seed_results]
            ),
            "top3_acc": compute_statistics_for_metric_list(
                [r["ground_truth_test"]["top3_acc"] for r in seed_results]
            ),
            "top5_acc": compute_statistics_for_metric_list(
                [r["ground_truth_test"]["top5_acc"] for r in seed_results]
            ),
            "macro_f1": compute_statistics_for_metric_list(
                [r["ground_truth_test"]["macro_f1"] for r in seed_results]
            ),
        },
        "raw_yolo_test": {
            "top1_acc": compute_statistics_for_metric_list(
                [r["raw_yolo_test"]["top1_acc"] for r in seed_results]
            ),
            "top3_acc": compute_statistics_for_metric_list(
                [r["raw_yolo_test"]["top3_acc"] for r in seed_results]
            ),
            "top5_acc": compute_statistics_for_metric_list(
                [r["raw_yolo_test"]["top5_acc"] for r in seed_results]
            ),
            "macro_f1": compute_statistics_for_metric_list(
                [r["raw_yolo_test"]["macro_f1"] for r in seed_results]
            ),
        },
        "calibrated_yolo_test": {
            "top1_acc": compute_statistics_for_metric_list(
                [r["calibrated_yolo_test"]["top1_acc"] for r in seed_results]
            ),
            "top3_acc": compute_statistics_for_metric_list(
                [r["calibrated_yolo_test"]["top3_acc"] for r in seed_results]
            ),
            "top5_acc": compute_statistics_for_metric_list(
                [r["calibrated_yolo_test"]["top5_acc"] for r in seed_results]
            ),
            "macro_f1": compute_statistics_for_metric_list(
                [r["calibrated_yolo_test"]["macro_f1"] for r in seed_results]
            ),
        },
        "calibration_gain": {
            "top1_acc_gain": compute_statistics_for_metric_list(
                [
                    r["calibrated_yolo_test"]["top1_acc"] - r["raw_yolo_test"]["top1_acc"]
                    for r in seed_results
                ]
            ),
            "top5_acc_gain": compute_statistics_for_metric_list(
                [
                    r["calibrated_yolo_test"]["top5_acc"] - r["raw_yolo_test"]["top5_acc"]
                    for r in seed_results
                ]
            ),
            "macro_f1_gain": compute_statistics_for_metric_list(
                [
                    r["calibrated_yolo_test"]["macro_f1"] - r["raw_yolo_test"]["macro_f1"]
                    for r in seed_results
                ]
            ),
        },
    }
    return aggregated


def generate_markdown_report(all_aggregates, seeds, epochs, output_report_path):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# BÁO CÁO THỰC NGHIỆM CHUẨN HÓA ĐA SEED (PROTOCOL V2)")
    md.append(f"**Thời gian thực hiện**: {now_str}  ")
    md.append(f"**Cấu hình**: 50 Epochs | Seeds = {seeds} | Protocol v2 (20% Val split từ Train)")
    md.append("**Phần cứng**: NVIDIA GeForce RTX 3050 Laptop GPU (PyTorch 2.6.0+cu124)\n")
    md.append("---\n")

    md.append("## 1. BẢNG TỔNG HỢP SO SÁNH CHUẨN HÓA (MEAN ± STD)\n")
    md.append(
        "Số liệu thống kê trên 5 seeds độc lập (Test độc lập trên 1.068 video sequences của Penn Action):\n"
    )

    # Table header
    md.append(
        "| Cấu hình Mô hình | Ground Truth Top-1 (%) | Raw YOLO Top-1 (%) | Calibrated YOLO Top-1 (%) | Calibration Gain (Δ Top-1) | Calibrated Top-5 (%) | Calibrated Macro-F1 (%) |"
    )
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    names = {
        "baseline": "Baseline (BiLSTM 24D)",
        "exp1_kinematics": "Exp 1: Kinematics (64D)",
        "exp2_augmentation": "Exp 2: Augmentation",
        "exp3_spatiotemporal": "Exp 3: Spatio-Temporal Attention",
        "exp4_combined": "Exp 4: Combined SOTA",
    }

    for mode in [
        "baseline",
        "exp1_kinematics",
        "exp2_augmentation",
        "exp3_spatiotemporal",
        "exp4_combined",
    ]:
        if mode not in all_aggregates:
            continue
        agg = all_aggregates[mode]
        gt_top1 = f"{agg['ground_truth_test']['top1_acc']['mean']:.2f} ± {agg['ground_truth_test']['top1_acc']['std']:.2f}"
        raw_top1 = f"{agg['raw_yolo_test']['top1_acc']['mean']:.2f} ± {agg['raw_yolo_test']['top1_acc']['std']:.2f}"
        cal_top1 = f"**{agg['calibrated_yolo_test']['top1_acc']['mean']:.2f} ± {agg['calibrated_yolo_test']['top1_acc']['std']:.2f}**"
        gain_top1 = f"+{agg['calibration_gain']['top1_acc_gain']['mean']:.2f} ± {agg['calibration_gain']['top1_acc_gain']['std']:.2f}"
        cal_top5 = f"{agg['calibrated_yolo_test']['top5_acc']['mean']:.2f} ± {agg['calibrated_yolo_test']['top5_acc']['std']:.2f}"
        cal_f1 = f"{agg['calibrated_yolo_test']['macro_f1']['mean']:.2f} ± {agg['calibrated_yolo_test']['macro_f1']['std']:.2f}"

        md.append(
            f"| **{names[mode]}** | {gt_top1} | {raw_top1} | {cal_top1} | **{gain_top1}** | {cal_top5} | {cal_f1} |"
        )

    md.append("\n---\n")

    md.append("## 2. PHÂN TÍCH KẾT QUẢ KHOA HỌC\n")
    md.append("1. **Tính Ổn định và Độc lập (Statistical Robustness)**:")
    md.append(
        f"   * Tất cả các mô hình được huấn luyện trên {len(seeds)} random seeds ({', '.join(map(str, seeds))}) với cách chia validation ngẫu nhiên theo lớp (1.005 train / 253 validation)."
    )
    md.append(
        "   * Checkpoint tốt nhất được lựa chọn nghiêm ngặt theo **Validation GT Top-1 Accuracy**, đảm bảo không xảy ra rò rỉ dữ liệu (data leakage) từ tập test."
    )
    md.append("2. **Đóng góp của Pipeline Hiệu chỉnh KineCalib**:")
    base_gain = all_aggregates["baseline"]["calibration_gain"]["top1_acc_gain"]["mean"]
    md.append(
        f"   * KineCalib giúp phục hồi trung bình **+{base_gain:.2f}% Top-1 Accuracy** trên Baseline (từ {all_aggregates['baseline']['raw_yolo_test']['top1_acc']['mean']:.2f}% lên {all_aggregates['baseline']['calibrated_yolo_test']['top1_acc']['mean']:.2f}%)."
    )
    md.append(
        f"   * Top-5 Accuracy của Calibrated Skeleton đạt **{all_aggregates['baseline']['calibrated_yolo_test']['top5_acc']['mean']:.2f}%**, chứng minh mô hình nắm bắt rất tốt cử động thể thao thực tế."
    )
    md.append("3. **Hiệu quả của các Hướng Cải tiến Độc lập**:")
    for mode in ["exp1_kinematics", "exp2_augmentation", "exp3_spatiotemporal", "exp4_combined"]:
        if mode in all_aggregates:
            m_cal = all_aggregates[mode]["calibrated_yolo_test"]["top1_acc"]["mean"]
            b_cal = all_aggregates["baseline"]["calibrated_yolo_test"]["top1_acc"]["mean"]
            diff = m_cal - b_cal
            sign = "+" if diff >= 0 else ""
            md.append(
                f"   * **{names[mode]}**: Đạt Calibrated Top-1 **{m_cal:.2f}%** ({sign}{diff:.2f}% so với Baseline)."
            )

    md.append("\n---\n")
    md.append("## 3. CHI TIẾT SỐ LIỆU TỪNG SEED (PER-SEED REPRODUCIBILITY TABLE)\n")
    md.append(
        "| Cấu hình | Seed | Val GT Top-1 (%) | Test GT Top-1 (%) | Test Raw Top-1 (%) | Test Calib Top-1 (%) | Δ Gain (%) | Calib Top-5 (%) | Calib F1 (%) |"
    )
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for mode in [
        "baseline",
        "exp1_kinematics",
        "exp2_augmentation",
        "exp3_spatiotemporal",
        "exp4_combined",
    ]:
        if mode not in all_aggregates:
            continue
        agg = all_aggregates[mode]
        for idx, s in enumerate(agg["seeds"]):
            val_acc = agg["validation_gt_top1"]["values"][idx]
            gt_acc = agg["ground_truth_test"]["top1_acc"]["values"][idx]
            raw_acc = agg["raw_yolo_test"]["top1_acc"]["values"][idx]
            cal_acc = agg["calibrated_yolo_test"]["top1_acc"]["values"][idx]
            gain = round(cal_acc - raw_acc, 2)
            top5 = agg["calibrated_yolo_test"]["top5_acc"]["values"][idx]
            f1 = agg["calibrated_yolo_test"]["macro_f1"]["values"][idx]
            md.append(
                f"| {mode} | {s} | {val_acc:.2f} | {gt_acc:.2f} | {raw_acc:.2f} | {cal_acc:.2f} | +{gain:.2f} | {top5:.2f} | {f1:.2f} |"
            )

    md.append("\n---\n")
    md.append(
        "Báo cáo được khởi tạo tự động từ pipeline `run_multi_seed_benchmark.py` chuẩn Protocol v2.\n"
    )

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved comprehensive report to: {output_report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--modes",
        nargs="+",
        type=str,
        default=[
            "baseline",
            "exp1_kinematics",
            "exp2_augmentation",
            "exp3_spatiotemporal",
            "exp4_combined",
        ],
    )
    parser.add_argument(
        "--base_dir", type=str, default="outputs/action_recognition/protocol_v2/multi_seed"
    )
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()

    print("==================================================================")
    print("STARTING MULTI-SEED BENCHMARK (PROTOCOL V2)")
    print(f"Modes: {args.modes}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per run: {args.epochs} | Device: {args.device}")
    print("==================================================================\n")

    splits_cache = build_or_load_penn_action_splits()

    total_runs = len(args.modes) * len(args.seeds)
    current_run = 0
    t_start = time.time()

    all_seed_results = {mode: [] for mode in args.modes}

    for mode in args.modes:
        for seed in args.seeds:
            current_run += 1
            run_dir = os.path.join(args.base_dir, mode, f"seed{seed}")
            summary_name = (
                "action_recognition_summary.json" if mode == "baseline" else f"{mode}_summary.json"
            )
            summary_file = os.path.join(run_dir, summary_name)

            print(f"\n[{current_run}/{total_runs}] >>> Running {mode.upper()} | Seed {seed} <<<")

            if args.skip_existing and os.path.exists(summary_file):
                print(f"Found existing summary at {summary_file}. Skipping execution.")
                with open(summary_file) as f:
                    summary = json.load(f)
                all_seed_results[mode].append(summary)
                continue

            if mode == "baseline":
                summary = run_baseline_for_seed(
                    seed=seed,
                    splits_cache=splits_cache,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    output_dir=run_dir,
                )
            else:
                summary = run_experiment_for_seed(
                    mode=mode,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    output_dir=run_dir,
                )

            all_seed_results[mode].append(summary)

    total_elapsed = time.time() - t_start
    print("\n==================================================================")
    print(
        f"ALL {total_runs} RUNS COMPLETED in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} mins)!"
    )
    print("Aggregating statistics...")
    print("==================================================================")

    all_aggregates = {}
    for mode in args.modes:
        all_aggregates[mode] = aggregate_seed_results(all_seed_results[mode])

    # Save aggregated JSON
    agg_json_path = os.path.join(args.base_dir, "multi_seed_benchmark_summary.json")
    with open(agg_json_path, "w") as f:
        json.dump(all_aggregates, f, indent=2)
    print(f"Saved aggregated JSON to: {agg_json_path}")

    # Generate Markdown Report
    report_path = "reports/MULTI_SEED_BENCHMARK_REPORT.md"
    generate_markdown_report(all_aggregates, args.seeds, args.epochs, report_path)


if __name__ == "__main__":
    main()
