"""Vietnamese report for confidence/mask and torso-normalization experiments."""

import csv
from collections import Counter

import gt_crop_confidence_experiment as study
import numpy as np

NAMES = {
    "baseline": "Baseline — tọa độ gốc",
    "mean5": "Trung bình 5 frame (kết quả trước)",
    "confidence_mask": "Tọa độ gốc + confidence/mask",
    "torso": "Chuẩn hóa thân người",
    "torso_confidence_mask": "Chuẩn hóa thân + confidence/mask",
    "capacity_control": "Tọa độ gốc + 24 kênh zero",
    "torso_capacity_control": "Chuẩn hóa thân + 24 kênh zero",
}


def main(out=study.OUT):
    s = study.source.read_json(out / "summary.json")
    records = study.source.read_json(out / "results.json")
    verification = study.source.read_json(out / "verification.json")
    diagnostics = study.source.read_json(out / "normalization_diagnostics.json")
    lines = [
        "# Confidence/mask và chuẩn hóa thân người trên YOLO crop GT → BiLSTM",
        "",
        "## Thiết lập",
        "",
        "- Cùng cache YOLO11n-Pose crop bbox GT + chọn detection bằng IoU GT ≥0,3, swap cố định; 1.005 video train / 253 validation, 15 lớp, 12 khớp. Không truy cập official test.",
        "- Ba seed 0/1/2, 50 epoch, batch 32, cùng BiLSTM + attention; Adam lr 0,001 và weight decay 0,0001; cosine scheduler và clip norm 1. Chọn epoch accuracy validation cao nhất, ưu tiên epoch sớm khi bằng nhau.",
        "- Nhánh confidence/mask giữ nguyên 24 tọa độ chuẩn hóa của baseline, nối 12 confidence và 12 mask tin cậy. Confidence được clip [0,1], bằng 0 cho khớp không có tọa độ hợp lệ; mask yêu cầu confidence ≥0,30, finite x/y>0. Khi resampling 64 bước, mask chỉ bằng 1 nếu cả hai quan sát nguồn đều tin cậy.",
        "- Nhánh torso dùng tâm hông của các frame có hai hông tin cậy; nội suy tuyến tính tâm giữa anchor, giữ anchor gần nhất ở đầu/cuối. Nếu cả clip không có hai hông tin cậy, dùng trọng tâm các khớp thân tin cậy, hoặc các khớp tin cậy khác. Nếu không có khớp tin cậy thì tâm bằng 0.",
        "- Scale của clip là trung vị chiều dài từ trung điểm vai đến trung điểm hông, lấy các frame đủ bốn khớp tin cậy và chiều dài >5 pixel. Nếu thiếu, lần lượt dùng trung vị bề rộng vai, bán kính các khớp tin cậy, cuối cùng 50 pixel. Không dùng bbox GT để tính tâm/scale.",
        "- Tất cả tọa độ khớp dự đoán hợp lệ vẫn được giữ, kể cả confidence thấp; chỉ phép tính tâm/scale dùng confidence. Không điền tọa độ khớp thiếu; đặt zero trước resampling như pipeline cũ. Nhánh kết hợp nối thêm confidence/mask vào tọa độ torso.",
        "- Hai đối chứng 48 chiều nối 24 tọa độ với 24 kênh zero; có cùng số tham số và cùng dạng khởi tạo với các nhánh confidence/mask tương ứng. Tái sử dụng sáu checkpoint baseline/mean5 đã kiểm chứng; huấn luyện mới 15 checkpoint.",
        "- Root interpolation và scale trung vị sử dụng cả clip, gồm frame tương lai; đây là xử lý offline. Không dùng tọa độ GT, visibility GT, nhãn hành động hoặc tham số fit từ thống kê tổng hợp validation để tạo đặc trưng.",
        "",
        "## Kết quả validation",
        "",
        "| Phương án | Chiều | Accuracy TB ± SD (%) | Macro-F1 TB (%) | Δ so baseline (điểm %) | Bootstrap 95% Δ |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, r in s["methods_vs_baseline"].items():
        lo, hi = r["paired_video_bootstrap_95_pp"]
        lines.append(
            f"| {NAMES[name]} | {study.METHODS[name]} | {r['accuracy_mean_pct']:.2f} ± {r['accuracy_sd_pct']:.2f} | {r['macro_f1_mean_pct']:.2f} | {r['gain_pp']:+.2f} | [{lo:+.2f}; {hi:+.2f}] |"
        )
    lines += [
        "",
        "| Phương án | Seed 0 | Seed 1 | Seed 2 | Epoch chọn |",
        "|---|---:|---:|---:|---|",
    ]
    for name, r in s["methods_vs_baseline"].items():
        accuracy = " | ".join(f"{a['accuracy_pct']:.2f}%" for a in r["seeds"])
        epoch = ", ".join(str(a["epoch"]) for a in r["seeds"])
        lines.append(f"| {NAMES[name]} | {accuracy} | {epoch} |")
    lines += ["", "## Đối chứng theo cặp", ""]
    for key, label in [
        ("confidence_vs_capacity_control", "Confidence/mask so với tọa độ gốc + zero"),
        ("combined_vs_torso_capacity_control", "Torso + confidence/mask so với torso + zero"),
        ("combined_vs_confidence", "Thêm chuẩn hóa torso khi đã có confidence/mask"),
        ("combined_vs_torso", "Thêm confidence/mask vào torso (khác số tham số)"),
    ]:
        r = s[key]
        lo, hi = r["paired_video_bootstrap_95_pp"]
        lines.append(
            f"- {label}: **{r['gain_pp']:+.2f} điểm phần trăm**, bootstrap 95% [{lo:+.2f}; {hi:+.2f}]."
        )
    lines += ["", "### So với nhánh trung bình 5 frame đạt 93,81%", ""]
    for name in ("confidence_mask", "torso", "torso_confidence_mask"):
        r = s["methods_vs_mean5"][name]
        lo, hi = r["paired_video_bootstrap_95_pp"]
        lines.append(
            f"- {NAMES[name]}: **{r['gain_pp']:+.2f} điểm phần trăm**, bootstrap 95% [{lo:+.2f}; {hi:+.2f}]."
        )
    lines += ["", "## Chẩn đoán chuẩn hóa", ""]
    stats = {}
    for split in ("train", "validation"):
        group = [r for r in diagnostics if r["split"] == split]
        scales = np.array([r["scale_px"] for r in group])
        stats[split] = {
            "videos": len(group),
            "scale_sources": dict(Counter(r["scale_source"] for r in group)),
            "root_sources": dict(Counter(r["root_source"] for r in group)),
            "scale_median_px": float(np.median(scales)),
            "scale_min_px": float(scales.min()),
            "scale_max_px": float(scales.max()),
        }
        lines.append(
            f"- {split}: nguồn scale {stats[split]['scale_sources']}; nguồn tâm {stats[split]['root_sources']}; trung vị scale **{np.median(scales):.2f} px**."
        )
    study.source.write_json(out / "normalization_summary.json", stats)
    best = max(
        ("confidence_mask", "torso", "torso_confidence_mask"),
        key=lambda m: s["methods_vs_baseline"][m]["accuracy_mean_pct"],
    )
    r = s["methods_vs_baseline"][best]
    lo, hi = r["paired_video_bootstrap_95_pp"]
    lines += [
        "",
        "## Diễn giải và giới hạn",
        "",
        f"- Phương án chính có accuracy TB cao nhất: **{NAMES[best]}**, **{r['accuracy_mean_pct']:.2f}%**, chênh baseline **{r['gain_pp']:+.2f} điểm %**, bootstrap 95% [{lo:+.2f}; {hi:+.2f}].",
        "- Mỗi phương án mới thay đổi các yếu tố đã ghi trong protocol; không coi tăng số tham số là bằng chứng confidence có ích. Các đối chứng zero cần được đọc cùng kết quả chính.",
        "- Bootstrap lấy mẫu lại theo video, trung bình chênh lệch ba seed trên mỗi video. Khoảng chứa 0 chưa xác nhận cải thiện rõ và không chứng minh tương đương. Khoảng chưa tính biến thiên seed, lựa chọn checkpoint hoặc thử nhiều cấu hình.",
        "- Validation đã được dùng nhiều lần để phát triển; các kết quả vẫn mang tính thăm dò. Không dùng kết quả này làm kiểm định test độc lập. Official test từng tham gia phát triển lịch sử.",
        "- Crop và chọn detection vẫn dùng bbox GT mỗi frame. Đây là hiệu chỉnh biểu diễn đầu vào, chưa xác nhận pose chính xác hơn hay hệ tự chọn người tốt hơn.",
        "",
        "## Kiểm chứng và artifact",
        "",
        f"- Đã nạp lại **{verification['runs']} checkpoint** trên CPU; cả **{verification['cpu_prediction_matches']} dự đoán khớp** kết quả lưu.",
        "- 15 mô hình mới có đủ 50 epoch; có protocol, hash mã/nguồn/checkpoint, feature cache, lịch sử train, dự đoán từng video và độ bất định theo cặp.",
        f"- Artifact: `{out.relative_to(study.ROOT).as_posix()}`.",
        "- Mã: `src/gt_baselines/confidence_torso_features.py`, `src/gt_baselines/gt_crop_confidence_experiment.py`.",
        "- Chạy: `.venv/Scripts/python.exe -u src/gt_baselines/gt_crop_confidence_experiment.py --stage all` với thư mục output mới.",
        "",
    ]
    destination = study.ROOT / "report/03112026_nckh/CONFIDENCE_MASK_CHUAN_HOA_TORSO_2026-11-03.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    class_rows = []
    for name in study.METHODS:
        group = [r for r in records if r["method"] == name]
        for label, action in enumerate(study.base.ACTION_CLASSES):
            pp = [[p for p in r["predictions"] if p["target"] == label] for r in group]
            acc = [
                100 * sum(p["predicted"] == label for p in predictions) / len(predictions)
                for predictions in pp
            ]
            class_rows.append(
                {
                    "method": name,
                    "action": action,
                    "validation_videos": len(pp[0]),
                    "accuracy_mean_pct": float(np.mean(acc)),
                }
            )
    with (out / "per_class_accuracy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(class_rows[0]))
        w.writeheader()
        w.writerows(class_rows)
    by_class = {(r["method"], r["action"]): r for r in class_rows}
    changes = []
    for action in study.base.ACTION_CLASSES:
        b = by_class["baseline", action]
        r = by_class[best, action]
        changes.append((r["accuracy_mean_pct"] - b["accuracy_mean_pct"], action, b, r))
    changes.sort(key=lambda row: row[0], reverse=True)
    with destination.open("a", encoding="utf-8") as f:
        f.write("\n## Các lớp thay đổi nhiều nhất ở phương án chính tốt nhất\n\n")
        f.write(
            "| Lớp | Video validation | Baseline (%) | Phương án tốt nhất (%) | Δ điểm % |\n|---|---:|---:|---:|---:|\n"
        )
        for delta, action, b, r in changes[:3] + changes[-3:]:
            f.write(
                f"| {action} | {r['validation_videos']} | {b['accuracy_mean_pct']:.2f} | {r['accuracy_mean_pct']:.2f} | {delta:+.2f} |\n"
            )
        f.write(
            "\nCác lớp có ít video; đây là mô tả thăm dò, chưa kiểm định khác biệt ở từng lớp.\n"
        )
    print(destination)


if __name__ == "__main__":
    main()
