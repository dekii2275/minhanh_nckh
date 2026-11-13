"""Summarize bounded paper-informed midhip normalization experiments."""

import csv

import gt_crop_midhip_refinement as study
import numpy as np

NAMES = {
    "baseline": "Baseline ban đầu",
    "reference": "Torso + confidence/mask — tham chiếu 94,86%",
    "strict_midhip_shoulder": "Midhip–shoulder từng frame, lọc frame/clip tọa độ",
    "median_shoulder": "Midhip + shoulder scale trung vị clip",
    "smooth_midhip": "Torso + làm mượt riêng tâm hông",
    "interpolate3": "Nội suy khoảng thiếu ≤3 frame + torso",
}


def main(out=study.OUT):
    source = study.previous.source
    summary = source.read_json(out / "summary.json")
    records = source.read_json(out / "results.json")
    verified = source.read_json(out / "verification.json")
    diagnostics = source.read_json(out / "transformation_diagnostics.json")
    lines = [
        "# Tối ưu midhip/scale sau mốc 94,86%",
        "",
        "## Nguồn tham khảo và giới hạn xác minh",
        "",
        "[A BiLSTM with Attention Approach Using 2D Skeletons for Autism Activity Recognition](https://doi.org/10.1109/ICCA66035.2025.11431081), Kassir và cộng sự, ICCA 2025. Danh tính bài đã đối chiếu qua [trang tác giả tại Arab Open University](https://web.aou.edu.lb/faculties/computer/Pages/portfolio.aspx?iid=ksamrouth%40aou.edu.lb) và chương trình hội nghị. Chưa truy cập được toàn văn gốc hoặc mã tác giả; không tuyên bố tái lập chính xác bài BiLSTM.",
        "",
        "Bài [GCN trên MMASD năm 2026](https://www.mdpi.com/1424-8220/26/14/4638), phần preprocessing, trực tiếp mô tả midhip–shoulder, bỏ frame thiếu tâm/vai và clip tọa độ trong pipeline riêng của họ. Các ngưỡng confidence 0,30, min-width 5 px, clip ±5, cách suy midhip từ hai hông YOLO và quy tắc missing trong lần thử này do dự án quy định.",
        "",
        "## Thiết lập đã cố định trước khi chạy",
        "",
        "- Cùng YOLO11n-Pose crop/chọn người bằng bbox GT, cùng swap cố định. 1.005 train / 253 validation, 15 lớp, 12 khớp. Không truy cập official test.",
        "- Giữ BiLSTM + attention; mọi phương án mới dùng 48 chiều giống tham chiếu 94,86% (24 tọa độ + 12 confidence + 12 mask), cùng số tham số. Ba seed 0/1/2; 50 epoch, batch32, Adam lr0,001 wd0,0001, cosine scheduler, clip norm1. Chọn epoch accuracy validation cao nhất, ưu tiên epoch sớm khi bằng nhau.",
        "- **Strict midhip–shoulder:** từng frame cần đủ hai hông/hai vai có confidence ≥0,30, shoulder width >5px; căn trung điểm hông, chia shoulder width, clip tọa độ ±5. Frame không đạt: zero tọa độ và metadata trước resampling.",
        "- **Median shoulder:** chỉ đổi scale torso trung vị clip sang shoulder width trung vị các frame hai vai tin cậy; thiếu thì fallback scale torso. Giữ nguyên root và confidence/mask của tham chiếu.",
        "- **Smooth midhip:** chỉ làm mượt root bằng trung bình ±2 frame, giữ nguyên torso scale, tọa độ khớp và uncertainty policy. Không lọc trực tiếp toàn bộ khớp.",
        "- **Interpolate3:** nội suy tuyến tính khoảng thiếu/khớp confidence thấp có hai đầu tin cậy, tối đa 3 frame, rồi tính lại torso normalization. Nối confidence/mask gốc để khớp được nội suy không bị coi là quan sát thật.",
        "- Đã tái tạo chính xác feature tham chiếu ở cả 1.258 video. Giữ các video không có pose; không thay bằng GT keypoint.",
        "",
        "## Kết quả validation",
        "",
        "| Phương án | Accuracy TB ± SD (%) | Macro-F1 TB (%) | Δ so 94,86% (điểm %) | Bootstrap 95% Δ |",
        "|---|---:|---:|---:|---|",
    ]
    methods = summary["methods_vs_reference"]
    for name, r in methods.items():
        lo, hi = r["paired_video_bootstrap_95_pp"]
        lines.append(
            f"| {NAMES[name]} | {r['accuracy_mean_pct']:.2f} ± {r['accuracy_sd_pct']:.2f} | {r['macro_f1_mean_pct']:.2f} | {r['gain_pp']:+.2f} | [{lo:+.2f}; {hi:+.2f}] |"
        )
    lines += [
        "",
        "| Phương án | Seed 0 | Seed 1 | Seed 2 | Epoch chọn |",
        "|---|---:|---:|---:|---|",
    ]
    for name, r in methods.items():
        accuracy = " | ".join(f"{p['accuracy_pct']:.2f}%" for p in r["seeds"])
        epochs = ", ".join(str(p["epoch"]) for p in r["seeds"])
        lines.append(f"| {NAMES[name]} | {accuracy} | {epochs} |")
    lines += ["", "## Ảnh hưởng lên dữ liệu đầu vào", ""]
    for split in ("train", "validation"):
        group = [r for r in diagnostics if r["split"] == split]
        frames = sum(r["frames"] for r in group)
        valid = sum(r["strict_valid_frames"] for r in group)
        filled = sum(r["filled_joint_frames"] for r in group)
        lines.append(
            f"- {split}: strict giữ **{valid}/{frames} frame** ({100 * valid / frames:.2f}%); nội suy thêm **{filled} joint-frame**. Đây là thống kê phép biến đổi, không phải metric pose với GT."
        )
    best = max(
        ("reference", *study.features.METHODS), key=lambda m: methods[m]["accuracy_mean_pct"]
    )
    r = methods[best]
    lo, hi = r["paired_video_bootstrap_95_pp"]
    lines += [
        "",
        "## Diễn giải và giới hạn",
        "",
        f"- Phương án có accuracy TB cao nhất: **{NAMES[best]}**, **{r['accuracy_mean_pct']:.2f}%**, chênh tham chiếu **{r['gain_pp']:+.2f} điểm %**, bootstrap 95% [{lo:+.2f}; {hi:+.2f}].",
        "- Đây là thích ứng độc lập các ý tưởng chuẩn hóa. Tên “strict” không có nghĩa trùng preprocessing của bài Kassir; các chi tiết chưa truy cập được phải được xác nhận từ toàn văn/mã tác giả trước khi tuyên bố tái lập.",
        "- Căn tâm midhip đã có trong cả baseline ban đầu lẫn tham chiếu. Lần thử này kiểm tra cách scale, xử lý frame thiếu, root smoothing và imputation; không gán chênh lệch chung cho việc thêm midhip.",
        "- Bootstrap lấy mẫu lại theo video và trung bình chênh lệch ba seed; các khoảng chưa tính biến thiên seed, lựa chọn checkpoint hoặc hiệu chỉnh nhiều phép thử. Khoảng chứa 0 chưa xác nhận cải thiện rõ, không chứng minh tương đương.",
        "- Validation đã tham gia nhiều vòng phát triển. Chưa đánh giá trên test độc lập. GT bbox vẫn được dùng cho crop/chọn người; xử lý clip trung vị, nội suy và root smoothing dùng frame tương lai.",
        "",
        "## Kiểm chứng",
        "",
        f"- **{verified['reference_preprocessing_matches']} video** tái tạo đúng feature tham chiếu; nạp lại **{verified['runs']} checkpoint**, **{verified['cpu_prediction_matches']} dự đoán CPU khớp**.",
        "- 12 mô hình mới có đủ 50 epoch. Có protocol/hash mã, feature cache, checkpoint, lịch sử, dự đoán và so sánh theo cặp.",
        f"- Artifact: `{out.relative_to(study.ROOT).as_posix()}`.",
        "- Mã: `src/gt_baselines/midhip_refinement_features.py`, `gt_crop_midhip_refinement.py`, `fixed_bilstm_training.py`.",
        "",
    ]
    destination = study.ROOT / "report/03112026_nckh/TOI_UU_MIDHIP_SAU_9486_2026-11-03.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    rows = []
    for method in study.METHODS:
        group = [r for r in records if r["method"] == method]
        for label, action in enumerate(study.base.ACTION_CLASSES):
            pp = [[p for p in r["predictions"] if p["target"] == label] for r in group]
            accuracy = [100 * sum(p["predicted"] == label for p in a) / len(a) for a in pp]
            rows.append(
                {
                    "method": method,
                    "action": action,
                    "validation_videos": len(pp[0]),
                    "accuracy_mean_pct": float(np.mean(accuracy)),
                }
            )
    with (out / "per_class_accuracy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if (out / "ensemble_results.json").exists():
        ensembles = source.read_json(out / "ensemble_results.json")
        with destination.open("a", encoding="utf-8") as f:
            f.write("\n## Phân tích phụ: ensemble ba seed\n\n")
            f.write(
                "Lấy trung bình đều ba vector softmax của ba seed rồi argmax; không học trọng số trên validation. Đã khai báo cùng quy tắc cho mọi phương án trong lúc huấn luyện. Mỗi dự đoán cần chạy ba mô hình: kết quả này khác chi phí với accuracy trung bình của một mô hình qua ba seed. Các khoảng dưới đây so các ensemble cùng ba thành viên.\n\n"
            )
            f.write(
                "| Ensemble | Accuracy (%) | Macro-F1 (%) | Δ so ensemble tham chiếu (điểm %) | Bootstrap 95% Δ |\n|---|---:|---:|---:|---|\n"
            )
            for name, r in ensembles["methods"].items():
                lo, hi = r["paired_video_bootstrap_95_pp"]
                f.write(
                    f"| {NAMES[name]} | {r['accuracy_pct']:.2f} | {r['macro_f1_pct']:.2f} | {r['gain_vs_reference_ensemble_pp']:+.2f} | [{lo:+.2f}; {hi:+.2f}] |\n"
                )
            count = len(ensembles["verification"])
            diff = max(r["max_abs_probability_difference"] for r in ensembles["verification"])
            gpu_diff = max(
                r["max_abs_original_backend_probability_difference"]
                for r in ensembles["verification"]
            )
            same = all(
                r["cpu_ensemble_predicted_labels_match"] for r in ensembles["methods"].values()
            )
            f.write(
                f"\nĐã nạp lại {count} checkpoint thành viên trên CPU và backend gốc để kiểm tra xác suất. Sai khác CPU/GPU tuyệt đối lớn nhất là {diff:.8g}; sai khác backend gốc so xác suất đã lưu là {gpu_diff:.8g}. Nhãn ensemble CPU/GPU khớp cho toàn bộ phương án: {same}. Đây vẫn là phân tích validation thăm dò, chưa có test độc lập.\n"
            )
    print(destination)


if __name__ == "__main__":
    main()
