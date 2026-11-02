"""Render the frozen validation experiment artifacts into a Vietnamese report."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/alignment_step2"


def main():
    data = json.loads((OUT / "results.json").read_text())
    provenance = json.loads((OUT / "provenance.json").read_text())
    gt = json.loads((OUT / "gt_reproduction_check.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    vids = pd.read_csv(OUT / "per_video_predictions.csv", dtype={"id": str})
    names = {"C1": "Raw", "C2": "Fixed Swap", "C3": "Heuristic Only", "C4": "Full KineCalib"}
    metrics = data["metrics"]
    comps = {c["comparison"]: c for c in data["comparisons"]}
    residual = comps["C4-C3"]
    fixed = comps["C4-C2"]
    if fixed["accuracy_ci95_high_pp"] < 0:
        verdict = "Full KineCalib kém Fixed Swap trên validation này; CI của chênh lệch nằm dưới 0."
    elif fixed["accuracy_ci95_low_pp"] > 0:
        verdict = (
            "Full KineCalib tốt hơn Fixed Swap trên validation này; CI của chênh lệch nằm trên 0."
        )
    else:
        verdict = "Chưa có bằng chứng rõ Full KineCalib tốt hơn Fixed Swap: CI của chênh lệch chứa 0. Điều này không chứng minh hai phương án tương đương."
    if residual["accuracy_ci95_low_pp"] > 0:
        residual_text = "Nhóm hiệu chỉnh tọa độ bổ sung mang lại lợi ích rõ trên validation này khi giữ nguyên quyết định alignment production."
    elif residual["accuracy_ci95_high_pp"] < 0:
        residual_text = "Nhóm hiệu chỉnh tọa độ làm giảm accuracy trên validation này khi giữ nguyên quyết định alignment production."
    else:
        residual_text = "Chênh lệch của nhóm hiệu chỉnh tọa độ có CI chứa 0; chưa có bằng chứng rõ về lợi ích bổ sung trên validation này."
    report = f"""# Bước 2 — Ảnh hưởng Left/Right Alignment đến accuracy, Penn Action validation

**Gain quan sát được chủ yếu đến từ alignment:** C3−C1 = {comps["C3-C1"]["accuracy_delta_pp"]:+.2f} điểm phần trăm; C4−C3 = {residual["accuracy_delta_pp"]:+.2f} điểm phần trăm. {verdict} {residual_text}

Thực nghiệm giữ nguyên một checkpoint BiLSTM 24D GT-trained của Protocol v2, chạy cùng **{data["videos"]} video validation / {data["frames"]:,} frame** cho cả bốn cấu hình. Không train lại, không tuning, không đổi thuật toán alignment, không chọn swap theo GT và không chạy official test.

| Cấu hình | Top-1 Accuracy (%) | Macro-F1 (%) | Cải thiện so Raw (điểm phần trăm) | Đúng / tổng |
|---|---:|---:|---:|---:|
"""
    for c, m in metrics.items():
        report += f"| {c} – {names[c]} | {m['top1_accuracy']:.2f} | {m['macro_f1']:.2f} | {m['improvement_vs_raw_pp']:+.2f} | {m['correct']}/{m['videos']} |\n"
    report += """
| So sánh | Δ Top-1 (điểm phần trăm) | 95% CI cho Δ Top-1 | Δ Macro-F1 (điểm phần trăm) | Sai→đúng | Đúng→sai |
|---|---:|---:|---:|---:|---:|
"""
    for c in data["comparisons"]:
        report += f"| {c['comparison']} | {c['accuracy_delta_pp']:+.2f} | [{c['accuracy_ci95_low_pp']:+.2f}, {c['accuracy_ci95_high_pp']:+.2f}] | {c['macro_f1_delta_pp']:+.2f} | {c['improved_videos']} | {c['harmed_videos']} |\n"
    report += f"""
C2−C1 đo tác động của swap cố định. C3−C1 đo tác động của quyết định alignment production trên raw. **C4−C3 = {residual["accuracy_delta_pp"]:+.2f} điểm phần trăm** đo tác động gộp của xử lý confidence và interpolation PCHIP, bone constraint và One-Euro với **cùng quyết định swap từng video**. C4−C2 so sánh hai lựa chọn hoàn chỉnh; chênh lệch này trộn hiệu ứng policy alignment và calibration, không phải phép tách riêng calibration.

Tổng gain Full−Raw là **{metrics["C4"]["improvement_vs_raw_pp"]:+.2f} điểm phần trăm**. Phân rã theo đường C1→C3→C4: **{comps["C3-C1"]["accuracy_delta_pp"]:+.2f}** từ alignment production trên raw, rồi **{residual["accuracy_delta_pp"]:+.2f}** từ các bước còn lại. Đây là phân rã kết quả của một checkpoint trên mẫu validation cố định, không phải tỷ lệ đóng góp nhân quả độc lập của từng thuật toán. Không có ablation từng module ở bước này.

Thiết lập và kiểm soát:

- Checkpoint: `{provenance["checkpoint"]}`. Chốt seed0 trước khi tính kết quả, không so nhiều seed để chọn mô hình. SHA256: `{provenance["checkpoint_sha256"]}`. Kiến trúc 24D, BiLSTM 2 lớp ×128, hai chiều, attention 4 heads, mean pooling; `eval()`, không cập nhật trọng số.
- Đúng 1.005 ID train và 253 ID validation từ metadata đi kèm checkpoint, giao rỗng. Tái dựng độc lập split phân tầng theo lớp với seed0/fraction=0.2 và khớp hoàn toàn. Tất cả 1.258 ID thuộc official `train==1`. Script không gọi loader/cache có test, không đọc frame/annotation/prediction official test.
- Kiểm tra checkpoint/feature/protocol: tái hiện Top-1 validation GT **{gt["observed_top1"]:.8f}%**, khớp giá trị đã lưu **{gt["expected_top1"]:.8f}%**, checkpoint best epoch {protocol["checkpoint_best_epoch"]}. Chỉ dùng GT joints trong phép kiểm tra này; tuyệt đối không truyền GT joints vào quyết định alignment. Nhãn hành động GT dùng để kiểm tra split, tính metric/CI.
- YOLO11n-Pose sẵn có: imgsz=640, box conf=0.25, CPU, batch16; người được chọn bằng box confidence cao nhất, không chọn theo GT. Không loại clip thiếu detection, chọn nhầm người hay clip khó. Cùng một dự đoán raw được chia thành C1–C4.
- Feature giữ nguyên `normalize_skeleton_sequence`: 12 khớp ×xy =24D, mid-hip centering, shoulder scaling với fallback torso, NaN→0, float32. Cùng thứ tự khớp, toàn bộ chiều dài clip, padding và lengths; batch classifier=32. Raw không thêm confidence threshold ngoài code gốc.
- C1 giữ nguyên raw. C2 swap cố định 6 cặp của 12 khớp classifier. C3 dùng quyết định stage4 tại đầu vào sau stage1–3 của production, sau đó **chỉ áp permutation lên raw**. C4 chạy pipeline mặc định: interpolation → bone → One-Euro → alignment. Quyết định được lấy bằng cách chạy chính class stage4 trên marker ở cổ tay với cột vai production giữ nguyên, không viết lại điều kiện median.
- {data["production_swap_clips"]}/{data["videos"]} clip có quyết định swap production; số còn lại giữ Identity. Kiểm tra trên **mọi video** rằng C4 khớp public `FullCalibrationPipeline.calibrate_sequence(raw)` ở toàn bộ cột xy/conf, và C3/C4 dùng cùng permutation. Kiểm tra synthetic bổ sung nhánh vai dương, âm và clip ≤2 frame.
- Tham số calibration giữ mặc định: confidence=0.35, max gap=15, bone ratio=0.40–1.20, freq=30Hz, min cutoff=1.0, beta=0.05, d cutoff=1.0, lr mode=`auto_viewer_centric`. Hash code calibration và checkpoint không đổi trước/sau run.

CI là **paired bootstrap theo video**, phân tầng theo lớp hành động để giữ nguyên số video từng lớp: 20.000 lần lấy mẫu có hoàn lại, seed=20261029, percentile 2.5–97.5%. Hai cấu hình trong mỗi cặp dùng cùng chỉ số video lấy lại. Đây là CI chẩn đoán có điều kiện trên checkpoint và class composition hiện tại; không bao gồm biến thiên giữa seed/train, không hiệu chỉnh bốn so sánh và không giải quyết sự phụ thuộc giữa các clip có thể cùng nguồn quay. CI được tính cho Δ Top-1; Δ Macro-F1 trong bảng là point estimate.

Checkpoint được chọn dựa trên chính validation GT này; do đó đây là phân tích validation phục vụ quyết định nghiên cứu, **chưa phải bằng chứng test độc lập**. 5 video validation đã xuất hiện trong pilot Bước 1; không có lựa chọn alignment theo kết quả review từng video trong run này. Fixed Swap là cấu hình được yêu cầu đánh giá, không được chọn lại sau khi nhìn accuracy.

Dự đoán từng video được lưu ở [per_video_predictions.csv](../outputs/alignment_step2/per_video_predictions.csv): ID, nhãn/action, bốn dự đoán, cờ đúng/sai, quyết định swap, số frame thiếu detection/nhiều người, và outcome của từng cặp (`improved`, `harmed`, `both_correct`, `both_wrong`). Posterior đủ 15 lớp của bốn cấu hình trong [probabilities.npz](../outputs/alignment_step2/probabilities.npz). Số sai→đúng trừ đúng→sai, chia 253, khớp mọi Δ Top-1 trong bảng.

Một số video cải thiện/bị ảnh hưởng khi thêm calibration (C4 so C3; chỉ minh họa, không chọn mẫu để tính metric):

| ID | Action GT | C3 dự đoán | C4 dự đoán | Kết quả |
|---|---|---|---|---|
"""
    for outcome in ["improved", "harmed"]:
        for row in vids[vids["C4-C3_outcome"] == outcome].head(5).to_dict("records"):
            report += f"| {row['id']} | {row['action']} | {row['C3_predicted_action']} | {row['C4_predicted_action']} | {outcome} |\n"
    report += """
Các artifact còn lại: [configuration_metrics.csv](../outputs/alignment_step2/configuration_metrics.csv), [paired_comparisons.csv](../outputs/alignment_step2/paired_comparisons.csv), [per_class_accuracy.csv](../outputs/alignment_step2/per_class_accuracy.csv), [alignment_decisions.csv](../outputs/alignment_step2/alignment_decisions.csv), [validation_manifest.json](../outputs/alignment_step2/validation_manifest.json), [protocol.json](../outputs/alignment_step2/protocol.json), [provenance.json](../outputs/alignment_step2/provenance.json), [GT reproduction check](../outputs/alignment_step2/gt_reproduction_check.json). Dự đoán YOLO từng frame (17 khớp xy/conf, box được chọn, số người) nằm trong `outputs/alignment_step2/raw_predictions/*.npz`. Không ghi đè artifact thực nghiệm cũ.

Tái lập (CPU; cache raw có fingerprint model/cấu hình/phiên bản được kiểm tra trước khi tái sử dụng):

```bash
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/evaluate_alignment_step2.py
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/report_alignment_step2.py
```

Kiểm chứng: [verification.json](../outputs/alignment_step2/verification.json), [script kiểm chứng độc lập](../src/verify_alignment_step2.py). Các metric được tính lại độc lập từ CSV; posterior C3 khớp C2 trên clip swap và C1 trên clip identity; hash model/code giữ nguyên. [Bộ test kiểm soát](../tests/test_alignment_step2.py) đã qua 4/4 test, gồm trường hợp median raw và production trái dấu.

Code: [evaluate_alignment_step2.py](../src/evaluate_alignment_step2.py), [report_alignment_step2.py](../src/report_alignment_step2.py). Chưa thực hiện Bước 3, chưa sửa alignment và chưa train hoặc chạy official test.
"""
    target = ROOT / "reports/LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md"
    target.write_text(report)
    print(target)


if __name__ == "__main__":
    main()
