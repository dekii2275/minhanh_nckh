# Bước 2 — Ảnh hưởng Left/Right Alignment đến accuracy, Penn Action validation

**Gain quan sát được chủ yếu đến từ alignment:** C3−C1 = +22.13 điểm phần trăm; C4−C3 = -2.37 điểm phần trăm. Chưa có bằng chứng rõ Full KineCalib tốt hơn Fixed Swap: CI của chênh lệch chứa 0. Điều này không chứng minh hai phương án tương đương. Chênh lệch của nhóm hiệu chỉnh tọa độ có CI chứa 0; chưa có bằng chứng rõ về lợi ích bổ sung trên validation này.

Thực nghiệm giữ nguyên một checkpoint BiLSTM 24D GT-trained của Protocol v2, chạy cùng **253 video validation / 17,747 frame** cho cả bốn cấu hình. Không train lại, không tuning, không đổi thuật toán alignment, không chọn swap theo GT và không chạy official test.

| Cấu hình | Top-1 Accuracy (%) | Macro-F1 (%) | Cải thiện so Raw (điểm phần trăm) | Đúng / tổng |
|---|---:|---:|---:|---:|
| C1 – Raw | 44.66 | 38.21 | +0.00 | 113/253 |
| C2 – Fixed Swap | 67.59 | 68.21 | +22.92 | 171/253 |
| C3 – Heuristic Only | 66.80 | 67.39 | +22.13 | 169/253 |
| C4 – Full KineCalib | 64.43 | 64.79 | +19.76 | 163/253 |

| So sánh | Δ Top-1 (điểm phần trăm) | 95% CI cho Δ Top-1 | Δ Macro-F1 (điểm phần trăm) | Sai→đúng | Đúng→sai |
|---|---:|---:|---:|---:|---:|
| C2-C1 | +22.92 | [+17.00, +28.85] | +30.00 | 75 | 17 |
| C3-C1 | +22.13 | [+17.79, +26.48] | +29.18 | 62 | 6 |
| C4-C3 | -2.37 | [-5.14, +0.40] | -2.60 | 4 | 10 |
| C4-C2 | -3.16 | [-7.11, +0.79] | -3.43 | 11 | 19 |

C2−C1 đo tác động của swap cố định. C3−C1 đo tác động của quyết định alignment production trên raw. **C4−C3 = -2.37 điểm phần trăm** đo tác động gộp của xử lý confidence và interpolation PCHIP, bone constraint và One-Euro với **cùng quyết định swap từng video**. C4−C2 so sánh hai lựa chọn hoàn chỉnh; chênh lệch này trộn hiệu ứng policy alignment và calibration, không phải phép tách riêng calibration.

Tổng gain Full−Raw là **+19.76 điểm phần trăm**. Phân rã theo đường C1→C3→C4: **+22.13** từ alignment production trên raw, rồi **-2.37** từ các bước còn lại. Đây là phân rã kết quả của một checkpoint trên mẫu validation cố định, không phải tỷ lệ đóng góp nhân quả độc lập của từng thuật toán. Không có ablation từng module ở bước này.

Thiết lập và kiểm soát:

- Checkpoint: `outputs/action_recognition/protocol_v2/multi_seed/baseline/seed0/best_bilstm_model.pt`. Chốt seed0 trước khi tính kết quả, không so nhiều seed để chọn mô hình. SHA256: `dc01f24210e57ccb7a52611b0b37fe91a40b35e938718fdb123847267fb74347`. Kiến trúc 24D, BiLSTM 2 lớp ×128, hai chiều, attention 4 heads, mean pooling; `eval()`, không cập nhật trọng số.
- Đúng 1.005 ID train và 253 ID validation từ metadata đi kèm checkpoint, giao rỗng. Tái dựng độc lập split phân tầng theo lớp với seed0/fraction=0.2 và khớp hoàn toàn. Tất cả 1.258 ID thuộc official `train==1`. Script không gọi loader/cache có test, không đọc frame/annotation/prediction official test.
- Kiểm tra checkpoint/feature/protocol: tái hiện Top-1 validation GT **94.46640316%**, khớp giá trị đã lưu **94.46640316%**, checkpoint best epoch 27. Chỉ dùng GT joints trong phép kiểm tra này; tuyệt đối không truyền GT joints vào quyết định alignment. Nhãn hành động GT dùng để kiểm tra split, tính metric/CI.
- YOLO11n-Pose sẵn có: imgsz=640, box conf=0.25, CPU, batch16; người được chọn bằng box confidence cao nhất, không chọn theo GT. Không loại clip thiếu detection, chọn nhầm người hay clip khó. Cùng một dự đoán raw được chia thành C1–C4.
- Feature giữ nguyên `normalize_skeleton_sequence`: 12 khớp ×xy =24D, mid-hip centering, shoulder scaling với fallback torso, NaN→0, float32. Cùng thứ tự khớp, toàn bộ chiều dài clip, padding và lengths; batch classifier=32. Raw không thêm confidence threshold ngoài code gốc.
- C1 giữ nguyên raw. C2 swap cố định 6 cặp của 12 khớp classifier. C3 dùng quyết định stage4 tại đầu vào sau stage1–3 của production, sau đó **chỉ áp permutation lên raw**. C4 chạy pipeline mặc định: interpolation → bone → One-Euro → alignment. Quyết định được lấy bằng cách chạy chính class stage4 trên marker ở cổ tay với cột vai production giữ nguyên, không viết lại điều kiện median.
- 168/253 clip có quyết định swap production; số còn lại giữ Identity. Kiểm tra trên **mọi video** rằng C4 khớp public `FullCalibrationPipeline.calibrate_sequence(raw)` ở toàn bộ cột xy/conf, và C3/C4 dùng cùng permutation. Kiểm tra synthetic bổ sung nhánh vai dương, âm và clip ≤2 frame.
- Tham số calibration giữ mặc định: confidence=0.35, max gap=15, bone ratio=0.40–1.20, freq=30Hz, min cutoff=1.0, beta=0.05, d cutoff=1.0, lr mode=`auto_viewer_centric`. Hash code calibration và checkpoint không đổi trước/sau run.

CI là **paired bootstrap theo video**, phân tầng theo lớp hành động để giữ nguyên số video từng lớp: 20.000 lần lấy mẫu có hoàn lại, seed=20261029, percentile 2.5–97.5%. Hai cấu hình trong mỗi cặp dùng cùng chỉ số video lấy lại. Đây là CI chẩn đoán có điều kiện trên checkpoint và class composition hiện tại; không bao gồm biến thiên giữa seed/train, không hiệu chỉnh bốn so sánh và không giải quyết sự phụ thuộc giữa các clip có thể cùng nguồn quay. CI được tính cho Δ Top-1; Δ Macro-F1 trong bảng là point estimate.

Checkpoint được chọn dựa trên chính validation GT này; do đó đây là phân tích validation phục vụ quyết định nghiên cứu, **chưa phải bằng chứng test độc lập**. 5 video validation đã xuất hiện trong pilot Bước 1; không có lựa chọn alignment theo kết quả review từng video trong run này. Fixed Swap là cấu hình được yêu cầu đánh giá, không được chọn lại sau khi nhìn accuracy.

Dự đoán từng video được lưu ở per_video_predictions.csv (`outputs/alignment_step2/per_video_predictions.csv`, chưa công bố trên nhánh này): ID, nhãn/action, bốn dự đoán, cờ đúng/sai, quyết định swap, số frame thiếu detection/nhiều người, và outcome của từng cặp (`improved`, `harmed`, `both_correct`, `both_wrong`). Posterior đủ 15 lớp của bốn cấu hình (CSV và npz giữ cục bộ) trong probabilities.csv (`outputs/alignment_step2/probabilities.csv`, chưa công bố trên nhánh này). Số sai→đúng trừ đúng→sai, chia 253, khớp mọi Δ Top-1 trong bảng.

Một số video cải thiện/bị ảnh hưởng khi thêm calibration (C4 so C3; chỉ minh họa, không chọn mẫu để tính metric):

| ID | Action GT | C3 dự đoán | C4 dự đoán | Kết quả |
|---|---|---|---|---|
| 0447 | bench_press | baseball_pitch | bench_press | improved |
| 0490 | bowl | tennis_forehand | bowl | improved |
| 1425 | pushup | strum_guitar | pushup | improved |
| 2237 | tennis_serve | tennis_forehand | tennis_serve | improved |
| 0165 | baseball_pitch | baseball_pitch | tennis_forehand | harmed |
| 0299 | baseball_swing | baseball_swing | pullup | harmed |
| 0542 | bowl | bowl | tennis_serve | harmed |
| 0777 | clean_and_jerk | clean_and_jerk | tennis_serve | harmed |
| 1089 | jumping_jacks | jumping_jacks | baseball_swing | harmed |

Các artifact còn lại: configuration_metrics.csv (`outputs/alignment_step2/configuration_metrics.csv`, chưa công bố trên nhánh này), paired_comparisons.csv (`outputs/alignment_step2/paired_comparisons.csv`, chưa công bố trên nhánh này), per_class_accuracy.csv (`outputs/alignment_step2/per_class_accuracy.csv`, chưa công bố trên nhánh này), alignment_decisions.csv (`outputs/alignment_step2/alignment_decisions.csv`, chưa công bố trên nhánh này), validation_manifest.json (`outputs/alignment_step2/validation_manifest.json`, chưa công bố trên nhánh này), protocol.json (`outputs/alignment_step2/protocol.json`, chưa công bố trên nhánh này), provenance.json (`outputs/alignment_step2/provenance.json`, chưa công bố trên nhánh này), GT reproduction check (`outputs/alignment_step2/gt_reproduction_check.json`, chưa công bố trên nhánh này). Dự đoán YOLO từng frame (17 khớp xy/conf, box được chọn, số người) nằm trong `outputs/alignment_step2/raw_predictions/*.npz`. Không ghi đè artifact thực nghiệm cũ.

Tái lập (CPU; cache raw có fingerprint model/cấu hình/phiên bản được kiểm tra trước khi tái sử dụng):

```bash
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/evaluate_alignment_step2.py
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/report_alignment_step2.py
```

Kiểm chứng: verification.json (`outputs/alignment_step2/verification.json`, chưa công bố trên nhánh này), script kiểm chứng độc lập (`outputs/research_snapshot/src/verify_alignment_step2.py`, chưa công bố trên nhánh này). Các metric được tính lại độc lập từ CSV; posterior C3 khớp C2 trên clip swap và C1 trên clip identity; hash model/code giữ nguyên. Bộ test kiểm soát (`outputs/research_snapshot/tests/test_alignment_step2.py`, chưa công bố trên nhánh này) đã qua 4/4 test, gồm trường hợp median raw và production trái dấu.

Code: evaluate_alignment_step2.py (`outputs/research_snapshot/src/evaluate_alignment_step2.py`, chưa công bố trên nhánh này), report_alignment_step2.py (`outputs/research_snapshot/src/report_alignment_step2.py`, chưa công bố trên nhánh này). Sau thực nghiệm này, [Bước 3](CALIBRATION_ABLATION_STEP3_2026-10-29.md) đã bổ sung ablation 2³ với cùng cờ swap. Không sửa alignment, train lại hoặc chạy official test trong các bước chẩn đoán.

Bản cục bộ: code thực nghiệm ở trên là snapshot giữ nguyên byte, không sửa production. Snapshot, CSV, hình, dataset, checkpoint và cache raw npz chưa được đưa lên nhánh tài liệu; xem [phạm vi công bố và khôi phục môi trường](REPRODUCIBILITY.md). Posterior được xuất CSV tại máy để kiểm tra.

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
