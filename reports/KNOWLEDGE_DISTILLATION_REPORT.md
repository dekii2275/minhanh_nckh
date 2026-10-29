# Knowledge distillation: kết quả phụ, chưa xác nhận contribution calibration

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn: kd_summary.json (`outputs/action_recognition/kd_experiment/kd_summary.json`, chưa công bố trên nhánh này). Một seed=0, 50 epoch, best validation GT 96,05% ở epoch 37. Student dùng GT với nhiễu/che khuất mô phỏng, không được mô tả là đã train trên YOLO thật nếu code không dùng YOLO train. Artifact lưu CE/logit/feature loss và temperature; chưa tách riêng đóng góp augmentation với distillation.

| Mô hình lịch sử | GT Top-1 (%) | Raw Top-1 (%) | Full Top-1 (%) | Full Macro-F1 (%) |
|---|---:|---:|---:|---:|
| Teacher | 91.67 | 42.04 | 64.89 | 65.03 |
| Student KD | 93.26 | 43.26 | 64.23 | 63.13 |

Student cải thiện point estimate GT và raw so Teacher nhưng Full thấp hơn Teacher (64,23% so 64,89%). Raw→Full +20,97 điểm của Student không chứng minh phối hợp KD và các module calibration là tối ưu: chưa có đối chứng Fixed Swap/alignment-only trên chính Student hoặc nhiều seed và CI ghép cặp.

Bước 2–3 dùng Teacher/baseline GT-trained cố định, không Student. Không dùng kết quả Student để giải thích âm/dương của từng stage trên classifier khác. Chưa kết luận regularization, robustness hay generalization dựa vào một run. Bản này không giữ trích dẫn ngoại vi chưa xác minh làm cơ sở cho tuyên bố contribution.

Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
