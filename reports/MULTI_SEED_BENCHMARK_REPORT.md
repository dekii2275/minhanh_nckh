# Protocol v2 nhiều seed: benchmark lịch sử và giới hạn diễn giải

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn: multi_seed_benchmark_summary.json (`outputs/action_recognition/protocol_v2/multi_seed/multi_seed_benchmark_summary.json`, chưa công bố trên nhánh này). Bảng chép mean ± std trong artifact, seeds 0–4, năm cấu hình huấn luyện (25 runs lịch sử), 50 epoch; test 1.068 clip. Không chạy thêm seed/train/test khi viết lại report.

| Cấu hình lịch sử | GT Top-1 (%) | Raw Top-1 (%) | Full Top-1 (%) | Full−Raw (điểm phần trăm) |
|---|---:|---:|---:|---:|
| baseline | 91.63 ± 0.40 | 45.51 ± 3.16 | 65.47 ± 1.34 | +19.96 |
| exp1_kinematics | 92.15 ± 0.37 | 45.00 ± 3.16 | 68.54 ± 1.45 | +23.54 |
| exp2_augmentation | 93.00 ± 0.56 | 44.53 ± 3.51 | 65.15 ± 1.66 | +20.62 |
| exp3_spatiotemporal | 93.16 ± 0.90 | 44.81 ± 6.44 | 63.56 ± 3.49 | +18.75 |
| exp4_combined | 93.33 ± 0.47 | 46.61 ± 1.50 | 69.83 ± 1.54 | +23.22 |

Cột Full−Raw là hiệu giữa mean đã lưu, không phải CI hoặc phần trăm tương đối. Std giữa seed phản ánh cả khởi tạo và split validation khác nhau, không tự xác nhận độ bất định contribution calibration. Các run chọn checkpoint theo validation GT; điều này không chứng minh lịch sử chọn kiến trúc/phương pháp hoàn toàn độc lập với test.

Điểm 69,83% của `exp4_combined` là kết quả cấu hình kết hợp trong benchmark này, không phải một tuyên bố SOTA. Chưa có cùng input/modality, split, compute và protocol với các phương pháp ngoài repository để xếp hạng. Không gọi `exp4_combined` là “Combined SOTA”.

Raw→Full tăng trên test lịch sử không tách alignment và hiệu chỉnh tọa độ. Bước 2 trên validation seed0 cho Fixed Swap 67,59%, heuristic-only 66,80%, Full 64,43%; Bước 3 không có cấu hình calibration vượt alignment-only Top-1. Các số validation này không thay thế số test trong bảng, cũng chưa chứng minh kết quả nhiều seed đã được phân rã tương tự.

Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
