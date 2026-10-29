# Pose ablation lịch sử và phân biệt với downstream ablation

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn: [ablation_summary.csv](../outputs/calibration/ablation_summary.csv). Đây là bảng pose metrics lịch sử trên official test, không phải ablation accuracy validation mới.

| Module lịch sử | Jitter (px) | Bone std (px) | Median error (px) | PCK@20px (%) |
|---|---:|---:|---:|---:|
| baseline | 19.72 | 7.32 | 35.30 | 22.87 |
| module1_one_euro | 11.74 | 7.02 | 35.22 | 22.75 |
| module2_interpolation | 16.77 | 8.51 | 35.17 | 22.95 |
| module3_bone_constraint | 19.28 | 5.75 | 35.46 | 22.49 |
| module4_left_right | 19.72 | 7.32 | 11.56 | 64.06 |
| module5_full_combined | 9.88 | 5.64 | 12.50 | 63.21 |

Full giảm jitter/bone std so raw nhưng có PCK thấp hơn alignment-only (63,21% so 64,06%) và median error lớn hơn (12,50px so 11,56px). Không suy giảm jitter thành tăng fidelity hoặc action accuracy. Bone std 2D còn phụ thuộc góc nhìn và foreshortening; không coi ràng buộc clip-median là chuẩn cơ sinh học đã xác nhận. One-Euro có thể gây độ trễ; bảng này chưa đo lag nên rút nhận định không làm trễ chuyển động.

Thời gian hậu xử lý batch ghi trong CSV không phải throughput streaming end-to-end: chưa cộng YOLO, I/O, classifier và lookahead. PCHIP dùng điểm tương lai; median tính trên clip.

Downstream ablation Bước 3 giữ cờ alignment, cùng classifier, dữ liệu và feature: A 66,80%, Full 64,43%; không cấu hình calibration nào vượt A về Top-1. Đây là bằng chứng khác loại và khác split, không gộp thành một bảng hiệu quả tổng quát.

Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
