# Penn Action: kết quả pose test lịch sử

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn: [full_test_summary.json](../outputs/full_test/metrics/full_test_summary.json). Artifact ghi 1068 sequence / 76298 frame; 75286 frame detection và 1012 frame không detection.

| Mapping lịch sử | Mean error (px) | Median (px) | PCK@20px (%) |
|---|---:|---:|---:|
| Direct nominal | 53.42 | 35.30 | 22.67 |
| Diagnostic swapped | 27.28 | 8.27 | 76.48 |

Hai mapping là hai phép đối chiếu khác nhau; không gọi mapping swapped là correspondence oracle đúng ở mọi frame. Bảng thể hiện sai số đo được dưới từng mapping, không chứng minh Penn viewer-centric toàn cục hoặc YOLO luôn đảo tên nhãn. Không đồng nhất PCK trong JSON này với pose ablation CSV: tập điểm hợp lệ/mẫu số có thể khác; ví dụ direct PCK@20px ở đây 22,67%, còn ablation CSV 22,87%.

Không lấy test lịch sử để chọn quy tắc mới ở Bước 1–3. Các phản ví dụ correspondence trên train và so sánh validation mới phải được đọc trước khi diễn giải kết quả này. Chưa có alignment-only downstream test cùng checkpoint làm đối chứng cho đóng góp riêng các module.

Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
