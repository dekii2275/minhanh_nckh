# Pilot YOLO–GT lịch sử: phạm vi và giới hạn

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn: [evaluation_summary.json](../outputs/metrics/evaluation_summary.json). Pilot trước đây có 3 clip / 142 frame, 1574 khớp GT hợp lệ; detection 142/142. Mean error 45.06px, median 39.71px, PCK@20px 21.09%. Đây là sai số dưới mapping lúc đó, chưa tách lỗi correspondence, định vị và chọn người.

Confidence pose cao không tự xác nhận đúng GT correspondence. Ba clip không đủ xác nhận quy ước toàn Penn Action hoặc năng lực tổng quát của YOLO. Không dùng các con số pilot để chứng minh KineCalib cải thiện nhận diện, AQA hay tốc độ end-to-end.

Pilot mới Bước 1 sử dụng 30 clip official train, lấy mẫu phân tầng 15 hành động; khác mẫu, mục đích và không gộp với pilot 142 frame này. Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
