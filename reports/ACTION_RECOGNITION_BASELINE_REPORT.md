# Baseline BiLSTM: kết quả lịch sử và baseline đối chứng mới

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn pre-v2: [action_recognition_summary.json](../outputs/action_recognition/action_recognition_summary.json). Code cũ từng chọn checkpoint theo test GT; các kết quả này không phải test độc lập theo quy trình hiện tại. JSON cũ không có Top-5 nên không bổ sung giá trị suy đoán.

| Đầu vào lịch sử | Top-1 (%) | Macro-F1 (%) |
|---|---:|---:|
| GT | 92.98 | 92.72 |
| Raw | 45.51 | 39.82 |
| Full | 67.23 | 66.31 |

Raw→Full +21,72 điểm trong artifact cũ là khác biệt giữa hai pipeline hoàn chỉnh, không xác định contribution PCHIP/Bone/One-Euro. Không suy ra khôi phục chuyển động chính xác hơn hoặc hiệu quả phổ quát.

Baseline chẩn đoán hiện tại là checkpoint Protocol v2 seed0 GT-trained 24D, train 1.005/validation 253 video. Validation GT tái hiện 94,4664%; YOLO raw 44,66%, Fixed Swap 67,59%, Heuristic Only 66,80%, Full 64,43%. Bước 3 dùng A=Heuristic Only 66,80% làm đối chứng; không có module/tổ hợp nào vượt A về Top-1. Macro-F1 cao nhất ở PCHIP+One-Euro 67,75%, nhưng accuracy thấp hơn A một video. Checkpoint/feature không đổi trong các so sánh này.

Không thay con số test lịch sử bằng số validation mới mà giữ nhãn test. Giới hạn một checkpoint và việc đã chọn bằng validation GT phải đi cùng kết quả.

Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
