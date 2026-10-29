# J-HMDB split 1: bằng chứng phụ lịch sử

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn [jhmdb_benchmark_summary.json](../outputs/jhmdb/jhmdb_benchmark_summary.json), split 1, 660 train / 268 test, 21 lớp. Đây là huấn luyện/đánh giá trong J-HMDB, không phải transfer classifier Penn→J-HMDB. Chưa xác nhận đủ ba split với quy trình chọn phương pháp độc lập.

| Metric lịch sử | Raw | Calibrated |
|---|---:|---:|
| Mean pose error (px) | 36.87 | 34.15 |
| PCK@20px (%) | 67.41 | 68.07 |
| Jitter (px/frame) | 11.44 | 5.22 |
| Bone std (px) | 6.49 | 5.11 |
| GT-trained Top-1 (%) | 40.30 | 37.69 |
| Calibrated-trained Top-1 (%) | 46.27 | 44.03 |

46,27% là test raw của mô hình calibrated-trained; test calibrated là 44,03%. Cả hai chế độ có calibrated accuracy thấp hơn raw dù jitter giảm. Không diễn giải đây là lợi ích generalization phổ quát của calibration. Ảnh hưởng độ phân giải, motion timing hoặc feature shift là giả thuyết, chưa được thí nghiệm này tách riêng.

Không chạy lại J-HMDB trong Bước 1–3. Các kết quả Penn validation mới không tự động chuyển sang dataset này. Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
