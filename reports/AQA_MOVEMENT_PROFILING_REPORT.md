# Movement profiler: điểm proxy tín hiệu, chưa phải AQA được thẩm định

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Nguồn: aqa_benchmark_summary.json (`outputs/aqa/aqa_benchmark_summary.json`, chưa công bố trên nhánh này). Artifact ghi 225 clip. Công thức điểm thưởng độ mượt, độ ổn định và tính đều của tín hiệu; smoothing có thể trực tiếp tăng chính các đại lượng mà hàm điểm sử dụng. Chưa có nhãn chất lượng chuyên gia để xác định điểm đo kỹ năng thật hay độ chính xác AQA.

| Đại lượng lưu trong JSON | Giá trị |
|---|---:|
| Proxy raw trung bình | 51.12 |
| Proxy calibrated trung bình | 64.01 |
| Δ điểm proxy | +12.89 |

| Hành động | Số clip | Proxy raw | Proxy calibrated | Δ điểm proxy |
|---|---:|---:|---:|---:|
| baseball_pitch | 15 | 50.49 | 63.56 | 13.07 |
| baseball_swing | 15 | 47.89 | 60.24 | 12.36 |
| bench_press | 15 | 41.16 | 52.12 | 10.96 |
| bowl | 15 | 41.98 | 58.88 | 16.90 |
| clean_and_jerk | 15 | 60.48 | 71.91 | 11.43 |
| golf_swing | 15 | 56.10 | 70.92 | 14.83 |
| jump_rope | 15 | 50.53 | 61.95 | 11.43 |
| jumping_jacks | 15 | 62.11 | 72.25 | 10.14 |
| pullup | 15 | 54.64 | 67.02 | 12.38 |
| pushup | 15 | 43.02 | 55.76 | 12.73 |
| situp | 15 | 48.96 | 61.31 | 12.35 |
| squat | 15 | 52.31 | 62.30 | 10.00 |
| strum_guitar | 15 | 46.27 | 62.42 | 16.15 |
| tennis_forehand | 15 | 55.95 | 70.84 | 14.90 |
| tennis_serve | 15 | 54.86 | 68.61 | 13.75 |

Bảng này được chép từ JSON theo từng lớp, thay bảng cũ có số lớp không khớp artifact. Δ +12,89 là điểm của hàm proxy, không phải cải thiện chất lượng thực hiện động tác của con người. Không chuyển jitter/bone variation 2D thành chẩn đoán sai kỹ thuật, phân bố lực, nguy cơ chấn thương hay khuyến nghị điều trị. Các case study cũ với lời khuyên kỹ thuật cụ thể không có thẩm định chuyên gia được rút khỏi báo cáo.

Bước 3 cho thấy cải thiện đầu vào hoặc độ đều tín hiệu không tự chuyển thành accuracy nhận diện. Điều đó cũng không chứng minh proxy này đo AQA đúng hoặc sai; hai mục tiêu khác nhau. Muốn nghiên cứu AQA cần nhãn chất lượng độc lập, độ tin cậy annotator, validity của thước đo và đánh giá theo môn trước khi đưa ra phản hồi kỹ thuật.

Không chạy hoặc tuning AQA trong ba bước mới. Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
