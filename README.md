# Correspondence và calibration cho nhận diện hành động từ skeleton 2D

Dự án kiểm tra YOLO11n-Pose → xử lý skeleton → BiLSTM trên Penn Action. **Ba bước chẩn đoán mới cho thấy gain accuracy chủ yếu đến từ alignment; PCHIP, Bone và One-Euro chưa chứng minh lợi ích Top-1 bổ sung trên checkpoint/validation đã kiểm tra.**

Đọc [báo cáo tổng hợp hiện tại](reports/COMPREHENSIVE_RESEARCH_REPORT.md), [protocol](reports/EXPERIMENT_PROTOCOL.md) và [phạm vi công bố/tái lập](reports/REPRODUCIBILITY.md) trước khi trích dẫn kết quả.

| Thực nghiệm validation | Top-1 (%) | Macro-F1 (%) |
|---|---:|---:|
| Raw YOLO | 44,66 | 38,21 |
| Fixed Swap | 67,59 | 68,21 |
| Heuristic Only | 66,80 | 67,39 |
| Full KineCalib | 64,43 | 64,79 |

Cùng 253 video / 17.747 frame, checkpoint GT-trained BiLSTM 24D Protocol v2 seed0. Alignment production trên raw tăng +22,13 điểm; phần calibration còn lại −2,37 điểm, CI 95% [−5,14; +0,40]. Không train, tuning hoặc chạy official test trong ba bước mới. Checkpoint đã được chọn bằng validation GT này, nên đây là chẩn đoán, chưa phải test độc lập.

| Báo cáo | Phạm vi |
|---|---|
| [Bước 1](reports/LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md) | 30 clip official train; lỗi heuristic, đảo cục bộ và nhầm target |
| [Bước 2](reports/LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md) | Raw / Fixed Swap / Heuristic Only / Full, paired CI |
| [Bước 3](reports/CALIBRATION_ABLATION_STEP3_2026-10-29.md) | Thiết kế 2³ với cùng cờ swap; A/H tái tạo C3/C4 |
| [Bản thảo được sửa](reports/MANUSCRIPT_DRAFT.md) | Nghiên cứu chẩn đoán có đối chứng; chưa tuyên bố SOTA/novelty đã xác nhận |

Trong ablation, A=Alignment Only đạt Top-1 cao nhất 66,80%; F=PCHIP+One-Euro đạt Macro-F1 cao nhất 67,75% nhưng Top-1 66,40%. Bone có point estimate âm ở bốn ngữ cảnh. Tác động phụ thuộc mục tiêu và điều kiện; không kết luận mọi calibration vô ích.

Tại working tree cục bộ, CSV dự đoán, metric, posterior, cờ alignment và hình audit nằm trong `outputs/alignment_step1`, `outputs/alignment_step2`, `outputs/calibration_step3`. Source dùng để đo được lưu nguyên byte trong `outputs/research_snapshot`, kèm hash và hướng dẫn khôi phục. Dataset, checkpoint và cache npz không được commit. Không ghi đè code production hoặc chọn lại thuật toán khi cập nhật tài liệu.

Ở máy có đủ CSV/snapshot cục bộ, kiểm tra bảng không cần dataset/model:

```bash
python outputs/research_snapshot/verify_published_tables.py
```

Các báo cáo phụ giữ riêng dữ liệu lịch sử: [multi-seed](reports/MULTI_SEED_BENCHMARK_REPORT.md), [pose ablation](reports/SKELETON_CALIBRATION_ABLATION_STUDY.md), [J-HMDB split 1](reports/JHMDB_EVALUATION_REPORT.md), [KD](reports/KNOWLEDGE_DISTILLATION_REPORT.md), [movement proxy](reports/AQA_MOVEMENT_PROFILING_REPORT.md). Các số này không thay thế đối chứng alignment, không chứng minh transfer, AQA được thẩm định hoặc tốc độ streaming end-to-end. [Review trước audit](reports/CRITICAL_RESEARCH_REVIEW_2026-10-29.md) được giữ để truy vết câu hỏi nghiên cứu.

Môi trường dự án dùng Python 3.10; dependencies trong `requirements.txt`, kiểm tra phát triển trong `requirements-dev.txt`. Sau khi khôi phục source snapshot vào một bản sao riêng và cung cấp dữ liệu/checkpoint đúng hash, dùng các lệnh tái lập ở report tương ứng. Các lệnh huấn luyện/evaluate test cũ không phải bước bắt buộc để kiểm tra bảng validation đã công bố.

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](reports/REPRODUCIBILITY.md).
