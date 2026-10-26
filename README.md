# Penn Action & J-HMDB — Skeleton Calibration and Action Recognition

Pipeline đánh giá YOLO11n-Pose, hiệu chỉnh khung xương 2D và nhận diện hành động trên Penn Action và J-HMDB.

**Trạng thái thực nghiệm:** Code Penn Action hiện tách validation từ train để chọn checkpoint; kết quả chạy mới lưu trong `protocol_v2`. Số liệu cũ chưa được chạy lại theo quy trình này. Đọc [quy trình và các sai khác đã xác định trong báo cáo](reports/EXPERIMENT_PROTOCOL.md) trước khi trích dẫn kết quả.

Chạy các lệnh bên dưới từ thư mục gốc repository.

## 1. Cấu trúc thư mục

```
minhanh_nckh/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- experiment.yaml
|   |-- joint_mapping.json
|-- data/
|   |-- raw/
|   |   |-- Penn_Action.tar.gz
|   |   |-- Penn_Action/
|   |-- processed/
|-- models/
|   |-- yolo11n-pose.pt
|-- src/
|   |-- download_dataset.py
|   |-- inspect_dataset.py
|   |-- select_sequences.py
|   |-- run_inference.py
|   |-- evaluate.py
|   |-- visualize.py
|   |-- run_pipeline.py
|   |-- audit_evaluation.py
|   |-- run_full_test_evaluation.py
|   |-- skeleton_calibrator.py
|   |-- run_calibration_ablation.py
|   |-- visualize_calibration_comparison.py
|   |-- data/
|   |   |-- action_dataset.py
|   |-- models/
|   |   |-- action_bilstm.py
|   |   |-- spatiotemporal_attention.py
|   |-- train_action_recognition.py
|   |-- run_action_experiments.py
|   |-- download_jhmdb.py
|   |-- run_jhmdb_pipeline.py
|-- outputs/
|   |-- predictions/
|   |-- metrics/
|   |-- figures/
|   |-- demo/
|   |   |-- pilot_demo.mp4
|   |   |-- pilot_demo.gif
|   |-- full_test/
|   |   |-- figures/
|   |   |-- metrics/
|   |-- calibration/
|   |   |-- figures/
|   |   |-- ablation_summary.csv
|   |-- action_recognition/
|   |   |-- experiments/
|   |   |-- confusion_matrix_*.png
|   |   |-- training_curves.png
|   |-- jhmdb/
|       |-- jhmdb_benchmark_summary.json
|-- reports/
|   |-- COMPREHENSIVE_RESEARCH_REPORT.md
|   |-- JHMDB_EVALUATION_REPORT.md
|   |-- PENN_ACTION_YOLO_PILOT.md
|   |-- PENN_ACTION_EVALUATION_AUDIT.md
|   |-- PENN_ACTION_FULL_TEST_EVALUATION.md
|   |-- SKELETON_CALIBRATION_ABLATION_STUDY.md
|   |-- SKELETON_CALIBRATION_BEFORE_AFTER_REPORT.md
|   |-- ACTION_RECOGNITION_BASELINE_REPORT.md
```

## 2. Hướng dẫn tái lập thực nghiệm

### Cài đặt môi trường Conda
```bash
conda create -n penn_pose python=3.10 -y
conda activate penn_pose
pip install -r requirements.txt
```

### Bước 1: Đánh giá Pose Estimation (YOLO11n-Pose trên Penn Action)
```bash
# Thử nghiệm Pilot nhanh (150 frames)
python src/run_pipeline.py

# Đánh giá toàn diện 100% Tập Test (76.298 frames / 1.068 video sequences)
python src/run_full_test_evaluation.py
```

### Bước 2: Hiệu chỉnh Khung xương 2D (Skeleton Calibration & Ablation Study)
```bash
# Đánh giá triệt tiêu 4 module hiệu chỉnh độc lập và kết hợp
python src/run_calibration_ablation.py

# Trực quan hóa đối sánh 5 cột (GT, Raw YOLO, Smooth, Calibrated, Overlay)
python src/visualize_calibration_comparison.py
```

### Bước 3: Nhận diện Hành động Hạ tầng (BiLSTM Action Recognition & 4 Hướng Cải tiến)
```bash
# Huấn luyện baseline 50 epochs và đối sánh 3 modality (Ground Truth, Raw YOLO, Calibrated YOLO)
python src/train_action_recognition.py

# Thực nghiệm 4 hướng cải tiến độc lập & Domain-Matched Training
python src/run_action_experiments.py
```

### Bước 4: Đánh giá Đa Tập Dữ Liệu trên J-HMDB (Cross-Dataset Generalization)
```bash
# Tải và giải nén J-HMDB (928 videos, 21 classes)
python src/download_jhmdb.py

# Chạy toàn bộ pipeline pose evaluation, calibration và nhận diện hành động trên J-HMDB
python src/run_jhmdb_pipeline.py
```

## 3. Các báo cáo nghiên cứu
* 🌟 **[Báo cáo Tổng hợp Toàn diện Đề tài Nghiên cứu](reports/COMPREHENSIVE_RESEARCH_REPORT.md)** (Tổng hợp bài toán, I/O, dữ liệu, kiểm toán, calibration, nhận diện hành động & khuyến nghị)
* 🚀 **[Báo cáo Thực nghiệm Đối chuẩn Tập Dữ liệu J-HMDB](reports/JHMDB_EVALUATION_REPORT.md)** (Đánh giá tính tổng quát hóa đa tập dữ liệu, 21 hành động, đối sánh Penn Action vs J-HMDB)
* [Báo cáo Thử nghiệm Pilot (142 frames)](reports/PENN_ACTION_YOLO_PILOT.md)
* [Báo cáo Kiểm toán Sai số & Quy ước Nhãn Giải phẫu](reports/PENN_ACTION_EVALUATION_AUDIT.md)
* [Báo cáo Đánh giá Toàn diện 100% Tập Test (76.298 frames)](reports/PENN_ACTION_FULL_TEST_EVALUATION.md)
* [Báo cáo Thử nghiệm Triệt tiêu Hiệu chỉnh Khung xương (Ablation Study)](reports/SKELETON_CALIBRATION_ABLATION_STUDY.md)
* [Báo cáo Đối sánh Toàn diện Trước & Sau Hiệu chỉnh (Before vs After)](reports/SKELETON_CALIBRATION_BEFORE_AFTER_REPORT.md)
* [Báo cáo Nhận diện Hành động BiLSTM + Attention (50 Epochs Downstream Task)](reports/ACTION_RECOGNITION_BASELINE_REPORT.md)

## 4. Phát triển và kiểm tra

```bash
pip install -r requirements-dev.txt
ruff check src tests
ruff format --check src tests
python -m pytest -q
```

Để định dạng code sau khi chỉnh sửa:

```bash
ruff format src tests
```

GitHub Actions chạy lint, kiểm tra định dạng và test trên CPU với Python 3.10.
Các test dùng dữ liệu tổng hợp, không yêu cầu tải dataset hay checkpoint YOLO.
Giữ dataset, cache và checkpoint ngoài Git theo `.gitignore`. Commit source,
cấu hình và tài liệu riêng theo từng thay đổi có thể kiểm tra độc lập.
