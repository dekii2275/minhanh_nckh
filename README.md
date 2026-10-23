# Penn Action — YOLO Pose Evaluation Pilot

Thực nghiệm độc lập đánh giá định lượng độ chính xác của mô hình pose estimation (YOLO11n-Pose) trên tập dữ liệu video chuyển động người Penn Action Dataset.

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
|       |-- experiments/
|       |-- confusion_matrix_*.png
|       |-- training_curves.png
|-- reports/
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

## 3. Các báo cáo nghiên cứu
* [Báo cáo Thử nghiệm Pilot (142 frames)](reports/PENN_ACTION_YOLO_PILOT.md)
* [Báo cáo Kiểm toán Sai số & Quy ước Nhãn Giải phẫu](reports/PENN_ACTION_EVALUATION_AUDIT.md)
* [Báo cáo Đánh giá Toàn diện 100% Tập Test (76.298 frames)](reports/PENN_ACTION_FULL_TEST_EVALUATION.md)
* [Báo cáo Thử nghiệm Triệt tiêu Hiệu chỉnh Khung xương (Ablation Study)](reports/SKELETON_CALIBRATION_ABLATION_STUDY.md)
* [Báo cáo Đối sánh Toàn diện Trước & Sau Hiệu chỉnh (Before vs After)](reports/SKELETON_CALIBRATION_BEFORE_AFTER_REPORT.md)
* [Báo cáo Nhận diện Hành động BiLSTM + Attention (50 Epochs Downstream Task)](reports/ACTION_RECOGNITION_BASELINE_REPORT.md)