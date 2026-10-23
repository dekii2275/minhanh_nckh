# BÁO CÁO KẾT QUẢ ĐỐI SÁNH TRƯỚC VÀ SAU HIỆU CHỈNH KHUNG XƯƠNG 2D
## Đánh Giá Định Lượng Toàn Diện Trên Tập Test Penn Action (76.298 Khung Hình / 1.068 Chuỗi Video)

---

## 1. TỔNG QUAN NGHIÊN CỨU

Đề tài: **"Đánh giá chất lượng và hiệu chỉnh dữ liệu khung xương 2D phục vụ phân tích chuyển động"**

* **Mô hình suy luận cơ sở (Baseline)**: YOLO11n-Pose (`models/yolo11n-pose.pt`) chạy ở độ phân giải $640 \times 640$.
* **Tập dữ liệu kiểm chứng**: Toàn bộ tập Test chính thức Penn Action Dataset (1.068 video sequences, 76.298 frames, 15 hành động thể thao).
* **Pipeline hiệu chỉnh đề xuất (Proposed Calibration Pipeline)**: Tích hợp 4 module không giám sát (Unsupervised):
  1. *Lọc tin cậy & Nội suy động học PCHIP (Kinematic PCHIP Interpolation)*
  2. *Ràng buộc nhân trắc học chiều dài xương (Anthropometric Bone Constraints)*
  3. *Làm mượt thích ứng vận tốc (One-Euro Adaptive Filter)*
  4. *Quy chuẩn hệ quy chiếu góc nhìn tự động (Viewer-Centric Alignment)*

---

## 2. BẢNG SO SÁNH CHỈ SỐ TOÀN DIỆN (BEFORE VS AFTER CALIBRATION)

Dữ liệu được đo đạc và trích xuất từ [outputs/calibration/ablation_summary.csv](file:///home/dekii2275/minhanh_nckh/outputs/calibration/ablation_summary.csv):

| Nhóm Chỉ Số Đánh Giá | Chỉ Số Cụ Thể | Trước Hiệu Chỉnh *(Raw YOLO11n)* | Sau Hiệu Chỉnh *(Full Calibrated Pipeline)* | Mức Độ Cải Thiện *(Impact)* |
| :--- | :--- | :---: | :---: | :---: |
| **Độ chính xác vị trí** *(Accuracy)* | **Sai số Euclidean trung bình** | **53.42 px** | **32.19 px** | **Giảm 39.7%** (-21.23 px) |
| | **Sai số trung vị (Median Error)** | **35.30 px** | **12.50 px** | **Giảm 64.6%** (-22.80 px) |
| | **Sai số phân vị 90% (P90 Error)** | 120.05 px | 87.45 px | **Giảm 27.2%** (-32.60 px) |
| **Tỷ lệ khớp chuẩn** *(PCK Metrics)* | **PCK@20px** *(sai số $\le$ 20px)* | **22.87%** | **63.21%** | **Tăng gấp 2.76 lần** (+40.34%) |
| | **PCK-torso@0.10** *(ngưỡng 10% thân)* | 0.87% | **13.97%** | **Tăng gấp 16.0 lần** |
| | **PCK-torso@0.20** *(ngưỡng 20% thân)* | 2.72% | **26.39%** | **Tăng gần 10 lần** (+23.67%) |
| **Tính liên tục động học** *(Biomechanical)*| **Độ giật gia tốc (Temporal Jitter)** | **19.72 px** | **9.88 px** | **Giảm 49.9%** (Mượt gấp đôi) |
| | **Biến dạng chiều dài xương ($\sigma_{\text{bone}}$)**| 7.32 px | **5.64 px** | **Giảm 23.0%** (Cố định giải phẫu) |
| **Hiệu năng hệ thống** *(Efficiency)* | **Tốc độ xử lý (Inference + Calib)** | 129.2 FPS *(GPU)* | **~3.750 FPS** *(Post-process CPU)* | Thêm chỉ 20.32s cho 76.298 frames |

---

## 3. BẢNG PHÂN RÃ CHI TIẾT THEO 12 KHỚP GIẢI PHẪU (PER-JOINT BREAKDOWN)

Đo đạc trên tập mẫu đại diện đa hành động đối sánh trực tiếp từng khớp:

| Khớp Giải Phẫu | Sai Số Trước (Raw) | PCK@20px Trước | Sai Số Sau (Calib) | PCK@20px Sau | Mức Giảm Sai Số |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vai trái (`left_shoulder`)** | 48.4 px | 20.5% | **31.4 px** | **65.5%** | **-17.0 px** |
| **Vai phải (`right_shoulder`)** | 46.4 px | 17.9% | **29.5 px** | **64.5%** | **-16.9 px** |
| **Khuỷu tay trái (`left_elbow`)** | 62.6 px | 15.0% | **36.1 px** | **58.6%** | **-26.5 px** |
| **Khuỷu tay phải (`right_elbow`)**| 61.1 px | 15.3% | **35.3 px** | **61.2%** | **-25.8 px** |
| **Cổ tay trái (`left_wrist`)** | 56.1 px | 36.8% | **38.8 px** | **58.8%** | **-17.3 px** |
| **Cổ tay phải (`right_wrist`)** | 51.1 px | 39.0% | **34.1 px** | **62.2%** | **-17.0 px** |
| **Hông trái (`left_hip`)** | 43.1 px | 28.2% | **31.2 px** | **66.6%** | **-12.0 px** |
| **Hông phải (`right_hip`)** | 45.0 px | 21.5% | **34.9 px** | **62.5%** | **-10.1 px** |
| **Đầu gối trái (`left_knee`)** | 56.5 px | 20.1% | **35.9 px** | **60.0%** | **-20.6 px** |
| **Đầu gối phải (`right_knee`)** | 59.9 px | 13.3% | **40.4 px** | **54.5%** | **-19.5 px** |
| **Mắt cá trái (`left_ankle`)** | 70.7 px | 18.0% | **37.6 px** | **60.3%** | **-33.1 px** |
| **Mắt cá phải (`right_ankle`)** | 71.8 px | 14.4% | **37.6 px** | **58.6%** | **-34.3 px** |

---

## 4. BẰNG CHỨNG HÌNH ẢNH TRỰC QUAN (VISUAL ARTIFACTS)

Các bảng ảnh đối sánh 5 cột chuẩn $2400 \times 400$ px (Original $\mid$ Ground Truth $\mid$ Before $\mid$ After $\mid$ Overlay) được tạo tự động tại thư mục [outputs/calibration/figures/](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/):

1. **Chuỗi `jumping_jacks` (Seq 1039)**:
   * [panel_seq1039_jumping_jacks_frame007.jpg](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/panel_seq1039_jumping_jacks_frame007.jpg): Sai số giảm từ **79.4 px xuống 16.4 px** (Giảm **79.3%**).
   * [panel_seq1039_jumping_jacks_frame017.jpg](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/panel_seq1039_jumping_jacks_frame017.jpg): Sai số giảm từ **78.9 px xuống 12.9 px** (Giảm **83.7%**).
2. **Chuỗi `squat` (Seq 1659)**:
   * [panel_seq1659_squat_frame026.jpg](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/panel_seq1659_squat_frame026.jpg): Sai số giảm từ **13.2 px xuống 4.3 px** (Giảm **67.2%**).
   * [panel_seq1659_squat_frame062.jpg](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/panel_seq1659_squat_frame062.jpg): Sai số giảm từ **11.4 px xuống 5.0 px** (Giảm **56.1%**).
3. **Chuỗi `baseball_pitch` (Seq 0001)**:
   * [panel_seq0001_baseball_pitch_frame037.jpg](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/panel_seq0001_baseball_pitch_frame037.jpg): Sai số giảm từ **26.7 px xuống 7.5 px** (Giảm **71.9%**).
   * [panel_seq0001_baseball_pitch_frame090.jpg](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/panel_seq0001_baseball_pitch_frame090.jpg): Sai số giảm từ **52.3 px xuống 7.5 px** (Giảm **85.6%**).
4. **Biểu đồ Động học Quỹ đạo & Vận tốc Khớp**:
   * [trajectory_and_dynamics_comparison.png](file:///home/dekii2275/minhanh_nckh/outputs/calibration/figures/trajectory_and_dynamics_comparison.png): Minh chứng triệt tiêu gai nhiễu gia tốc, ổn định chiều dài xương và làm mượt đường cong vận tốc khớp.

---

## 5. KẾT LUẬN VÀ GIÁ TRỊ ĐÓNG GÓP CHO ĐỀ TÀI (CONTRIBUTIONS)

1. **Hoàn thiện tên đề tài nghiên cứu**: Không chỉ dừng lại ở bước "Đánh giá chất lượng", đề tài đã xây dựng thành công bộ giải pháp "Hiệu chỉnh dữ liệu khung xương 2D" hoạt động thực tế.
2. **Giải quyết triệt để vấn đề chi xa**: Các khớp ngoại vi chuyển động nhanh nhất (mắt cá chân, khuỷu tay) ghi nhận mức giảm sai số cao nhất (giảm $> 25 - 34$ px), đưa độ chính xác PCK@20px của toàn bộ 12 khớp lên ngưỡng đồng đều $\sim 60\%$.
3. **Triệt tiêu 50% độ rung giật gia tốc**: Mang lại giá trị cốt lõi cho các ứng dụng cơ sinh học và phân tích động học thể thao, loại bỏ hiện tượng nhấp nháy/teleportation.
4. **Tính khả thi thực tiễn cao (Zero Ground-Truth Dependency & High FPS)**: Thuật toán hoàn toàn tự thích ứng (unsupervised), chạy hậu xử lý cực nhanh (~3.750 FPS), sẵn sàng đóng gói thành API hoặc thư viện phục vụ sản xuất.
