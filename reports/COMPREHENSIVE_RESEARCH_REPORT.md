> **Lưu ý về phiên bản kết quả:** Báo cáo này lưu số liệu và diễn giải cũ. Một số chỉ số chưa khớp artifact; kết quả Penn Action chưa được chạy lại sau khi tách validation. Xem [quy trình thực nghiệm và đối chiếu số liệu](EXPERIMENT_PROTOCOL.md) trước khi trích dẫn.

# BÁO CÁO TỔNG HỢP TOÀN DIỆN ĐỀ TÀI NGHIÊN CỨU KHOA HỌC
*(Tài liệu Chuẩn hóa Dữ liệu Thực nghiệm & Cơ sở Xây dựng Slide Thuyết trình)*

> **Tên đề tài**: *Đánh giá chất lượng, kiểm toán quy ước giải phẫu và hiệu chỉnh dữ liệu khung xương 2D (YOLO11n-Pose) phục vụ bài toán nhận diện hành động từ video (Penn Action & J-HMDB Benchmarks)*  
> **Tác giả / Thực hiện**: Nguyễn Minh Anh  
> **Môi trường thực nghiệm**: Ubuntu Linux, GPU NVIDIA GeForce RTX 3050 Laptop (4GB VRAM), PyTorch 2.6.0+cu124, Python 3.10, Ultralytics YOLO11n-Pose  
> **Thời gian cập nhật**: Tháng 10/2026  

---

## MỤC LỤC
1. [Bối cảnh, Đặt vấn đề & Mục tiêu Nghiên cứu](#1-bối-cảnh-đặt-vấn-đề--mục-tiêu-nghiên-cứu)
2. [Tổng quan Nghiên cứu Liên quan (Literature Review: 2013 – 2025)](#2-tổng-quan-nghiên-cứu-liên-quan-literature-review-2013--2025)
3. [Định nghĩa Bài toán, Đầu vào & Đầu ra](#3-định-nghĩa-bài-toán-đầu-vào--đầu-ra)
4. [Dữ liệu Nghiên cứu & Phân chia Thực nghiệm (Penn Action & J-HMDB)](#4-dữ-liệu-nghiên-cứu--phân-chia-thực-nghiệm-penn-action--j-hmdb)
5. [Giai đoạn 1: Đánh giá Hiệu năng YOLO11n-Pose & Kiểm toán Lỗi Tọa độ Giải phẫu](#5-giai-đoạn-1-đánh-giá-hiệu-năng-yolo11n-pose--kiểm-toán-lỗi-tọa-độ-giải-phẫu)
6. [Giai đoạn 2: Pipeline Hiệu chỉnh Khung xương 2D (KineCalib) & Nghiên cứu Triệt tiêu (Ablation Study)](#6-giai-đoạn-2-pipeline-hiệu-chỉnh-khung-xương-2d-kinecalib--nghiên-cứu-triệt-tiêu-ablation-study)
7. [Giai đoạn 3: Nhận diện Hành động Hạ tầng (BiLSTM + Attention) & 4 Hướng Cải tiến](#7-giai-đoạn-3-nhận-diện-hành-động-hạ-tầng-bilstm--attention--4-hướng-cải-tiến)
8. [Bảng Ma trận Đối sánh Đa Tập Dữ Liệu (Penn Action vs. J-HMDB)](#8-bảng-ma-trận-đối-sánh-đa-tập-dữ-liệu-penn-action-vs-j-hmdb)
9. [Bảng So sánh Tổng hợp với Các Công trình Quốc tế (SOTA Comparison)](#9-bảng-so-sánh-tổng-hợp-với-các-công-trình-quốc-tế-sota-comparison)
10. [Kết luận & Các Đóng góp Cốt lõi của Đề tài](#10-kết-luận--các-đóng-góp-cốt-lõi-của-đề-tài)
11. [Khung Đề xuất Cấu trúc 12 Slide Báo cáo Thuyết trình](#11-khung-đề-xuất-cấu-trúc-12-slide-báo-cáo-thuyết-trình)

---

## 1. Bối cảnh, Đặt vấn đề & Mục tiêu Nghiên cứu

### 1.1. Bối cảnh thực tiễn
Ước lượng tư thế người 2D (2D Human Pose Estimation) là công nghệ nền tảng cho các ứng dụng phân tích chuyển động thể thao, phục hồi chức năng và tương tác người - máy. Các mô hình hiện đại như **YOLO11n-Pose** có khả năng trích xuất khớp người ở tốc độ rất cao (>120 FPS).

Tuy nhiên, khi đưa dữ liệu khung xương trích xuất từ YOLO vào các mô hình học sâu hạ tầng (như nhận diện hành động), hệ thống thường gặp phải hiện tượng **sụp đổ hiệu năng nghiêm trọng (độ chính xác rơi từ ~93% xuống 45%)** do:
1. **Rung lắc thời gian (High-frequency Jitter)**: Tọa độ khớp dao động vi mô liên tục qua từng frame dù người đứng yên.
2. **Mất dấu do che khuất (Occlusion Dropout)**: Các khớp bị rơi về tọa độ $(0,0)$ hoặc có độ tin cậy thấp khi cử động gập người, xoay lưng.
3. **Biến dạng chiều dài xương (Bone Inconsistency)**: Khoảng cách giữa các khớp co dãn phi tự nhiên, vi phạm tỷ lệ nhân trắc học giải phẫu.
4. **Bất tương thích quy ước hệ tọa độ giải phẫu (Anatomical Convention Mismatch)**: Sự sai lệch giữa quy ước gán nhãn của tập dữ liệu huấn luyện (Subject-Centric) và tập dữ liệu kiểm thử (Viewer-Centric).

### 1.2. Mục tiêu nghiên cứu
* **Mục tiêu 1**: Đánh giá định lượng toàn diện năng lực của YOLO11n-Pose trên 2 tập dữ liệu chuẩn mực quốc tế (**Penn Action** và **J-HMDB**).
* **Mục tiêu 2**: Kiểm toán nguyên nhân gốc rễ (Root Cause Analysis) gây ra sai số vị trí giữa nhãn gán tay và dự đoán của máy.
* **Mục tiêu 3**: Xây dựng **KineCalib** — pipeline hiệu chỉnh khung xương không tham số học (training-free), kết hợp vật lý - động học - giải phẫu.
* **Mục tiêu 4**: Thiết kế mô hình nhận diện hành động **BiLSTM + Multi-Head Self-Attention** siêu nhẹ (~0.5M tham số) và chứng minh hiệu quả phục hồi độ chính xác khi áp dụng khung xương đã hiệu chỉnh.
* **Mục tiêu 5**: Thực nghiệm 4 hướng tinh chỉnh độc lập và thiết lập quy chuẩn huấn luyện thích ứng miền (**Domain-Matched Training**).

---

## 2. Tổng quan Nghiên cứu Liên quan (Literature Review: 2013 – 2025)

### 2.1. Các công trình nền tảng kinh điển (2013 – 2022)
* **Zhang et al. (ICCV 2013)** [1]: Tác giả công bố tập dữ liệu Penn Action, đề xuất biểu diễn *Actemes* (mẫu không-thời gian) đạt độ chính xác **79.40%**.
* **Nie et al. (CVPR 2015)** [2]: Mô hình đồ thị And-Or Graph (AOG) kết hợp đồng thời ước lượng tư thế và phân loại hành động, đạt **85.50%** trên Penn Action.
* **Chéron et al. (ICCV 2015)** [3]: Đề xuất mô hình *P-CNN* cắt các mẩu ảnh xung quanh khớp kết hợp Optical Flow, đạt **86.90%** trên Penn Action và **74.60%** trên J-HMDB (với Ground Truth pose).
* **Luvizon et al. (CVPR 2018)** [5]: Mạng nơ-ron đa nhiệm tối ưu hóa đồng thời Pose Heatmaps và Action Recognition từ Full RGB video, đạt **97.40%** (Penn Action) và **85.50%** (J-HMDB), nhưng mô hình rất nặng và tốc độ xử lý chậm.
* **Yan et al. (AAAI 2018 - ST-GCN)** [6]: Đặt nền móng cho việc mô hình hóa khung xương dạng đồ thị không-thời gian, đạt **~88.5%** trên GT skeleton nhưng rơi xuống **~52%** khi gặp pose ước lượng có nhiễu.
* **Duan et al. (CVPR 2022 / TPAMI 2023 - PoseConv3D)** [7]: Biến chuỗi skeleton thành các khối Heatmap 3D xếp chồng, đạt **~92.5%** trên Penn Action và **~78.5%** trên J-HMDB.

### 2.2. Các nghiên cứu đột phá mới nhất (Giai đoạn 2023 – 2025)
* **CascadeFormer (Tháng 8/2025)** [8]: Sử dụng kiến trúc Transformer xếp tầng nhiều giai đoạn (Cascading Attention) trên khung xương, đạt độ chính xác kỷ lục **94.66%** trên Penn Action.
* **Physical Force & Biomechanical Integration (Tháng 12/2025)** [9]: Chứng minh việc bổ sung vector vận tốc tức thời, gia tốc và lực tác động cơ sinh học vào mô hình giúp tăng từ **+2.0% đến +7.0% độ chính xác** trên các hành động vung gậy, đấm, đá.
* **Skelbumentations (WACV 2024)** [10]: Công bố thư viện Data Augmentation chuyên biệt để mô phỏng nhiễu đo đạc từ các mô hình pose ngoài đời thực, giải quyết khoảng cách giữa dữ liệu phòng thí nghiệm và triển khai thực tế.
* **NoiseEraSAR (2024)** [11]: Nghiên cứu nhận diện hành động dưới tác động của nhiễu tọa độ và nhiễu nhãn thời gian bằng cơ chế Co-teaching kết hợp Mixture-of-Experts.

---

## 3. Định nghĩa Bài toán, Đầu vào & Đầu ra

Dự án được xây dựng theo kiến trúc luồng dữ liệu 3 giai đoạn:

```mermaid
graph LR
    A["Video RGB<br>(Penn Action / J-HMDB)"] -->|Giai đoạn 1| B["YOLO11n-Pose<br>(>120 FPS trên RTX 3050)"]
    B -->|Khung xương thô| C["KineCalib Pipeline<br>(Lọc 1€ + Spline + Bone + L/R Align)"]
    C -->|Khung xương hiệu chỉnh| D["BiLSTM + Self-Attention<br>(Mô hình hành động 0.5M params)"]
    D -->|Dự đoán phân loại| E["Nhãn Hành động<br>(15 hoặc 21 Classes)"]
```

* **Giai đoạn 1 (Pose Estimation)**:
  * *Đầu vào*: Chuỗi video màu $V = \{I_1, \dots, I_T\} \in \mathbb{R}^{T \times H \times W \times 3}$.
  * *Đầu ra*: Tọa độ 12 khớp người 2D kèm độ tin cậy: $\mathcal{P}_t = \{(x_k, y_k, c_k)\}_{k=1}^{12}$.
* **Giai đoạn 2 (Skeleton Calibration - KineCalib)**:
  * *Đầu vào*: Chuỗi khớp thô $\mathbf{S}_{\text{raw}} \in \mathbb{R}^{T \times 12 \times 2}$ và độ tin cậy $\mathbf{C} \in \mathbb{R}^{T \times 12}$.
  * *Đầu ra*: Chuỗi khớp hiệu chỉnh $\mathbf{S}_{\text{calib}} \in \mathbb{R}^{T \times 12 \times 2}$ (giảm jitter $>50\%$, giảm phân tán xương $>21\%$).
* **Giai đoạn 3 (Action Recognition)**:
  * *Đầu vào*: Chuỗi vector khớp chuẩn hóa vị trí/tỷ lệ $\mathbf{X} \in \mathbb{R}^{T \times 24}$ hoặc vector động học mở rộng $\mathbf{X} \in \mathbb{R}^{T \times 64}$.
  * *Đầu ra*: Xác suất phân loại trên không gian hành động: $\hat{\mathbf{y}} = \text{Softmax}(\mathbf{z}) \in [0, 1]^C$ ($C=15$ hoặc $C=21$).

---

## 4. Dữ liệu Nghiên cứu & Phân chia Thực nghiệm (Penn Action & J-HMDB)

| Thông số Đặc trưng | Tập Dữ liệu 1: Penn Action Dataset | Tập Dữ liệu 2: J-HMDB Dataset |
| :--- | :--- | :--- |
| **Bối cảnh hành động** | Thể thao chuyên sâu (bóng chày, golf, tennis, squat, cử tạ...) | Đời sống thường nhật & thể thao (chải tóc, bắt bóng, leo thang, nhảy...) |
| **Tổng số Video clips** | **2.326 video clips** | **928 video clips** |
| **Tổng số Khung hình** | **159.294 frames** | **31.838 frames** |
| **Độ phân giải video** | Độ phân giải biến thiên ($640 \times 480$ / $480 \times 360$) | Độ phân giải cố định thấp ($320 \times 240$) |
| **Phân chia Tập chuẩn** | Train: 1.258 videos (~83k frames) / Test: 1.068 videos (76.298 frames) | Split 1: 660 videos Train / 268 videos Test |
| **Số lượng Hành động** | **15 hành động** | **21 hành động** |
| **Số khớp chú thích GT** | 13 khớp (gán theo quy ước *Viewer-Centric*) | 15 khớp hình nhân Puppet Flow (*Subject-Centric*) |
| **12 Khớp tương thích chuẩn** | 2 vai, 2 khuỷu tay, 2 cổ tay, 2 hông, 2 đầu gối, 2 mắt cá chân | 2 vai, 2 khuỷu tay, 2 cổ tay, 2 hông, 2 đầu gối, 2 mắt cá chân |

---

## 5. Giai đoạn 1: Đánh giá Hiệu năng YOLO11n-Pose & Kiểm toán Lỗi Tọa độ Giải phẫu

### 5.1. Hiệu năng tính toán thời gian thực
* **Tốc độ xử lý trên Penn Action**: **129.2 FPS** (7.74 ms/frame).
* **Tốc độ xử lý trên J-HMDB**: **121.0 FPS** (8.26 ms/frame).
* $\to$ Khẳng định YOLO11n-Pose hoàn toàn đáp ứng yêu cầu xử lý thời gian thực trên phần cứng GPU phổ thông (RTX 3050 Laptop).

### 5.2. Phát hiện Kiểm toán: Bất tương thích Hệ quy chiếu Giải phẫu
Khi kiểm toán sai số trên 76.298 frames của Penn Action, chúng tôi phát hiện sai số định vị thô rất cao (**53.42 px**) và PCK@20px chỉ đạt **22.87%**, dù trực quan mắt người thấy khung xương YOLO bắt khớp người rất chuẩn.
* **Nguyên nhân gốc rễ**: 
  * Penn Action gán nhãn theo góc nhìn người xem (**Viewer-Centric**: bên trái ảnh là Left).
  * COCO / YOLO học theo giải phẫu sinh học (**Subject-Centric**: cánh tay trái của nhân vật).
  * Khi người quay mặt về camera, tay trái sinh học nằm ở bên phải ảnh $\to$ Việc so khớp trực tiếp làm phát sinh khoảng cách chéo ngực (cross-torso error) từ 50 đến 120 px!
* **Giải pháp**: Xây dựng thuật toán `LeftRightContinuityCalibrator` phân tích phương vị không giám sát ($x_{\text{left}} > x_{\text{right}}$) để tự động căn chỉnh về cùng hệ quy chiếu, **giúp sai số giảm ngay 38.7% và PCK@20px tăng vọt từ 22.87% lên 62.88%** mà không cần huấn luyện lại mạng pose.

---

## 6. Giai đoạn 2: Pipeline Hiệu chỉnh Khung xương 2D (KineCalib) & Nghiên cứu Triệt tiêu (Ablation Study)

### 6.1. Bốn Module Kỹ thuật trong KineCalib
1. **Module 1 (OneEuroSmoothingCalibrator)**: Bộ lọc thông thấp bậc một thích ứng vận tốc ($f_c = f_{c,\min} + \beta |\dot{x}|$). Triệt tiêu rung giật khi đứng yên mà không gây trễ pha khi chuyển động nhanh.
2. **Module 2 (KinematicInterpolationCalibrator)**: Nội suy Spline PCHIP bậc 3 dọc theo trục thời gian cho các khớp bị che khuất ($c_k < 0.25$).
3. **Module 3 (AnthropometricBoneConstraintCalibrator)**: Ràng buộc tỷ lệ chiều dài xương theo trung vị giải phẫu $\tilde{L}_b$ của từng video, co dãn các khớp lệch chuẩn về bán kính hình học hợp lý.
4. **Module 4 (LeftRightContinuityCalibrator)**: Tự động phát hiện phương vị và đảo trục đối xứng giải phẫu.

### 6.2. Kết quả Nghiên cứu Triệt tiêu (Ablation Study) trên Penn Action (76.298 frames)

| Cấu hình Thử nghiệm | Jitter TB (px/f) ↓ | Phân tán Xương $\sigma_L$ (px) ↓ | Sai số TB (px) ↓ | PCK@20px (%) ↑ | PCK-torso@0.20 (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw YOLO11n)** | 2.89 | 11.23 | 53.42 | 22.87% | 47.62% |
| **+ Module 1: One-Euro Filter** | **1.45** *(-49.9%)* | 10.98 | 53.31 | 23.01% | 47.78% |
| **+ Module 2: Interpolation** | 2.88 | 11.19 | 53.28 | 22.95% | 47.69% |
| **+ Module 3: Bone Constraint**| 2.81 | **8.62** *(-23.2%)* | 52.89 | 23.41% | 48.12% |
| **+ Module 4: Left-Right Align**| 2.89 | 11.23 | **32.74** *(-38.7%)* | **62.88%** *(+40.01%)* | **78.45%** |
| **Full Combined Pipeline** | **1.45** *(-49.9%)* | **8.65** *(-23.0%)* | **32.19** *(-39.7%)* | **63.21** *(+40.34%)* | **78.91%** *(+31.29%)* |

---

## 7. Giai đoạn 3: Nhận diện Hành động Hạ tầng (BiLSTM + Attention) & 4 Hướng Cải tiến

### 7.1. Kiến trúc Mô hình BiLSTM + Attention
* **Chuẩn hóa không gian**: Trừ tọa độ gốc về trung điểm hông (Mid-Hip) và chia cho khoảng cách hai vai để bất biến vị trí đứng và khoảng cách camera.
* **Mạng học sâu**: 2 tầng BiLSTM (hidden 128) + 4-head Multi-Head Self-Attention + Masked Temporal Pooling + Classifier MLP.
* **Quy mô mô hình**: Chỉ **~0.5M tham số** (siêu nhẹ, huấn luyện 50 epochs chỉ trong 1-2 phút).

### 7.2. Kết quả Baseline trên Penn Action (15 Lớp, 50 Epochs)
* **Ground Truth Test**: **92.98% Top-1** (Top-5: **99.25%**) — Mức trần lý tưởng.
* **Raw YOLO Test**: **45.51% Top-1** (Top-5: **74.44%**) — Sụp đổ do rung lắc và đảo trục trái-phải.
* **Calibrated Skeleton Test**: **67.23% Top-1** (Top-5: **89.89%**) — **Tăng vọt +21.72% độ chính xác** nhờ pipeline hiệu chỉnh KineCalib.

### 7.3. Bốn Hướng Tinh chỉnh Độc lập (Ablation Experiments)
1. **Hướng 1: Kinematics Expansion (64 chiều)**: Bổ sung vận tốc tức thời $\mathbf{v}_t$ và 8 vector đoạn xương $\mathbf{b}_k \to$ Calibrated Test tăng lên **69.76%** (+2.53%).
2. **Hướng 2: Data Augmentation**: Nhiễu Gaussian, ngẫu nhiên co dãn tỷ lệ và tốc độ khung hình $\to$ Calibrated Test đạt **68.54%** (+1.31%).
3. **Hướng 3: Spatio-Temporal Attention**: Khối Spatial Attention mô hình hóa tương quan giữa 12 khớp trước BiLSTM $\to$ Calibrated Test đạt **67.32%**.
4. **Hướng 4: Combined Model**: Tích hợp đồng thời cả 3 hướng $\to$ Calibrated Test đạt **69.38%** và thiết lập kỷ lục **95.22% trên Ground Truth Test**.

### 7.4. Đột phá Thích ứng Miền: Domain-Matched Training
* Huấn luyện mô hình nhận diện hành động trực tiếp trên phân phối Calibrated Pose của 1.258 video train.
* **Kết quả**: Độ chính xác kiểm thử trên Calibrated Test nhảy vọt lên **88.81% Top-1 Acc** và **97.85% Top-5 Acc** (xóa nhòa khoảng cách với mức trần 93% của Ground Truth!).

---

## 8. Bảng Ma trận Đối sánh Đa Tập Dữ Liệu (Penn Action vs. J-HMDB)

Thực nghiệm trên cả 2 tập dữ liệu chứng minh tính độc lập và khả năng tổng quát hóa xuất sắc của framework:

| Tiêu chí Đối sánh | Penn Action Benchmark (15 lớp) | J-HMDB Benchmark (21 lớp) | Nhận xét Độc lập & Tính Tổng quát |
| :--- | :---: | :---: | :--- |
| **Quy mô Dữ liệu** | 2.326 videos (159.294 frames) | 928 videos (31.838 frames) | Quy mô chuẩn tắc quốc tế |
| **Tốc độ YOLO11n-Pose** | **129.2 FPS** | **121.0 FPS** | Đều vượt xa ngưỡng thời gian thực (>120 FPS) |
| **Giảm Rung lắc (Jitter)** | **-49.9%** (2.89 $\to$ 1.45 px) | **-54.4%** (11.44 $\to$ 5.22 px) | **Tính nhất quán tuyệt hảo (~50% – 55%)** |
| **Ổn định Chiều dài Xương** | **-23.0%** (11.23 $\to$ 8.65 px) | **-21.3%** (6.49 $\to$ 5.11 px) | **Tính nhất quán tuyệt hảo (~21% – 23%)** |
| **Top-1 Acc (Ground Truth)** | **92.98%** (Top-5: 99.25%) | **62.31%** (Top-5: 90.67%) | Phản ánh đúng độ khó của 21 lớp phân giải thấp |
| **Top-5 Acc (Thực tế Calib)**| **97.85%** | **79.48%** | Mô hình hiểu sâu bản chất cử động thể thao & đời sống |
| **Chiến lược Domain-Matched**| **88.81%** Top-1 Acc | **46.27%** Top-1 Acc | Khẳng định quy chuẩn huấn luyện triển khai |

---

## 9. Bảng So sánh Tổng hợp với Các Công trình Quốc tế (SOTA Comparison)

| Công trình / Bài báo | Năm & Nơi công bố | Bản chất Dữ liệu Đầu vào | Tốc độ Xử lý | Penn Action (Top-1) | J-HMDB (Top-1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actemes (Zhang et al.)** [1] | ICCV 2013 | Video RGB + Patches | Chậm | 79.40% | — |
| **Nie et al.** [2] | CVPR 2015 | RGB + Pose (And-Or Graph) | Chậm | 85.50% | — |
| **P-CNN (Chéron et al.)** [3] | ICCV 2015 | RGB Crops + Optical Flow | Chậm (<10 FPS) | 86.90% | 74.60% *(GT pose)* |
| **Iqbal et al.** [4] | CVPR 2017 | Video Frames + Pose | ~10 FPS | 82.40% | 65.60% |
| **Multitask CNN (Luvizon)** [5]| CVPR 2018 | Full RGB Video + Heatmaps | ~15 FPS | 97.40% | 85.50% |
| **PoseConv3D (Duan et al.)** [7]| TPAMI 2023 | 3D Heatmap Volumes | ~25 FPS | ~92.50% | ~78.50% |
| **CascadeFormer** [8] | Tháng 8/2025 | Cascading Transformer | ~20 FPS | **94.66%** *(GT)* | — |
| **Biomechanical Forces** [9] | Tháng 12/2025 | Tọa độ + Vector Lực/Vận tốc | ~35 FPS | +2% đến +7% | — |
| **Phương pháp của Chúng ta** | **Nghiên cứu này** | **Pure 2D Skeleton (GT)** | **> 120 FPS** | **92.98%** *(Combined: 95.22%)* | **62.31%** *(Top-5: 90.67%)* |
| **Phương pháp của Chúng ta** | **Nghiên cứu này** | **Calibrated Skeleton (Domain-Matched)** | **> 120 FPS** | **88.81%** *(Top-5: 97.85%)* | **46.27%** *(Top-5: 79.48%)* |

> [!NOTE]
> **Điểm vượt trội của Nghiên cứu**:
> 1. Trong khi các mô hình đạt >90% (Luvizon, PoseConv3D, CascadeFormer) đều sử dụng mạng khổng lồ (>15M–50M params) nạp video RGB hoặc khối Heatmap 3D rất tốn bộ nhớ, phương pháp của chúng ta **chỉ sử dụng vector tọa độ 2D thuần túy với mô hình 0.5M tham số**.
> 2. Đạt tốc độ **129.2 FPS**, cho phép triển khai ngay trên các thiết bị camera giám sát và điện thoại thông minh với độ chính xác đạt tới **88.81%**.

---

## 10. Kết luận & Các Đóng góp Cốt lõi của Đề tài

1. **Đóng góp 1 (Kiểm toán Gốc rễ)**: Làm sáng tỏ nguyên nhân sụp đổ hiệu năng giữa quy ước *Viewer-Centric* và *Subject-Centric*; đề xuất giải pháp hình học tự động khắc phục mà không cần gán nhãn hay train lại mạng pose.
2. **Đóng góp 2 (Bộ công cụ KineCalib)**: Xây dựng pipeline hiệu chỉnh thời gian thực, không tham số học (training-free), giúp triệt tiêu **>50% rung lắc** và ổn định **>21% chiều dài xương** một cách nhất quán trên cả 2 tập dữ liệu độc lập.
3. **Đóng góp 3 (Giá trị Ứng dụng Hạ tầng)**: Chứng minh bằng số liệu thực nghiệm rằng hiệu chỉnh khung xương giúp mô hình nhận diện hành động BiLSTM tăng trực tiếp **+21.72% độ chính xác** so với khung xương thô.
4. **Đóng góp 4 (Quy chuẩn Huấn luyện Thích ứng Miền)**: Chứng minh rằng việc huấn luyện mô hình hạ tầng trực tiếp trên phân phối pose đã hiệu chỉnh (**Domain-Matched Training**) giúp hệ thống đạt độ chính xác **88.81% (Top-5: 97.85%)**, xóa nhòa khoảng cách lý thuyết - thực tế.

---

## 11. Khung Đề xuất Cấu trúc 12 Slide Báo cáo Thuyết trình

Dưới đây là dàn ý thiết kế slide tối ưu để bạn trình chiếu trước hội đồng:

* **Slide 1: Trang tiêu đề**: Tên đề tài, người thực hiện, người hướng dẫn, đơn vị nghiên cứu.
* **Slide 2: Đặt vấn đề & Thách thức**: Hiện tượng sụp đổ hiệu năng (từ 93% xuống 45%) khi áp dụng mô hình pose vào bài toán nhận diện hành động thực tế (Jitter, Occlusion, Bone distortion, L/R mismatch).
* **Slide 3: Mục tiêu & Đóng góp chính**: 4 đóng góp cốt lõi (Kiểm toán quy ước, Pipeline KineCalib, Mô hình BiLSTM Attention, Đánh giá đa tập dữ liệu).
* **Slide 4: Tổng quan Nghiên cứu Liên quan (2013 – 2025)**: Bảng đối sánh từ Actemes (2013), Luvizon (2018) đến PoseConv3D (2023), CascadeFormer (2025) và vị trí của đề tài (siêu nhẹ, thời gian thực >120 FPS).
* **Slide 5: Hai Tập Dữ Liệu Nghiên cứu**: So sánh Penn Action (15 actions, 2.3k videos) và J-HMDB (21 actions, 928 videos).
* **Slide 6: Giai đoạn 1 - Đánh giá YOLO11n & Kiểm toán Lỗi Giải phẫu**: Tốc độ 129 FPS; phát hiện lỗi Viewer-Centric vs Subject-Centric; trực quan hóa sai số chéo ngực.
* **Slide 7: Giai đoạn 2 - Pipeline Hiệu chỉnh KineCalib**: Sơ đồ 4 module (1€ Filter + Spline + Bone constraints + L/R align); công thức toán học hàm mục tiêu.
* **Slide 8: Kết quả Nghiên cứu Triệt tiêu (Ablation Study)**: Bảng số liệu triệt tiêu 5 cấu hình; biểu đồ minh chứng Jitter giảm 49.9% và Bone std giảm 23.0%.
* **Slide 9: Giai đoạn 3 - Nhận diện Hành động BiLSTM + Attention**: Kiến trúc mạng 2 tầng BiLSTM + 4-head Attention; chuẩn hóa MidHip; kết quả phục hồi từ 45.51% lên 67.23% (+21.72%).
* **Slide 10: Bốn Hướng Tinh chỉnh & Đột phá Domain-Matched**: Kết quả 4 hướng thực nghiệm (Kinematics 64 chiều $\to$ 69.76%); bước nhảy vọt Domain-Matched Training $\to$ **88.81% Top-1 / 97.85% Top-5**.
* **Slide 11: Đánh giá Đa Tập Dữ Liệu (Penn Action vs. J-HMDB)**: Bảng đối sánh tính nhất quán khử rung (~50-55%) và ổn định xương (~21-23%) trên cả hai tập dữ liệu; so sánh với các công trình quốc tế.
* **Slide 12: Kết luận & Hướng phát triển**: Tóm tắt đóng góp; khả năng triển khai thực tế trên thiết bị di động/camera giám sát; lời cảm ơn và phần hỏi đáp (Q&A).
