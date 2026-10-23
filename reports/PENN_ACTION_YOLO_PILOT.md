# BÁO CÁO THỰC NGHIỆM ĐỊNH LƯỢNG: PENN ACTION — YOLO POSE EVALUATION PILOT

**Đề tài:** Đánh giá chất lượng và hiệu chỉnh dữ liệu khung xương 2D phục vụ phân tích chuyển động  
**Ngày báo cáo:** 22/10/2026  
**Người thực hiện:** Antigravity AI Assistant & Nhóm nghiên cứu  
**Trạng thái thực nghiệm:** **VALIDATED**

---

## 1. MỤC TIÊU NGHIÊN CỨU

Nghiên cứu hướng tới việc đánh giá độ tin cậy và sai số định lượng của các mô hình ước lượng tư thế 2D (Pose Estimation) tiền huấn luyện trên các tác vụ phân tích chuyển động con người trong video thể thao / vận động.

Trong phạm vi pilot này:
1. Thiết lập một pipeline đánh giá độc lập hoàn chỉnh, có khả năng tái lập 100% trên tập dữ liệu video chuyển động chuẩn **Penn Action Dataset**.
2. Đánh giá mô hình **YOLO11n-Pose** (phiên bản nano chính thức của Ultralytics, pretrained trên COCO-17 keypoints) khi suy luận Zero-Shot (không huấn luyện, không fine-tune).
3. Đo lường sai lệch tọa độ Euclidean pixel, tỷ lệ phát hiện chính xác (PCK@20px, PCK-torso@0.10, PCK-torso@0.20), phân rã sai số theo từng khớp giải phẫu và phân tích hiện tượng "Confidence cao nhưng sai số lớn".
4. Cung cấp cơ sở dữ liệu định lượng phục vụ việc thiết kế thuật toán hiệu chỉnh khung xương (skeleton filtering / smoothing / kinematic correction) ở các giai đoạn tiếp theo.

---

## 2. NGUỒN VÀ MÔ TẢ DỮ LIỆU PENN ACTION

* **Nguồn chính thức:** University of Pennsylvania (GRASP Laboratory / CIS Department)  
  URL: `https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz` (Website: `https://dreamdragon.github.io/PennAction/`)
* **Trích dẫn:** Weiyu Zhang, Menglong Zhu, Konstantinos Derpanis, *"From Actemes to Action: A Strongly-supervised Representation for Detailed Action Understanding"*, ICCV 2013.
* **Quy mô tập dữ liệu gốc:** 2.326 chuỗi video (video sequences), 15 loại hành động thể thao/vận động.
* **Dung lượng tệp lưu trữ:** 3.235.203.923 bytes (3,01 GB nén `.tar.gz`), giải nén đầy đủ thành 2.326 thư mục ảnh RGB và 2.326 tệp nhãn MATLAB `.mat`.
* **Cấu trúc annotation `.mat`:** Mỗi tệp chứa ma trận tọa độ `x`, `y` kích thước $(N \times 13)$, ma trận `visibility` $(N \times 13)$, nhãn hành động `action`, góc nhìn `pose`, hộp bao `bbox`, kích thước không gian-thời gian `dimensions` $[H, W, N]$, và số khung hình `nframes`.
* **Hệ thống 13 khớp chuẩn Penn Action:**
  1. Head
  2. Left Shoulder
  3. Right Shoulder
  4. Left Elbow
  5. Right Elbow
  6. Left Wrist
  7. Right Wrist
  8. Left Hip
  9. Right Hip
  10. Left Knee
  11. Right Knee
  12. Left Ankle
  13. Right Ankle

---

## 3. MÔI TRƯỜNG THỰC NGHIỆM

* **Hệ điều hành:** Linux x86_64 (Ubuntu 24.04 LTS, Kernel `7.0.0-38-generic`)
* **Bộ xử lý đồ họa (GPU):** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB GDDR6 VRAM, Driver 580.178.04, Compute Capability 8.6 - Ampere)
* **Phiên bản CUDA:** CUDA 12.4 runtime (Driver CUDA 13.0)
* **Môi trường Python:** Conda environment `penn_pose` (Python 3.10.22)
* **Thư viện cốt lõi:**
  * `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`
  * `ultralytics==8.4.183`
  * `opencv-python==5.0.0.93`
  * `scipy==1.15.3`
  * `pandas==2.3.3`
  * `matplotlib==3.10.9`
  * `tqdm==4.70.1`

---

## 4. QUY CÁCH CHỌN CHUỖI VÀ KHUNG HÌNH (SAMPLING PROTOCOL)

Nhằm đảm bảo tính khách quan khoa học, **tiêu chí chọn mẫu được xác định tiên nghiệm (a priori)** trước khi chạy bất kỳ bước suy luận nào:
1. **Đa dạng về động học (Kinematic Diversity):** Chọn 3 hành động có đặc tính chuyển động rõ rệt:
   * `jumping_jacks`: Chuyển động dạng/khép liên tục đồng thời cả 2 tay và 2 chân.
   * `squat`: Chuyển động gập duỗi khớp gối và hông theo trục dọc, thân trên chịu tải.
   * `tennis_forehand`: Chuyển động xoay thân trên, vung tay cầm vợt và bước chân định hướng.
2. **Quy tắc chọn chuỗi:**
   * Chuỗi có tỷ lệ nhãn hợp lệ cao (visibility > 85%).
   * Ưu tiên chuỗi có một người mục tiêu rõ ràng trong khung hình để giảm thiểu nhiễu đa đối tượng.
   * Chọn chuỗi đầu tiên thỏa mãn điều kiện theo thứ tự sắp xếp cố định.
3. **Lấy mẫu theo thời gian (Uniform Temporal Sampling):**
   * Với mỗi chuỗi, các khung hình được chọn cách đều theo thời gian từ frame đầu tiên đến frame cuối cùng: $\text{indices} = \text{round}(\text{linspace}(0, N-1, K))$.
   * *Giới hạn thực tế phát hiện:* Trong Penn Action, toàn bộ 112 chuỗi `jumping_jacks` đều có độ dài tối đa 42 frame (trung bình 30 frame). Chuỗi dài nhất `1101` có đúng 42 frame. Để không lặp lại ảnh, toàn bộ 42 frame của chuỗi này được đưa vào đánh giá; kết hợp với 50 frame phân bố đều từ chuỗi `1665` (`squat`, tổng 109 frame) và 50 frame từ chuỗi `1986` (`tennis_forehand`, tổng 51 frame).
   * Tổng số khung hình thực nghiệm: **142 khung hình độc lập**.
   * Danh sách khung hình được lưu tại: [data/processed/selected_frames.csv](file:///home/dekii2275/minhanh_nckh/data/processed/selected_frames.csv).

---

## 5. MAPPING HỆ KHỚP (KEYPOINT MAPPING)

YOLO11n-Pose dự đoán 17 khớp theo chuẩn COCO-17, trong khi Penn Action cung cấp 13 khớp. Chỉ **12 khớp tương thích giải phẫu** được đưa vào so sánh định lượng.

Khớp **Head** (index 0 trong Penn Action) bị loại bỏ khỏi đánh giá định lượng vì COCO-17 định nghĩa 5 điểm mặt riêng biệt (mũi, 2 mắt, 2 tai) mà không có một tâm đầu đơn lẻ, dẫn đến không đồng nhất về mặt giải phẫu.

Bảng mapping chuẩn được lưu tại [configs/joint_mapping.json](file:///home/dekii2275/minhanh_nckh/configs/joint_mapping.json):

| Penn Action Index (0-based) | Penn Action Joint Name | COCO Index | COCO Joint Name | Ghi chú giải phẫu |
| :---: | :---: | :---: | :---: | :--- |
| 1 | `left_shoulder` | 5 | `left_shoulder` | Khớp vai trái |
| 2 | `right_shoulder` | 6 | `right_shoulder` | Khớp vai phải |
| 3 | `left_elbow` | 7 | `left_elbow` | Khớp khuỷu tay trái |
| 4 | `right_elbow` | 8 | `right_elbow` | Khớp khuỷu tay phải |
| 5 | `left_wrist` | 9 | `left_wrist` | Khớp cổ tay trái |
| 6 | `right_wrist` | 10 | `right_wrist` | Khớp cổ tay phải |
| 7 | `left_hip` | 11 | `left_hip` | Khớp háng trái |
| 8 | `right_hip` | 12 | `right_hip` | Khớp háng phải |
| 9 | `left_knee` | 13 | `left_knee` | Khớp đầu gối trái |
| 10 | `right_knee` | 14 | `right_knee` | Khớp đầu gối phải |
| 11 | `left_ankle` | 15 | `left_ankle` | Khớp cổ chân trái |
| 12 | `right_ankle` | 16 | `right_ankle` | Khớp cổ chân phải |

*Lưu ý quan trọng qua kiểm chứng ảnh:* Dữ liệu Penn Action được gán nhãn qua Mechanical Turk với công cụ VATIC, xuất hiện sự không đồng nhất giữa góc nhìn người quan sát (viewer-centric) và giải phẫu tự thân (subject-centric) ở một số chuỗi, dẫn đến việc đổi bên trái/phải là một trong những nguồn sai số lớn nhất.

---

## 6. CẤU HÌNH YOLO11n-POSE & QUY TRÌNH SUY LUẬN

* **Trọng số:** `models/yolo11n-pose.pt` chính thức từ Ultralytics release (kích thước 6,0 MB).
* **Kích thước ảnh đầu vào:** `imgsz = 640`.
* **Ngưỡng tin cậy phát hiện:** `conf = 0.25`.
* **Thiết bị chạy:** GPU `cuda:0` (tốc độ đạt ~58,7 frames/giây, tổng thời gian suy luận cho 142 ảnh chỉ mất 2,4 giây).
* **Quy tắc chọn đối tượng khi có nhiều người:** Tự động chọn đối tượng có `box_confidence` phát hiện người cao nhất. **Tuyệt đối không dùng Ground Truth để lọc hay chọn đối tượng.**
* Tệp dự đoán thô: [outputs/predictions/yolo_predictions_raw.json](file:///home/dekii2275/minhanh_nckh/outputs/predictions/yolo_predictions_raw.json)  
* Tệp dự đoán đã chọn: [outputs/predictions/yolo_predictions.csv](file:///home/dekii2275/minhanh_nckh/outputs/predictions/yolo_predictions.csv)

---

## 7. QUY TRÌNH ĐÁNH GIÁ ĐỊNH LƯỢNG

1. **Sai số khoảng cách Euclidean:**
   $$\text{Error}_{px} = \sqrt{(x_{pred} - x_{gt})^2 + (y_{pred} - y_{gt})^2}$$
   Chỉ tính trên các khớp có GT hợp lệ (`visibility == 1` và tọa độ $> 0$).
2. **PCK (Percentage of Correct Keypoints):**
   * $\text{PCK@20px}$: Khớp dự đoán có sai số $\le 20$ pixels.
   * $\text{PCK-torso@0.10}$: Sai số $\le 0.10 \times \text{Torso Scale}$.
   * $\text{PCK-torso@0.20}$: Sai số $\le 0.20 \times \text{Torso Scale}$.
   * *Torso Scale* tính bằng khoảng cách Euclidean giữa trung điểm 2 vai và trung điểm 2 háng theo nhãn GT:
     $$d_{torso} = \left\| \frac{\mathbf{p}_{L\_sh} + \mathbf{p}_{R\_sh}}{2} - \frac{\mathbf{p}_{L\_hip} + \mathbf{p}_{R\_hip}}{2} \right\|_2$$
   * **Nguyên tắc bảo toàn mẫu số:** Mẫu số của PCK là **tổng số keypoint GT hợp lệ**. Nếu mô hình bỏ sót (missing prediction), điểm đó được tính là **không chính xác** (incorrect), không bị loại khỏi mẫu số.

---

## 8. BẢNG KẾT QUẢ ĐỊNH LƯỢNG TỔNG HỢP (MANDATORY SUMMARY)

| Metric | Giá trị thực nghiệm | Ghi chú / Diễn giải |
| :--- | :---: | :--- |
| **Sequences** | **3** | `1101` (jumping_jacks), `1665` (squat), `1986` (tennis_forehand) |
| **Frames** | **142** | 42 (jumping_jacks) + 50 (squat) + 50 (tennis_forehand) |
| **Valid GT keypoints** | **1.574** | Số điểm khớp có nhãn visibility = 1 |
| **Predicted keypoints** | **1.574** | 100% frame phát hiện được người mục tiêu |
| **Missing predictions** | **0** | Không có khung hình nào bị mất phát hiện |
| **Coverage Rate** | **100,0%** | Độ bao phủ phát hiện người hoàn hảo |
| **Mean error (px)** | **45,06 px** | Sai số khoảng cách Euclidean trung bình |
| **Median error (px)** | **39,71 px** | Trung vị sai số Euclidean |
| **Standard deviation (px)** | **30,27 px** | Độ lệch chuẩn sai số |
| **P90 error (px)** | **85,94 px** | 90% các điểm có sai số dưới 85,94 px |
| **PCK@20px** | **21,09%** | Tỷ lệ khớp có sai số $\le 20$ pixels |
| **PCK-torso@0.10** | **0,95%** | Ngưỡng $0.10 \times 63,67 \approx 6,37$ px (rất khắt khe trên video độ phân giải thấp) |
| **PCK-torso@0.20** | **2,92%** | Ngưỡng $0.20 \times 63,67 \approx 12,73$ px |

*Dữ liệu chi tiết lưu tại:* [outputs/metrics/evaluation_summary.json](file:///home/dekii2275/minhanh_nckh/outputs/metrics/evaluation_summary.json)

---

## 9. SAI SỐ CHI TIẾT THEO TỪNG KHỚP GIẢI PHẪU

Bảng xếp hạng sai số từ thấp nhất đến cao nhất (theo Mean Error):

| STT | Khớp giải phẫu | Số điểm GT hợp lệ | Mean Error (px) | Median Error (px) | Std Error (px) | P90 Error (px) | PCK@20px |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **right_hip** | 142 | **24,89** | 27,25 | 11,50 | 38,52 | 35,21% |
| 2 | **left_hip** | 142 | **27,78** | 29,48 | 11,23 | 41,28 | 33,80% |
| 3 | **right_shoulder** | 139 | **34,35** | 39,72 | 13,05 | 48,71 | 15,11% |
| 4 | **left_knee** | 137 | **37,27** | 38,60 | 18,52 | 60,98 | 26,28% |
| 5 | **left_shoulder** | 133 | **37,78** | 41,54 | 14,47 | 53,16 | 20,30% |
| 6 | **right_knee** | 142 | **41,19** | 28,46 | 29,16 | 84,09 | 37,32% |
| 7 | **right_ankle** | 142 | **41,19** | 36,88 | 24,21 | 66,62 | 26,06% |
| 8 | **left_ankle** | 142 | **43,74** | 41,05 | 23,41 | 73,65 | 14,08% |
| 9 | **right_elbow** | 82 | **56,11** | 49,17 | 33,19 | 101,86 | 14,63% |
| 10 | **left_wrist** | 118 | **64,57** | 48,74 | 45,88 | 151,82 | 9,32% |
| 11 | **right_wrist** | 133 | **72,10** | 73,77 | 43,35 | 142,99 | 12,78% |
| 12 | **left_elbow** | 122 | **72,35** | 80,64 | 23,34 | 98,61 | 0,00% |

### Nhận xét quy luật phân bố sai số:
1. **Các khớp thân mình (Torso - Hip & Shoulder):** Có sai số thấp nhất ($24,89 - 37,78$ px). Khối thân mình ít bị biến dạng mạnh và có vùng đặc trưng ổn định giúp mô hình dự đoán vị trí chính xác hơn.
2. **Các khớp ngoại vi (Extremities - Elbow & Wrist):** Có sai số lớn nhất ($56,11 - 72,35$ px). Khớp cổ tay phải (`right_wrist`) và khuỷu tay trái (`left_elbow`) có sai số P90 vượt trên 140 - 150 px.
3. **Nguyên nhân chính:**
   * Tốc độ vung tay chân cao gây hiện tượng mờ chuyển động (motion blur).
   * Khớp bị che khuất (occlusion) bởi thân người hoặc dụng cụ thể thao (thanh tạ barbell trong bài squat, vợt tennis trong bài forehand).
   * Nhầm lẫn đảo bên (left/right flipping) giữa hai cánh tay khi cơ thể xoay góc nghiêng/lưng.

---

## 10. PHÂN TÍCH ĐỘ TIN CẬY (CONFIDENCE) VÀ SAI SỐ

Phân tích quan hệ giữa `keypoint_confidence` của YOLO và sai số Euclidean thực tế:

| Khoảng Confidence | Số lượng Keypoints | Mean Error (px) | Median Error (px) | Tỷ lệ sai số > 20px |
| :---: | :---: | :---: | :---: | :---: |
| $[0.0, 0.3)$ | 18 | 46,32 | 45,28 | 100,00% |
| $[0.3, 0.5)$ | 59 | 49,40 | 47,43 | 81,36% |
| $[0.5, 0.7)$ | 47 | 55,13 | 69,01 | 70,21% |
| $[0.7, 0.9)$ | 77 | 39,70 | 32,38 | 71,43% |
| $[0.9, 1.0]$ | **1.373** | **44,82** | **39,31** | **79,24%** |

### Phát hiện quan trọng (Key Finding):
* **Có hiện tượng Confidence rất cao nhưng sai số rất lớn không?**  
  **CÓ, VÀ CHIẾM TỶ LỆ ÁP ĐẢO.**
  * Trong nhóm có confidence cực cao $[0.9, 1.0]$, có tới **79,24%** các khớp vẫn bị sai lệch $> 20$ pixels.
  * Có **1.143 khớp** đạt $\text{confidence} \ge 0.7$ nhưng sai số $> 20$ px.
  * Thậm chí có những điểm confidence đạt $> 0.95$ nhưng sai số $> 150$ px do mô hình tự tin gán nhầm cánh tay trái sang cánh tay phải khi người tập xoay lưng.
* **Kết luận định lượng:** `confidence` trả về bởi mạng nơ-ron chỉ phản ánh xác suất đặc trưng thị giác của mô hình phát hiện, **hoàn toàn không tỷ lệ nghịch với sai số tọa độ hình học thực tế**. Điều này khẳng định sự cần thiết bắt buộc của việc nghiên cứu **thuật toán hiệu chỉnh hậu xử lý (post-processing calibration & filtering)**.

---

## 11. HÌNH ẢNH MINH HỌA VÀ ĐỐI SÁNH TRỰC QUAN

Các hình ảnh và tệp demo được xuất tại thư mục [outputs/figures/](file:///home/dekii2275/minhanh_nckh/outputs/figures) và [outputs/demo/](file:///home/dekii2275/minhanh_nckh/outputs/demo):

1. **Biểu đồ sai số theo khớp:** [outputs/figures/chart_mean_error_per_joint.png](file:///home/dekii2275/minhanh_nckh/outputs/figures/chart_mean_error_per_joint.png)
2. **Biểu đồ phân phối sai số:** [outputs/figures/chart_error_distribution.png](file:///home/dekii2275/minhanh_nckh/outputs/figures/chart_error_distribution.png)
3. **Biểu đồ quan hệ Confidence vs Error:** [outputs/figures/chart_confidence_vs_error.png](file:///home/dekii2275/minhanh_nckh/outputs/figures/chart_confidence_vs_error.png)
4. **Bộ 6 Panel đối sánh 4 cột (RGB | GT xanh lá | YOLO đỏ | Sai số đường vàng):**
   * Trường hợp sai số thấp: [panel_error_low_1.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/panel_error_low_1.jpg) (13,0 px) và [panel_error_low_2.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/panel_error_low_2.jpg) (20,1 px) — Tennis forehand góc nhìn ngang rõ ràng.
   * Trường hợp sai số trung bình: [panel_error_med_1.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/panel_error_med_1.jpg) (49,8 px) và [panel_error_med_2.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/panel_error_med_2.jpg) (53,5 px) — Jumping jacks góc chính diện.
   * Trường hợp sai số cao: [panel_error_high_1.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/panel_error_high_1.jpg) (62,3 px) và [panel_error_high_2.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/panel_error_high_2.jpg) (70,6 px) — Squat góc nhìn từ phía sau với thanh tạ che khuất vai và khuỷu tay.
5. **Video & GIF Demo động:**
   * Video MP4: [outputs/demo/pilot_demo.mp4](file:///home/dekii2275/minhanh_nckh/outputs/demo/pilot_demo.mp4)
   * Ảnh động GIF: [outputs/demo/pilot_demo.gif](file:///home/dekii2275/minhanh_nckh/outputs/demo/pilot_demo.gif)

---

## 12. GIỚI HẠN VÀ ĐỀ XUẤT HƯỚNG NGHIÊN CỨU HIỆU CHỈNH

### Giới hạn quan sát được từ thực nghiệm:
1. **Nhiễu rung giật theo thời gian (Temporal Jitter):** YOLO11n suy luận độc lập từng khung hình (frame-by-frame) nên tọa độ các khớp ngoại vi (cổ tay, bàn chân) bị dao động từ 10 - 30 px giữa các frame liên tiếp dù chuyển động thực của con người rất mượt mà.
2. **Nhầm lẫn đối xứng trái/phải khi xoay thân:** Khi người tập quay lưng hoặc quay nghiêng, mô hình thường xuyên lật nhầm nhãn khớp trái sang phải.
3. **Che khuất do tương tác với vật thể:** Dụng cụ thể thao (thanh tạ, vợt) làm biến dạng biên cạnh quang học của cẳng tay, dẫn tới sai số vị trí khớp khuỷu tay lên tới $>70$ px.

### Hướng phát triển thuật toán hiệu chỉnh tiếp theo:
1. **Lọc làm mượt theo thời gian có ràng buộc động học (Kinematically-Constrained Temporal Filtering):** Sử dụng One-Euro Filter hoặc Extended Kalman Filter (EKF) với mô hình độ dài xương bất biến ($L_{bone} = \text{const}$).
2. **Mô hình học sửa lỗi dựa trên chuỗi thời gian (BiLSTM / Spatial-Temporal Graph Convolutional Networks - ST-GCN):** Nhận chuỗi skeleton 2D thô từ YOLO làm đầu vào và dự đoán offset hiệu chỉnh tọa độ khớp.
3. **Cơ chế tái hiệu chuẩn độ tin cậy (Confidence Recalibration):** Không dùng trực tiếp raw confidence của detector mà hiệu chuẩn lại dựa trên độ nhất quán vận tốc ($\Delta v$) và gia tốc ($\Delta a$) của từng khớp.
