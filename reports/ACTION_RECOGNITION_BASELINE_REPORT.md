# BÁO CÁO THỰC NGHIỆM NHẬN DIỆN HÀNH ĐỘNG VỚI MÔ HÌNH BiLSTM + ATTENTION
## Đánh Giá Đóng Góp Của Pipeline Hiệu Chỉnh Khung Xương 2D Lên Bài Toán Hạ Nguồn (Downstream Action Recognition)

---

## 1. TỔNG QUAN VÀ CÂU HỎI NGHIÊN CỨU

Trong các nghiên cứu về xử lý khung xương người (Skeleton-based Human Motion Analysis), việc giảm sai số pixel ở mức độ cục bộ (low-level pose estimation error) chỉ thực sự có giá trị thực tiễn nếu nó **trực tiếp nâng cao hiệu quả của các bài toán hạ nguồn (downstream tasks)**, tiêu biểu nhất là **Nhận diện hành động (Action Recognition)**.

Dựa trên kiến trúc tiêu chuẩn từ nghiên cứu tại `nckh_tretuky/BiLSTM`, chúng tôi thiết kế và huấn luyện mô hình **ActionBiLSTMAttentionClassifier** trong **50 epochs** trên tập dữ liệu video thể thao Penn Action (15 hành động) nhằm giải quyết câu hỏi:

> *"Dữ liệu khung xương sau khi được hiệu chỉnh (Calibrated Skeleton) có giúp mô hình nhận diện hành động chính xác hơn so với khung xương thô (Raw YOLO) hay không?"*

---

## 2. KIẾN TRÚC MÔ HÌNH VÀ THIẾT LẬP THỰC NGHIỆM

* **Kiến trúc mạng (PyTorch)**:
  * **Input Layer**: 12 khớp giải phẫu chuẩn hóa $\times 2$ tọa độ $(x, y) = 24$ chiều/frame.
  * **Tiền xử lý hình thái**:
    * *Centering*: Dời gốc tọa độ $(0, 0)$ về trọng tâm hông $\text{MidHip} = (\text{LHip} + \text{RHip}) / 2$.
    * *Scaling*: Chuẩn hóa theo khoảng cách hai vai hoặc thân người để bất biến theo khoảng cách camera.
  * **Backbone**: BiLSTM 2 tầng (`hidden_size=128`, `bidirectional=True`, output 256 chiều).
  * **Attention Layer**: Masked Multi-Head Self-Attention (4 heads, `embed_dim=256`, `dropout=0.3`).
  * **Pooling**: Masked Temporal Mean Pooling loại bỏ padding frames.
  * **Classification Head**: `Linear(256, 15)` dự đoán phân phối xác suất 15 lớp hành động.
* **Tổng số tham số mô hình**: **819.983 parameters**.
* **Tập dữ liệu phân chia chuẩn**:
  * **Tập huấn luyện (Train Split)**: 1.258 chuỗi video (87.543 frames).
  * **Tập kiểm thử (Test Split)**: 1.068 chuỗi video (76.298 frames).
* **Siêu tham số huấn luyện**:
  * Số epochs: **50 epochs**.
  * Batch size: 32.
  * Optimizer: Adam (`lr=0.001`, `weight_decay=1e-4`), `gradient_clip_norm=1.0`.
  * Scheduler: Cosine Annealing LR (`T_max=50`, `eta_min=1e-5`).
  * Phần cứng: NVIDIA GeForce RTX 3050 GPU (thời gian huấn luyện 50 epochs: **108.45 giây / ~1.8 phút**).

---

## 3. BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH

Dữ liệu được trích xuất từ [outputs/action_recognition/action_recognition_summary.json](file:///home/dekii2275/minhanh_nckh/outputs/action_recognition/action_recognition_summary.json):

| Phương Thức Khung Xương Đưa Vào Nhận Diện | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Đánh Giá Ý Nghĩa |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Ground Truth Skeleton** *(Upper Bound)* | **92.98%** | **97.85%** | **92.72%** | **92.99%** | Giới hạn trần lý thuyết của khung xương 2D chuẩn |
| **2. Raw YOLO-Pose Skeleton** *(Trước hiệu chỉnh)* | **45.51%** | **66.85%** | **39.82%** | **43.04%** | Độ chính xác sụt giảm nghiêm trọng do rung giật & lệch quy ước |
| **3. Calibrated YOLO Skeleton** *(Sau hiệu chỉnh)* | **67.23%** | **81.27%** | **66.31%** | **65.75%** | **Khôi phục mạnh mẽ độ chính xác và độ ổn định** |
| **MỨC ĐỘ CẢI THIỆN NHỜ HIỆU CHỈNH ($\Delta$)** | **+21.72%** | **+14.42%** | **+26.49%** | **+22.71%** | **Hiệu quả vượt trội của Calibration Pipeline** |

---

## 4. PHÂN TÍCH VÀ BÌNH LUẬN KHOA HỌC

### 1. Hiện tượng suy giảm hiệu năng khi dùng Raw YOLO:
Khi đưa dữ liệu thô từ mô hình pose estimation vào bài toán nhận diện hành động, độ chính xác Top-1 sụt giảm từ **92.98% xuống 45.51%** (giảm gần một nửa).
* *Nguyên nhân*: Các cú nhảy tọa độ đột ngột (teleportation), rung lắc chi xa (jitter ở cổ tay, mắt cá) và đặc biệt là hiện tượng hoán đổi trái/phải khi người xoay lưng khiến mạng BiLSTM nhận chuỗi động học bị biến dạng, dẫn tới dự đoán nhầm các hành động có quỹ đạo tương tự (ví dụ nhầm giữa `baseball_pitch` và `tennis_serve`).

### 2. Sự phục hồi vượt bậc nhờ Pipeline Hiệu chỉnh (Calibration Gain):
Khi áp dụng toàn diện 4 module hiệu chỉnh (PCHIP + Bone Constraints + One-Euro Filter + Viewpoint Alignment):
* **Top-1 Accuracy tăng vọt từ 45.51% lên 67.23%** (tăng **+21.72%** tuyệt đối).
* **Macro-F1 tăng vọt từ 39.82% lên 66.31%** (tăng **+26.49%** tuyệt đối).
* **Top-3 Accuracy đạt 81.27%** (tăng **+14.42%**).

Điều này cung cấp **bằng chứng thực nghiệm đanh thép** cho bài báo nghiên cứu khoa học: Khung hiệu chỉnh đề xuất không chỉ làm mượt ảnh nhìn bằng mắt thường mà thực sự cung cấp các vector đặc trưng cơ sinh học chuẩn xác hơn cho các mô hình học sâu phân loại chuyển động.

---

## 5. CÁC TỆP DỮ LIỆU VÀ BIỂU ĐỒ MINH CHỨNG LIÊN QUAN

* Checkpoint mô hình tốt nhất (Epoch 23): [outputs/action_recognition/best_bilstm_model.pt](file:///home/dekii2275/minhanh_nckh/outputs/action_recognition/best_bilstm_model.pt)
* Đồ thị huấn luyện 50 Epochs (Loss & Accuracy): [outputs/action_recognition/training_curves.png](file:///home/dekii2275/minhanh_nckh/outputs/action_recognition/training_curves.png)
* Ma trận nhầm lẫn Ground Truth: [outputs/action_recognition/confusion_matrix_gt.png](file:///home/dekii2275/minhanh_nckh/outputs/action_recognition/confusion_matrix_gt.png)
* Ma trận nhầm lẫn Raw YOLO (Trước hiệu chỉnh): [outputs/action_recognition/confusion_matrix_raw.png](file:///home/dekii2275/minhanh_nckh/outputs/action_recognition/confusion_matrix_raw.png)
* Ma trận nhầm lẫn Calibrated (Sau hiệu chỉnh): [outputs/action_recognition/confusion_matrix_calib.png](file:///home/dekii2275/minhanh_nckh/outputs/action_recognition/confusion_matrix_calib.png)
* Tệp JSON tổng hợp số liệu: [outputs/action_recognition/action_recognition_summary.json](file:///home/dekii2275/minhanh_nckh/outputs/action_recognition/action_recognition_summary.json)
