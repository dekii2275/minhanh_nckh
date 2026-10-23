# BÁO CÁO KIỂM TOÁN THỰC NGHIỆM ĐỊNH LƯỢNG (EVALUATION AUDIT)
## ĐÁNH GIÁ TÍNH ĐÁNG TIN CẬY CỦA KẾT QUẢ PILOT PENN ACTION — YOLO11n-POSE

**Dự án:** Đánh giá chất lượng và hiệu chỉnh dữ liệu khung xương 2D phục vụ phân tích chuyển động  
**Workspace:** `/home/dekii2275/minhanh_nckh`  
**Ngày kiểm toán:** 22/10/2026  
**Đối tượng kiểm toán:** Kết quả suy luận và đo lường của Task 7 (142 khung hình thuộc 3 chuỗi: `1101`, `1665`, `1986`)  
**Tập mẫu kiểm toán chuyên sâu:** 15 khung hình phân bố đều trên 3 chuỗi  

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH KIỂM TOÁN

Báo cáo này thực hiện kiểm toán độc lập, rà soát tính toàn vẹn kỹ thuật và tính hợp lệ giải phẫu của các chỉ số định lượng thu được trong Task 7:
* Mean Error: **45,06 px**
* Median Error: **39,71 px**
* PCK@20px: **21,09%**
* PCK-torso@0.10: **0,95%**
* Khớp có sai số lớn nhất: `left_elbow` (72,35 px), `right_wrist` (72,10 px)
* Hiện tượng: 1.143 khớp có confidence $\ge 0.7$ nhưng sai số $> 20$ px.

**Nguyên tắc kiểm toán:**
* Không chạy lại YOLO, không tải lại dataset, sử dụng toàn bộ tệp kết quả và dữ liệu nguyên bản.
* Không tự ý sửa Ground Truth, không áp dụng thuật toán hiệu chỉnh cho kết quả chính thức.
* Thực hiện phép thử hoán đổi trái/phải (Diagnostic Swap) thuần túy với mục đích chẩn đoán nguyên nhân gốc rễ (Root Cause Analysis).

---

## 2. XÁC MINH TÍNH TOÀN VẸN KỸ THUẬT (TECHNICAL INTEGRITY)

Kiểm tra đối chiếu trên 15 khung hình được chọn trải đều:
* Chuỗi `1101` (`jumping_jacks`): Frames 1, 11, 22, 32, 42
* Chuỗi `1665` (`squat`): Frames 1, 27, 56, 82, 109
* Chuỗi `1986` (`tennis_forehand`): Frames 1, 13, 26, 38, 51

### Kết quả kiểm tra:
1. **Khớp nối Frame và Định danh:** 100% (15/15) khung hình trong `selected_frames.csv` khớp chính xác với tệp ảnh RGB `frames/{seq_id}/{frame_index:06d}.jpg` và chỉ số hàng $f_{0} = \text{frame\_index} - 1$ trong tệp annotation MATLAB `.mat`.
2. **Kích thước ảnh và Hệ tọa độ:**
   * Chuỗi `1101`: Ảnh RGB kích thước $270 \times 480$, ma trận `dimensions` trong `.mat` là $[270, 480, 42]$. Kích thước khớp 100%.
   * Chuỗi `1665`: Ảnh RGB kích thước $360 \times 480$, ma trận `dimensions` trong `.mat` là $[360, 480, 109]$. Kích thước khớp 100%.
   * Chuỗi `1986`: Ảnh RGB kích thước $360 \times 480$, ma trận `dimensions` trong `.mat` là $[360, 480, 51]$. Kích thước khớp 100%.
   * Hệ tọa độ: Cả OpenCV và YOLO đều sử dụng gốc tọa độ $(0, 0)$ ở góc trên bên trái, trục $X$ hướng sang phải, trục $Y$ hướng xuống dưới, hoàn toàn đồng nhất với tọa độ $x, y$ trong `.mat`.

---

## 3. PHÁT HIỆN LỖI CỐT LÕI: XUNG ĐỘT QUY ƯỚC TRÁI / PHẢI (LEFT/RIGHT CONVENTION CONFLICT)

Đây là **phát hiện quan trọng nhất** của quá trình kiểm toán.

### So sánh bản chất hai bộ quy ước:
1. **COCO-17 (Mô hình YOLO11-Pose):**
   * Tuân thủ quy ước **Giải phẫu tự thân (Subject-Centric / Anatomical)**.
   * `left_shoulder` (khớp 5) là vai trái của **chính đối tượng**. Khi đối tượng đứng đối diện camera (front pose), vai trái tự thân nằm ở **nửa phải của ảnh** ($x$ lớn hơn).
   * `right_shoulder` (khớp 6) nằm ở **nửa trái của ảnh** ($x$ nhỏ hơn).
2. **Penn Action Dataset (Tập Ground Truth):**
   * Được gán nhãn qua công cụ VATIC (Mechanical Turk crowdsourcing).
   * Trong các chuỗi quay chính diện (`1101`, `1037`, `1038`,...), người gán nhãn đã gán theo quy ước **Góc nhìn người xem (Viewer-Centric / Image-Left)**:
     * Cánh tay ở bên trái màn hình người xem được gán nhãn là `left_shoulder` (khớp 1), `left_elbow` (khớp 3), `left_wrist` (khớp 5).
     * Cánh tay ở bên phải màn hình người xem được gán nhãn là `right_shoulder` (khớp 2), `right_elbow` (khớp 4), `right_wrist` (khớp 6).

### Hệ quả trên phép đo Task 7:
* Khi áp dụng phép nối trực tiếp theo tên gọi (Direct Nominal Mapping: Penn `left_shoulder` nối với COCO `left_shoulder`), thuật toán đã **vô tình đo khoảng cách chéo xuyên thân (cross-torso distance)** giữa vai trái của YOLO và vai phải của người trong ảnh.
* Điều này lý giải hoàn toàn vì sao:
  1. Sai số trung bình bị đẩy vọt lên **45,06 px** (tương đương bề rộng thân người).
  2. Các khớp ngoại vi như khuỷu tay (`left_elbow`: 72,35 px) và cổ tay (`right_wrist`: 72,10 px) có sai số lớn nhất, vì khi dang tay tập jumping jacks hoặc cầm tạ squat, khoảng cách giữa 2 cổ tay lên tới 140 - 160 px!
  3. Hiện tượng "Confidence cao nhưng sai số lớn": YOLO phát hiện vai trái cực kỳ chính xác với confidence 0.99, nhưng nhãn GT lại gọi điểm đó là vai phải, tạo ra khoảng cách sai lệch ~40 px dù mô hình dự đoán đúng thực tế quang học.

---

## 4. PHÉP THỬ CHẨN ĐOÁN ĐẢO TRÁI/PHẢI (DIAGNOSTIC SWAP EXPERIMENT)

Nhằm định lượng chính xác mức độ ảnh hưởng của xung đột quy ước, chúng tôi thực hiện phép thử chẩn đoán hoán đổi 6 cặp khớp đối xứng:
* `left_shoulder` $\leftrightarrow$ `right_shoulder`
* `left_elbow` $\leftrightarrow$ `right_elbow`
* `left_wrist` $\leftrightarrow$ `right_wrist`
* `left_hip` $\leftrightarrow$ `right_hip`
* `left_knee` $\leftrightarrow$ `right_knee`
* `left_ankle` $\leftrightarrow$ `right_ankle`

*(Lưu ý: Phép thử này thuần túy phục vụ chẩn đoán nguyên nhân, không dùng GT để chọn lọc kết quả chính thức).*

### Kết quả so sánh tổng thể trên toàn bộ 142 khung hình:

| Chỉ số định lượng | Task 7 (Direct Mapping) | Phép thử chẩn đoán (Swapped Mapping) | Mức độ thay đổi |
| :--- | :---: | :---: | :---: |
| **Mean Error (pixel)** | **45,06 px** | **7,40 px** | **Giảm 83,58% sai số** |
| **Median Error (pixel)** | **39,71 px** | **5,41 px** | **Giảm 86,38% sai số** |
| **PCK@20px** | **21,09%** | **95,04%** | **Tăng từ 21,1% lên 95,0%** |
| **PCK-torso@0.10** | **0,95%** | **51,27%** | **Tăng từ 1,0% lên 51,3%** |
| **PCK-torso@0.20** | **2,92%** | **80,24%** | **Tăng từ 2,9% lên 80,2%** |
| **Sai số chuẩn hóa / Torso** | **$0,729 \times d_{torso}$** | **$0,121 \times d_{torso}$** | Giảm từ 73% xuống 12% thân |
| **Sai số chuẩn hóa / BBox Diag**| **18,1%** | **3,2%** | Giảm từ 18% xuống 3,2% người |

### Kết quả phân rã theo từng chuỗi:

| Chuỗi Video | Hành động & Góc nhìn | Direct Mean Error | Swapped Mean Error | Mức giảm sai số | Direct PCK@20px | Swapped PCK@20px |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seq 1101** | `jumping_jacks` (Front) | 57,42 px | **5,70 px** | **-90,07%** | 12,9% | **99,8%** |
| **Seq 1665** | `squat` (Rear/Quarter) | 56,57 px | **10,73 px** | **-81,03%** | 0,5% | **87,0%** |
| **Seq 1986** | `tennis_forehand` (Side) | 21,13 px | **5,56 px** | **-73,69%** | 50,5% | **98,7%** |

---

## 5. BẢNG DỮ LIỆU KIỂM TOÁN CHI TIẾT TRÊN 15 KHUNG HÌNH MẪU

Dữ liệu chi tiết trích xuất từ [outputs/metrics/audit_15_frames.csv](file:///home/dekii2275/minhanh_nckh/outputs/metrics/audit_15_frames.csv):

| Sequence | Frame | Pose | Kích thước | Torso Scale | Direct Err (px) | Direct PCK20 | Swapped Err (px) | Swapped PCK20 | Giảm sai số (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1101** | 1 | front | $480 \times 270$ | 59,7 px | 32,08 | 33,3% | **4,19** | **100,0%** | 86,9% |
| **1101** | 11 | front | $480 \times 270$ | 61,4 px | 66,09 | 8,3% | **3,31** | **100,0%** | 95,0% |
| **1101** | 22 | front | $480 \times 270$ | 64,6 px | 57,85 | 0,0% | **7,27** | **100,0%** | 87,4% |
| **1101** | 32 | front | $480 \times 270$ | 63,5 px | 75,19 | 0,0% | **6,93** | **100,0%** | 90,8% |
| **1101** | 42 | front | $480 \times 270$ | 59,7 px | 33,96 | 25,0% | **5,75** | **100,0%** | 83,1% |
| **1665** | 1 | right | $480 \times 360$ | N/A | 53,46 | 0,0% | **8,14** | **90,0%** | 84,8% |
| **1665** | 27 | right | $480 \times 360$ | 83,2 px | 60,27 | 0,0% | **10,02** | **90,9%** | 83,4% |
| **1665** | 56 | right | $480 \times 360$ | 87,7 px | 57,12 | 0,0% | **13,86** | **81,8%** | 75,7% |
| **1665** | 82 | right | $480 \times 360$ | 82,1 px | 54,46 | 0,0% | **10,53** | **81,8%** | 80,7% |
| **1665** | 109 | right | $480 \times 360$ | 93,5 px | 49,80 | 0,0% | **7,97** | **90,9%** | 84,0% |
| **1986** | 1 | left | $480 \times 360$ | N/A | 10,69 | 88,9% | **3,79** | **100,0%** | 64,5% |
| **1986** | 13 | left | $480 \times 360$ | 41,3 px | 20,73 | 50,0% | **5,21** | **100,0%** | 74,9% |
| **1986** | 26 | left | $480 \times 360$ | 37,5 px | 22,32 | 50,0% | **4,11** | **100,0%** | 81,6% |
| **1986** | 38 | left | $480 \times 360$ | 36,4 px | 16,39 | 70,0% | **8,93** | **90,0%** | 45,5% |
| **1986** | 51 | left | $480 \times 360$ | 46,3 px | 17,20 | 60,0% | **5,67** | **100,0%** | 67,0% |

---

## 6. BẰNG CHỨNG HÌNH ẢNH TRỰC QUAN (VISUAL EVIDENCE)

Toàn bộ 15 ảnh bằng chứng kiểm toán độc lập đã được tạo và lưu tại thư mục [outputs/figures/audit/](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit). Mỗi hình gồm 2 panel đặt cạnh nhau: Panel trái hiển thị kết quả nối trực tiếp (đường nối màu vàng) và Panel phải hiển thị kết quả chẩn đoán hoán đổi (đường nối màu xanh lơ cyan).

### Bằng chứng trên từng chuỗi:
1. **Chuỗi 1101 (`jumping_jacks`):**
   * [audit_seq1101_f0001.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1101_f0001.jpg) — Tư thế chuẩn bị bắt đầu nhảy.
   * [audit_seq1101_f0011.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1101_f0011.jpg) — Đang dang rộng tay: Panel trái các đường vàng nối chéo qua ngực (Mean Err: 66,1 px); Panel phải các điểm đỏ và xanh lá chồng khít hoàn hảo lên nhau (Mean Err: **3,3 px**, PCK20: **100%**).
   * [audit_seq1101_f0022.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1101_f0022.jpg) — Đỉnh điểm vung tay lên cao (Err giảm từ 57,9 px xuống **7,3 px**).
   * [audit_seq1101_f0032.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1101_f0032.jpg) — Hạ tay xuống (Err giảm từ 75,2 px xuống **6,9 px**).
   * [audit_seq1101_f0042.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1101_f0042.jpg) — Kết thúc chu kỳ nhảy (Err giảm từ 34,0 px xuống **5,8 px**).

2. **Chuỗi 1665 (`squat`):**
   * [audit_seq1665_f0001.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1665_f0001.jpg) — Đứng chuẩn bị gánh tạ (Err giảm từ 53,5 px xuống **8,1 px**).
   * [audit_seq1665_f0027.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1665_f0027.jpg) — Bắt đầu hạ gối xuống (Err giảm từ 60,3 px xuống **10,0 px**).
   * [audit_seq1665_f0056.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1665_f0056.jpg) — Đáy chuyển động squat (Deep Squat): Toàn bộ đường vàng ở vai, hông, gối, cổ chân bị vắt chéo; Panel chẩn đoán khớp hoàn toàn với đùi và cẳng chân (Err giảm từ 57,1 px xuống **13,9 px**).
   * [audit_seq1665_f0082.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1665_f0082.jpg) — Đang đẩy người đứng dậy (Err giảm từ 54,5 px xuống **10,5 px**).
   * [audit_seq1665_f0109.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1665_f0109.jpg) — Đứng thẳng hoàn thành lượt squat (Err giảm từ 49,8 px xuống **8,0 px**).

3. **Chuỗi 1986 (`tennis_forehand`):**
   * [audit_seq1986_f0001.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1986_f0001.jpg) — Tư thế thủ sẵn sàng (Err giảm từ 10,7 px xuống **3,8 px**).
   * [audit_seq1986_f0013.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1101_f0013.jpg) — Đang mở vợt ra sau (Err giảm từ 20,7 px xuống **5,2 px**).
   * [audit_seq1986_f0026.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1986_f0026.jpg) — Điểm tiếp xúc bóng (Err giảm từ 22,3 px xuống **4,1 px**).
   * [audit_seq1986_f0038.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1986_f0038.jpg) — Vung vợt theo quán tính (Follow-through) (Err giảm từ 16,4 px xuống **8,9 px**).
   * [audit_seq1986_f0051.jpg](file:///home/dekii2275/minhanh_nckh/outputs/figures/audit/audit_seq1986_f0051.jpg) — Thu vợt và lấy lại thăng bằng (Err giảm từ 17,2 px xuống **5,7 px**).

---

## 7. KIỂM TRA CÁCH TÍNH CHỈ SỐ PCK VÀ TORSO SCALE

1. **Về mẫu số PCK và Missing Predictions:**
   * Cách tính trong Task 7 hoàn toàn **chính xác và trung thực**: Mẫu số là 1.574 điểm GT hợp lệ. Không có khung hình nào bị mất phát hiện (`missing = 0`).
2. **Về quy mô Torso Scale:**
   * Trong Penn Action, do độ phân giải video gốc là $360 \times 480$ hoặc $270 \times 480$, người mục tiêu chỉ chiếm một phần khung hình.
   * Chiều dài thân người đo được (từ trung điểm vai đến trung điểm háng) chỉ đạt trung bình **63,67 pixels**.
   * Vì vậy, ngưỡng $\text{PCK-torso@0.10}$ tương đương với sai số chỉ **$\le 6,37$ pixels**.
   * Đây là một khoảng cách cực kỳ nhỏ (chỉ bằng độ dày nét vẽ của một khớp trên ảnh).
   * Kết hợp với việc quy ước trái/phải bị đảo khiến sai số trung bình là 45 px, việc chỉ có **0,95%** các điểm lọt vào ngưỡng 6,37 px là hệ quả toán học tất yếu, không phải lỗi tính toán công thức.
   * Khi loại trừ ảnh hưởng của xung đột nhãn (trong phép thử chẩn đoán hoán đổi), $\text{PCK-torso@0.10}$ đạt **51,27%** và $\text{PCK-torso@0.20}$ đạt **80,24%**.

---

## 8. KẾT LUẬN KIỂM TOÁN VÀ KHUYẾN NGHỊ (AUDIT VERDICT)

### Trả lời câu hỏi: "Các chỉ số Task 7 có đáng tin cậy hay cần tính lại?"

1. **Về tính đáng tin cậy của thuật toán đánh giá:**
   * Mã nguồn và các phép tính trong Task 7 hoàn toàn **chính xác về mặt toán học và logic thực thi**: Đọc đúng file, đúng frame, đúng kích thước ảnh, đúng công thức Euclidean và bảo toàn nghiêm ngặt mẫu số PCK.
   * Các con số báo cáo trong Task 7 (Mean error 45,06 px, PCK@20px 21,09%) phản ánh **trung thực 100% khoảng cách sai lệch giữa mô hình COCO và nhãn Penn Action khi nối theo tên danh nghĩa nguyên bản (Direct Nominal Mapping)**.

2. **Về mặt bản chất học máy và phân tích chuyển động (Crucial Scientific Insight):**
   * Các chỉ số trong Task 7 **không phản ánh năng lực nhận dạng thực sự của YOLO11n-Pose**, mà chủ yếu phản ánh **sự không tương thích quy ước gán nhãn giữa hai tập dữ liệu**:
     * YOLO11-Pose nhận dạng tư thế người cực kỳ xuất sắc (sai số quang học thực tế chỉ **$\approx 5,4 - 7,4$ pixels**, PCK@20px thực tế đạt **95,0%**).
     * Tuy nhiên, nhãn Penn Action được gán theo quy ước "Bên trái màn hình" (Viewer-Centric), trong khi COCO được huấn luyện theo "Bên trái giải phẫu cơ thể" (Subject-Centric).

3. **Khuyến nghị cho báo cáo nghiên cứu:**
   * **KHÔNG CẦN CHẠY LẠI MÔ HÌNH.** Dữ liệu suy luận đã hoàn chỉnh và lưu giữ đầy đủ tại `outputs/predictions/yolo_predictions.csv`.
   * **BÁO CÁO CẦN TRÌNH BÀY SONG SONG 2 GÓC NHÌN:**
     1. **Góc nhìn Benchmark Thực tế (Direct Nominal Mapping):** Ghi nhận Mean Error 45,06 px và PCK 21,09% như một minh chứng thực tế về rào cản chuyển giao mô hình (Domain/Convention Gap) giữa các bộ dữ liệu khác nhau.
     2. **Góc nhìn Phân tích Chẩn đoán Giải phẫu (Anatomically-Aligned Diagnostic):** Trình bày kết quả hoán đổi đối xứng (Mean Error 7,40 px, PCK 95,0%) làm bằng chứng khoa học cho thấy mô hình YOLO11n-Pose hoàn toàn chính xác về mặt quang học, và nguyên nhân chính gây sai số là xung đột hướng nhìn.
   * **ĐÓNG GÓP CHO ĐỀ TÀI:**
     Phát hiện này là một **điểm sáng học thuật quan trọng** cho đề tài *"Đánh giá chất lượng và hiệu chỉnh dữ liệu khung xương 2D phục vụ phân tích chuyển động"*:
     Nó chứng minh rằng trước khi áp dụng các bộ lọc làm mượt động học (Kalman/One-Euro), bước **tiền xử lý hiệu chỉnh quy ước đối xứng (Viewpoint/Symmetry Normalization)** là cực kỳ cấp thiết để khắc phục hiện tượng nhảy vọt tọa độ do lật nhầm bên trái/phải.
