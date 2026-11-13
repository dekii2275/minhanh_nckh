# So sánh khung xương nhãn gốc và YOLO-Pose

Công cụ hiển thị cùng một frame thành hai khung đặt cạnh nhau:

- **Bên trái — nhãn gốc (GT):** đường xanh lá, điểm khớp vàng. Penn chỉ vẽ khớp được đánh dấu nhìn thấy; J-HMDB vẽ các tọa độ hợp lệ.
- **Bên phải — YOLO-Pose:** hộp và skeleton **vàng, nhãn `CHON`** là người có box confidence cao nhất trong frame — đúng quy tắc chọn của pipeline cũ. Hộp và skeleton **xám** là những người YOLO cũng phát hiện nhưng không được chọn. Mỗi khớp hợp lệ của người được chọn có tên và confidence (ví dụ `L_wrist 0.28`); `L`/`R` là trái/phải theo nhãn COCO. Số trên hộp là **box confidence**, khác confidence của khớp. Nếu không phát hiện ai sẽ hiện `CHON: none`; nếu model không cung cấp confidence khớp, nhãn hiện `N/A`.

YOLO sử dụng bộ 17 khớp COCO; nhãn gốc Penn và J-HMDB có quy ước khớp khác. Không nên hiểu các điểm là tương ứng một-một mà chưa kiểm tra. YOLO có thể phát hiện nhiều người trong khi nhãn gốc chỉ chú thích đối tượng của dataset; công cụ không tự ghép người.

Đối chiếu người có khớp GT ở bên trái với người màu vàng ở cùng frame bên phải. Nếu là hai người khác nhau, quy tắc chọn của công cụ chọn nhầm người ở frame đó. Đi từng frame liên tiếp để xem lựa chọn có đổi người; `CHON` được tính độc lập ở mỗi frame, không phải ID theo dõi. GT chỉ phục vụ quan sát, không tham gia quyết định chọn. Công cụ chạy YOLO trên **toàn frame**, khác pipeline nghiên cứu crop và chọn detection bằng bbox GT; không coi kết quả này là dự đoán lưu của pipeline 94,86%. Công cụ chạy inference mới, nên kết quả có thể khác artifact cũ nếu phiên bản thư viện, thiết bị hoặc model khác. Ngưỡng hiển thị khớp không thay đổi quy tắc chọn hộp.

## Chạy

Tại thư mục gốc của dự án trong PowerShell:

```powershell
.\.venv\Scripts\python.exe .\tool\pose_comparison_viewer\pose_comparison_viewer.py
```

Chọn bằng số trong terminal: dataset, hành động, video. Nhập `q` tại menu để thoát. Ví dụ mở trực tiếp:

```powershell
.\.venv\Scripts\python.exe .\tool\pose_comparison_viewer\pose_comparison_viewer.py --dataset penn --action baseball_pitch --sequence 0001
```

Ví dụ đã quan sát có nhiều người và có thể chọn nhầm người:

```powershell
.\.venv\Scripts\python.exe .\tool\pose_comparison_viewer\pose_comparison_viewer.py --dataset penn --action baseball_pitch --sequence 0137
```

Công cụ dùng `models/yolo11n-pose.pt` và mặc định chạy GPU CUDA 0. Nếu muốn chạy CPU, thêm `--device cpu`.

## Điều khiển cửa sổ video

- **Space**: dừng/phát.
- **A / D**: lùi/tiến một frame, đồng thời tạm dừng.
- **G**: bật/tắt khung xương nhãn gốc.
- **Y**: bật/tắt khung xương YOLO.
- **C**: bật/tắt tên và confidence từng khớp của người được chọn.
- **R**: về frame đầu.
- **Q / Esc**: đóng video và quay về lựa chọn. Khi dùng `--sequence`, chương trình kết thúc.

Chọn cửa sổ video trước khi nhấn phím. `GT action` là nhãn hành động của dataset, không phải dự đoán của YOLO. Các ảnh Penn được phát giả định 15 FPS. Tốc độ phát thực tế còn phụ thuộc thời gian suy luận.

Nếu số frame của video và nhãn gốc khác nhau, công cụ cảnh báo và chỉ xem phần chung theo thứ tự frame. Không thể suy ra hai nguồn đồng bộ chỉ từ thứ tự này; hãy kiểm tra trực quan trước khi dùng kết quả cho nghiên cứu.

## Tùy chọn

- `--conf 0.25`: ngưỡng phát hiện người của YOLO.
- `--kpt-conf 0.0`: ngưỡng hiển thị từng khớp YOLO; mặc định hiện cả khớp confidence thấp có tọa độ hợp lệ. Khớp thiếu tọa độ không thể vẽ. Ví dụ `--kpt-conf 0.3` chỉ hiện khớp có confidence từ 0,3; nếu model không xuất confidence, các khớp hợp lệ vẫn hiện `N/A`.
- `--device 0` hoặc `--device cpu`: GPU hoặc CPU.
- `--model "đường_dẫn.pt"`: model YOLO pose dùng 17 khớp COCO.
- `--data-root "đường_dẫn"`: thư mục chứa `Penn_Action` và/hoặc `jhmdb`.
- `--help`: danh sách tham số.

