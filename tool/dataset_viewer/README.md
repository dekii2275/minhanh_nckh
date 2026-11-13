# Xem video, nhãn và khung xương của dataset

Công cụ `tool/dataset_viewer/dataset_viewer.py` chạy trực tiếp bằng Python và mở cửa sổ OpenCV. Công cụ đọc chú thích gốc (ground truth), không chạy mô hình dự đoán.

## Chạy

Mở PowerShell tại thư mục gốc của dự án và chạy:

```powershell
.\.venv\Scripts\python.exe .\tool\dataset_viewer\dataset_viewer.py
```

Nhập số tương ứng để chọn dataset (`penn` hoặc `jhmdb`), hành động, sau đó video. Nhập `q` ở terminal để thoát. Sau khi đóng video, chương trình quay lại menu.

Có thể chọn nhanh dataset, hành động và ID video:

```powershell
.\.venv\Scripts\python.exe .\tool\dataset_viewer\dataset_viewer.py --dataset penn --action baseball_pitch --sequence 0001
```

Nếu chỉ truyền `--dataset` hoặc `--action`, chương trình sẽ yêu cầu chọn những mục còn lại. Để dùng thư mục dữ liệu khác, truyền `--data-root` tới thư mục chứa `Penn_Action` và/hoặc `jhmdb`. Xem thêm tham số bằng `--help`.

## Điều khiển trong cửa sổ video

- **Space**: tạm dừng hoặc phát tiếp.
- **A / D**: lùi / tiến một frame và tạm dừng.
- **S**: bật / tắt khung xương.
- **R**: về frame đầu.
- **Q / Esc**: đóng video và trở về lựa chọn (nếu chạy bằng `--sequence`, kết thúc chương trình).

Tên hành động, số frame và trạng thái được hiển thị trên video. Các đường xanh nối các khớp, chấm vàng biểu diễn khớp. Với Penn, chỉ khớp được đánh dấu nhìn thấy mới được vẽ; với J-HMDB, các tọa độ hợp lệ được vẽ. Penn chỉ có các ảnh frame nên công cụ dùng tốc độ phát giả định 15 FPS; J-HMDB dùng FPS của tệp video.

Nếu số frame video và số frame chú thích khác nhau, chương trình cảnh báo và giới hạn ở phần chung theo thứ tự frame. Sự khác biệt này không chứng minh hai nguồn đã đồng bộ: cần kiểm tra trực quan trước khi dùng để đánh giá nghiên cứu.

## Môi trường

Môi trường `.venv` hiện có đã chứa OpenCV, NumPy và SciPy. Nếu tạo môi trường mới, cài các gói này trong môi trường ảo và sử dụng bản `opencv-python` có hỗ trợ cửa sổ, không phải bản headless:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python numpy scipy
```

