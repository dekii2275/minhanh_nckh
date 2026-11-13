# Công cụ xem dự đoán YOLO-Pose

Chạy YOLO-Pose trên từng khung hình và hiển thị hộp phát hiện người cùng các khớp và khung xương dự đoán. Công cụ dùng model có sẵn `models/yolo11n-pose.pt`; không huấn luyện model hay thay đổi dữ liệu.

## Chạy với dataset

Trong PowerShell, ở thư mục gốc của dự án:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_pose_viewer\yolo_pose_viewer.py
```

Chọn bằng số trong terminal: dataset, hành động, rồi video. Nhập `q` trong menu để thoát. Chạy nhanh ví dụ:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_pose_viewer\yolo_pose_viewer.py --dataset penn --action baseball_pitch --sequence 0001
```

Danh sách dataset dựa trên các cặp video và nhãn tìm thấy trong `data/raw`; Penn là chuỗi ảnh, J-HMDB là tệp AVI. Công cụ chạy dự đoán trên toàn bộ frame của video, không cắt theo số frame nhãn khớp. Dòng `GT action` là nhãn hành động có sẵn của dataset; YOLO-Pose không dự đoán hành động. Với Penn, tốc độ phát giả định là 15 FPS do dữ liệu là chuỗi ảnh.

## Chạy với video riêng

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_pose_viewer\yolo_pose_viewer.py --video "D:\duong_dan\video.mp4"
```

Video riêng không có nhãn hành động, nên cửa sổ hiển thị `GT action: N/A`.

## Điều khiển cửa sổ

- **Space**: dừng/phát.
- **A / D**: lùi/tiến một frame và tạm dừng.
- **S**: bật/tắt lớp dự đoán YOLO.
- **R**: về frame đầu.
- **Q / Esc**: đóng video. Khi chọn từ menu sẽ trở về lựa chọn; khi dùng `--video` hoặc `--sequence` sẽ kết thúc.

Cần chọn cửa sổ video trước khi nhấn phím. Thời gian suy luận có thể khiến phát chậm hơn FPS gốc. Lần đầu tải model có thể mất thêm thời gian.

## Tùy chọn

- `--conf 0.25`: ngưỡng độ tin cậy của hộp phát hiện người, trong khoảng 0–1.
- `--device cpu` hoặc `--device 0`: chọn CPU hoặc GPU; mặc định dùng GPU CUDA số 0. Nếu CUDA không khả dụng, chương trình báo lỗi; dùng `--device cpu` để chạy trên CPU.
- `--model "đường_dẫn_model.pt"`: dùng model YOLO pose khác.
- `--data-root "đường_dẫn"`: thư mục chứa `Penn_Action` và/hoặc `jhmdb`.
- `--help`: xem các tham số.

Ví dụ chạy CPU:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_pose_viewer\yolo_pose_viewer.py --dataset penn --action baseball_pitch --sequence 0001 --device cpu
```

Môi trường `.venv` của dự án đã có các thư viện cần thiết. Công cụ cần `ultralytics`, `torch`, `opencv-python`, `numpy` và `scipy`; OpenCV cần hỗ trợ giao diện cửa sổ.

