# Xem lựa chọn người: YOLO và GT hỗ trợ

Công cụ phát lại **detection đã lưu của thí nghiệm 253 video validation seed 0**, không chạy YOLO lại. Bốn khung là cùng một ảnh:

1. **GT:** skeleton và bbox annotation màu xanh.
2. **YOLO confidence:** người có box confidence lớn nhất; cùng quy tắc nhánh YOLO cũ.
3. **GT hỗ trợ:** detection có IoU với bbox GT cao nhất, nếu IoU ≥ 0,30. Đây là phép chẩn đoán có dùng GT, không phải thuật toán tự chọn người.
4. **Đầu vào mô hình:** chỉ 12 khớp từ detection được GT hỗ trợ chọn, vẽ ở tọa độ pixel gốc. Không vẽ 5 khớp đầu COCO hoặc người khác, và không dùng ngưỡng confidence để quyết định khớp nào đi vào phép chuẩn hóa.

Trong hai khung bên phải, **vàng = người nhánh đó chọn**, xám = các detection còn lại, hộp xanh = GT. Các con số trên detection là box confidence; IoU với bbox GT nằm trên tiêu đề. Một detection giống nhau ở hai khung nghĩa là hai nhánh chọn cùng người theo chỉ số detection, chưa bảo đảm đúng người.

Nếu hiện **YOLO KHONG PHAT HIEN** thì frame không có detection nào. Nếu hiện **CO DETECTION, KHONG GHEP GT**, YOLO có detection nhưng không box nào có IoU ≥ 0,30. Nhánh GT hỗ trợ khi đó ghi **MISSING**, không vẽ người màu vàng. `GT BBOX KHONG HOP LE` là trường hợp bbox không thể dùng để ghép. IoU theo bbox chỉ gần đúng; nhiều người chồng lấp hoặc bbox lỗi có thể gây ghép nhầm.

Mặc định chỉ vẽ khớp có confidence ≥ 0,5 để nhìn rõ; đây là **ngưỡng hiển thị**, còn thí nghiệm đã lưu không dùng ngưỡng confidence khớp. Không có swap trái/phải trong hai nhánh. Kết quả classifier in ở terminal là dự đoán **cấp video** đã lưu cho ba checkpoint, không phải dự đoán của từng frame.

Khung thứ tư không áp dụng `--kpt-conf`. Khi không ghép được GT, khung ghi **24 số 0 trước lấy mẫu**: frame đó được đánh dấu thiếu, rồi chuyển thành 0 sau chuẩn hóa. Khi có detection, YOLO có thể trả một số khớp `(0,0)`; bộ chuẩn hóa cũ vẫn nhận các tọa độ đó và không có mask riêng. Khung ảnh không thể vẽ khớp ở tọa độ 0 và ghi rõ số khớp `(0,0)`.

Classifier không nhận trực tiếp từng frame gốc: sau chuẩn hóa, chuỗi được **nội suy lại thành 64 bước**. Vì vậy một frame rỗng có thể được trộn với frame bên cạnh ở một bước, hoặc không đóng góp trực tiếp vào bước nào. Bấm **I** để xem tại terminal tọa độ 12 khớp, vector 24 số trước lấy mẫu và các vector 64 bước thực sự nhận đóng góp từ frame hiện tại. Các số ở đây được dựng lại từ đúng detection đã lưu, không phải YOLO chạy mới.

## Chạy

Mở PowerShell ở thư mục dự án:

```powershell
.\.venv\Scripts\python.exe .\tool\person_selection_viewer\person_selection_viewer.py
```

Chọn nhóm và video trong terminal. Sau khi cửa sổ mở, click vào cửa sổ trước khi bấm phím. Ví dụ đi thẳng đến hai trường hợp có thật trong kết quả:

```powershell
# Hai nhánh chọn hai người khác nhau
.\.venv\Scripts\python.exe .\tool\person_selection_viewer\person_selection_viewer.py --sequence 0018 --frame 13

# YOLO không phát hiện người ở frame đầu
.\.venv\Scripts\python.exe .\tool\person_selection_viewer\person_selection_viewer.py --sequence 0399 --frame 1

# YOLO có detection nhưng không ghép được GT
.\.venv\Scripts\python.exe .\tool\person_selection_viewer\person_selection_viewer.py --sequence 0018 --frame 52
```

Các video khác có thể liệt kê bằng `--list`. Trong danh sách, `no_match_including_invalid_gt` tính cả frame GT bbox không hợp lệ; trong cửa sổ các trạng thái này được phân biệt.

## Điều khiển

- **Space:** phát/dừng. Ban đầu đang dừng.
- **A / D:** lùi/tiến một frame.
- **E:** đổi loại sự kiện cần tìm: `problem` (mọi vấn đề), `changed`, `no_detection`, `no_match`, `invalid_gt`, `all`.
- **N / P:** nhảy tới/lùi về frame kế tiếp/trước đó thuộc loại sự kiện đang chọn, không tự vòng qua đầu/cuối.
- **I:** in chi tiết đầu vào mô hình của frame hiện tại và các bước 64-frame liên quan.
- **R:** về frame đầu; **Q / Esc:** đóng video. Chạy qua menu thì quay lại menu.

Phím E/N/P chỉ giúp tìm frame; A/D giúp quan sát các frame liên tiếp trước và sau sự kiện. Penn được phát ở tốc độ giả định 15 FPS.

## Nguồn và giới hạn

Đọc `outputs/person_selection_validation_20261102/detections/*.npz`, `frame_matches_iou_0_3.csv`, `videos_iou_0_3.csv` và `predictions_iou_0_3.csv`. Chỉ 253 video validation trong artifact có thể xem. Khi mở video, chương trình kiểm tra hash annotation, hash toàn bộ ảnh của video, số frame và index detection; nếu dữ liệu đã thay đổi, chương trình báo lỗi thay vì hiển thị pose cũ trên ảnh khác. Việc kiểm tra hash ảnh có thể mất vài giây.

GT chỉ dùng để minh họa và chọn detection trong nhánh chẩn đoán. Không suy ra mọi frame được ghép là đúng hoặc mọi frame không ghép là YOLO không phát hiện người.
