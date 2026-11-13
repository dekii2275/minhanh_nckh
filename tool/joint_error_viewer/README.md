# Phân tích lỗi cổ tay và cổ chân

Công cụ xếp hạng video Penn Action validation và xem từng frame bằng hai khung đồng bộ: GT bên trái, pose YOLO bên phải. Dùng pose **đã lưu của thí nghiệm crop GT**, không chạy YOLO lại. Khớp đang phân tích có màu cam; vòng tròn xanh lam ở khung YOLO chỉ vị trí GT để đối chiếu.

## Chạy

Trong PowerShell tại thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe .\tool\joint_error_viewer\joint_error_viewer.py
```

Chọn khớp, nhóm, rồi video trong terminal. Video xếp theo **số frame trong nhóm giảm dần**, không phải tỷ lệ; danh sách cũng ghi số frame có GT hợp lệ để tham khảo. Khi chọn video, cửa sổ mở tại frame đầu tiên thuộc nhóm và tạm dừng. Tải và xếp hạng toàn validation có thể mất một lúc.

Các nhóm:

- `low_wrong`: confidence < 0,30 và sai số > 0,2 đường chéo bbox GT.
- `high_wrong`: confidence >= 0,70 và sai số > 0,2 đường chéo bbox GT.
- `low_correct`: confidence < 0,30 và sai số <= 0,2 đường chéo bbox GT.
- `missing`: GT hợp lệ nhưng không có tọa độ pose YOLO hợp lệ, bao gồm frame không ghép được người.
- `all`: tất cả frame, bao gồm frame không thể so với GT.

Ba nhóm đầu chỉ xét frame ghép được người và có cả GT lẫn tọa độ YOLO hợp lệ. Ngưỡng chỉ để **chọn ca xem**, không kết luận nguyên nhân hay ngưỡng tối ưu. Có thể thay bằng `--low`, `--high`, `--pck`.

## Điều khiển và ghi nhận

Chọn cửa sổ video trước khi nhấn phím:

- **Space:** phát/dừng; **A/D:** lùi/tiến một frame; **N/P:** nhảy tới/lùi frame tiếp theo thuộc nhóm đã chọn; **Q/Esc:** trở về danh sách.
- **Bấm trực tiếp nút loại lỗi ở bên phải cửa sổ**, hoặc nhấn phím **1–7**: che khuất, nhảy đột ngột, thiếu, nghi swap, nghi chọn sai người, chưa rõ, hoặc lỗi khác. Video tự dừng và một hộp thoại xuất hiện để nhập ghi chú tiếng Việt. Nhấn **Luu** để lưu (có thể để trống ghi chú), hoặc **Huy** để không lưu. Sau đó tiếp tục dùng Space hoặc A/D/N/P để xem.
- Hộp thoại sử dụng Windows Forms qua PowerShell đi kèm Windows; không cần cài Tkinter. Nếu không mở được hộp thoại, công cụ báo lỗi ở terminal và tiếp tục hiển thị video.
- Mỗi lần xác nhận thêm một dòng vào `outputs/joint_error_review/annotations_v2.csv`; có thể gắn nhiều nhãn hoặc ghi lại cùng frame. Đây là **quan sát của người xem**, không phải phân loại tự động đã được xác nhận. File `annotations.csv` của phiên bản trước được giữ nguyên; file mới có thêm cột `note`. Nếu dùng `--notes`, hãy chọn file mới hoặc file có đúng định dạng v2.

CSV lưu video, frame (đánh số từ 1), khớp, nhóm đang lọc, nhãn, ghi chú, confidence, sai số khi đo được, trạng thái frame, ngưỡng và hash nguồn. Có thể chọn file khác bằng `--notes`. Bộ lọc chỉ điều khiển xếp hạng và phím N/P: sau khi nhấn A/D hoặc phát, frame hiện tại có thể không thuộc nhóm hiển thị ở tiêu đề.

## Chạy trực tiếp và xuất danh sách

```powershell
# Cổ tay trái, confidence cao nhưng sai: xem xếp hạng
.\.venv\Scripts\python.exe .\tool\joint_error_viewer\joint_error_viewer.py --list --joint left_wrist --group high_wrong

# Mở trực tiếp frame 42 của video 0137
.\.venv\Scripts\python.exe .\tool\joint_error_viewer\joint_error_viewer.py --sequence 0137 --frame 42 --joint left_wrist --group low_wrong

# Xuất danh sách vào file mới; chương trình không ghi đè file đã có
.\.venv\Scripts\python.exe .\tool\joint_error_viewer\joint_error_viewer.py --joint right_ankle --group low_wrong --export outputs\ankle_low_wrong.csv
```

`--sequence` không có `--frame` sẽ mở frame đầu tiên phù hợp với `--group`. `--save-frame <đường_dẫn.png>` cùng `--sequence` lưu ảnh thay vì mở cửa sổ, không ghi đè ảnh có sẵn. `--help` liệt kê toàn bộ tùy chọn.

## Phạm vi và giới hạn

Pose nguồn sử dụng bbox GT để crop và chọn detection theo IoU; chưa phải hệ chọn người tự động. Tọa độ và confidence được **swap trái/phải cố định** như thí nghiệm EDA; tên trái/phải theo mapping đó, không bảo đảm đúng giải phẫu ở mọi video. GT dùng cho so sánh và là tham chiếu có thể có sai lệch; `high_wrong` không chứng minh YOLO sai trong mọi trường hợp. Frame không có GT hợp lệ không được tự động xếp là đúng hay sai.

Khi mở video, công cụ kiểm tra hash annotation và toàn bộ ảnh với cache đã lưu. Xếp hạng kiểm tra annotation và manifest nhưng không đọc hash toàn bộ ảnh cho từng video; xác minh ảnh được thực hiện lúc mở video. Confidence khớp khác confidence của hộp người. Penn phát mặc định 15 FPS, là tốc độ giả định.
