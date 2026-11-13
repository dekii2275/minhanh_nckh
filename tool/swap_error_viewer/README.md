# Xem 11 video swap làm tăng sai số

Công cụ xem lại **11 video validation** mà sai số pixel trung bình của YOLO với GT tăng khi tráo trái/phải. Chọn video trong menu:

```powershell
.\.venv\Scripts\python.exe .\tool\swap_error_viewer\swap_error_viewer.py
```

Hoặc mở trực tiếp một video, ví dụ 0377:

```powershell
.\.venv\Scripts\python.exe .\tool\swap_error_viewer\swap_error_viewer.py --sequence 0377 --frame 30
```

Thêm `--list` để xem cả 11 video. Có thể mở video validation khác bằng `--sequence` nếu video đó có khớp chung để so sánh.

## Cách đọc

Ba khung cùng frame: **GT**, **YOLO giữ nguyên**, **YOLO swap**. Vàng là bbox YOLO được GT hỗ trợ chọn; xám là detection khác; xanh lá là bbox GT. Khớp trái theo tên trong mã màu xanh lam nhạt, khớp phải màu tím. Trong hai khung YOLO, **chấm đặc là YOLO, vòng tròn rỗng là GT**. Bấm **O** để ẩn/hiện vòng GT.

Chữ `LS`, `RE`... là tên theo thứ tự trong mã: `L`/`R` là trái/phải; `S/E/W/H/K/A` là vai, khuỷu, cổ tay, hông, gối, cổ chân. Màu và nhãn thể hiện correspondence đang được thử, không chứng minh bên giải phẫu đúng. Swap chỉ đổi nhãn các cặp trái/phải, không lật ảnh.

Phía trên cho biết sai số pixel trung bình **của frame** và **của cả video**, số khớp GT hợp lệ, số khớp chung, số detection, IoU GT của box được chọn, có chọn khác nhánh confidence không, và số khớp confidence dưới 0,5. Khác nhánh confidence không chứng minh bên nào chọn đúng người; IoU cũng chỉ là proxy.

Sai số dùng cùng các khớp hợp lệ ở cả ba điều kiện confidence/GT hỗ trợ/GT hỗ trợ + swap, đúng như báo cáo gốc. Frame không có khớp chung hiện **N/A**, không coi là đúng hay sai. Video được chọn theo trung bình trên toàn bộ khớp chung, nên một trong 11 video vẫn có thể chứa các frame mà swap tốt hơn.

Thanh thời gian: **cam** là sai số giữ nguyên thấp hơn ít nhất 1 px; **xanh** là swap thấp hơn ít nhất 1 px; **trắng** là chênh lệch không quá 1 px; **xám** là không đủ dữ liệu. Vạch trắng dài là frame hiện tại. Ngưỡng 1 px chỉ dùng để tô màu/lọc frame, không ảnh hưởng việc chọn 11 video; có thể thay bằng `--margin`.

## Điều khiển

- **Space:** phát/dừng; **A/D:** lùi/tiến 1 frame; **R:** về đầu.
- **E:** đổi bộ lọc tìm frame: giữ nguyên tốt hơn, swap tốt hơn, không ghép GT, chọn khác confidence, dưới 6 khớp GT hợp lệ, tất cả.
- **N/P:** nhảy tới/lùi về frame thỏa bộ lọc; **O:** ẩn/hiện vòng GT; **Q/Esc:** đóng video.

Công cụ chỉ đọc detection và ảnh đã lưu, không chạy YOLO lại. Khi mở video, công cụ kiểm tra hash ảnh, annotation và tính lại sai số so với bảng đã lưu. Bộ dữ liệu là validation đã tham gia phát triển trước đây; GT hỗ trợ và kết quả swap dùng để chẩn đoán, không chứng nhận nguyên nhân hay thuật toán tự động.
