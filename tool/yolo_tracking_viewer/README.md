# Xem YOLO tracking nhiều người

Công cụ chạy **YOLO11n-Pose + ByteTrack hoặc BoT-SORT** trên từng frame và hiển thị ID theo dõi. GT chỉ được vẽ ở khung bên trái để đối chiếu; **GT không tham gia phát hiện, gán ID hay chọn người**.

## Chạy

Trong PowerShell, tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_tracking_viewer\yolo_tracking_viewer.py
```

Chọn hành động và video Penn trong menu. Hoặc chạy thẳng video nhiều người 0137:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_tracking_viewer\yolo_tracking_viewer.py --sequence 0137
```

Đổi tracker để xem lại cùng video:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_tracking_viewer\yolo_tracking_viewer.py --sequence 0137 --tracker botsort
```

Mặc định là `bytetrack`. Ví dụ mở frame 52 của video 0018:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_tracking_viewer\yolo_tracking_viewer.py --sequence 0018 --frame 52
```

Tracker vẫn xử lý **lần lượt từ frame 1** trước khi hiện frame 52; việc này có thể mất vài giây. Có thể dùng video hoặc thư mục ảnh riêng, không có GT:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_tracking_viewer\yolo_tracking_viewer.py --video "D:\duong_dan\video.mp4"
```

## Thử chọn người chuyển động nhiều

Mở video rồi nhấn **M** để bật/tắt tự chọn, hoặc bật ngay từ đầu:

```powershell
.\.venv\Scripts\python.exe .\tool\yolo_tracking_viewer\yolo_tracking_viewer.py --sequence 0137 --auto-motion
```

Mỗi ID đang xuất hiện được tính điểm `M` từ tối đa 20 cặp frame gần nhất. Điểm kết hợp độ dời trung bình của các khớp YOLO so với tâm box (75%) và độ dời tâm box (25%), chia theo kích thước box. Cần ít nhất 5 cặp frame liên tiếp hợp lệ; `M=--` nghĩa là chưa đủ dữ liệu. Nhấn **I** để xem điểm và số cặp trong cửa sổ dòng lệnh. Có thể chỉnh `--motion-window`, `--motion-min-pairs` và `--motion-conf`.

ID có điểm cao nhất được tô vàng; lựa chọn có thể đổi theo từng frame. Phương pháp chỉ dùng YOLO và lịch sử track, không dùng GT hay frame tương lai. Click box để chuyển về chọn tay; **C** bỏ chọn và tắt tự động.

Đây là giả thuyết để quan sát, chưa được kiểm chứng độ chính xác: người khác, camera chuyển động, khớp nhiễu, mất detection hoặc đổi ID đều có thể làm chọn sai. Người thực hiện cũng có thể đứng yên trong một đoạn; tracking không tự tạo pose khi YOLO không phát hiện được người.

## Cách xem

- Bên trái là ảnh kèm **GT annotation** nếu có; bên phải là mọi detection YOLO và ID tracker.
- `ID 3` là nhãn track, không phải tên người hay nhãn hành động; `ID ?` là detection hiện chưa được tracker gán ID. Màu giúp nhìn nhanh và có thể lặp lại, nên phân biệt bằng số ID.
- **Click box bên phải** để chọn một ID cần quan sát. Khi ID đó mất, công cụ báo vắng và không tự chuyển sang người khác. Bấm **C** để bỏ chọn; có thể click lại ID khác.
- Đường màu là quỹ đạo tâm box trong tối đa 40 frame gần nhất; đường không nối qua các khoảng không thấy ID. ID có thể đổi người hoặc đổi số, nên cần quan sát ảnh, không mặc định track đúng.
- Box và pose được vẽ theo detection YOLO của frame hiện tại; tracker chỉ gán ID cho detection. Khi không có detection gắn với ID, công cụ không tự vẽ pose dự đoán thay thế.
- Số sau ID là confidence của **box detection**. Ngưỡng vẽ khớp không quyết định điểm box hay gán track.

## Điều khiển

- **Space:** phát/dừng; **A/D:** lùi/tiến frame; **R:** về đầu; **Q/Esc:** đóng.
- **G:** ẩn/hiện GT; **P:** ẩn/hiện pose; **H:** ẩn/hiện quỹ đạo; **C:** bỏ chọn ID.
- Lùi frame dùng kết quả đã lưu trong bộ nhớ. Tracker chỉ được cập nhật theo thứ tự tiến, không chạy ngược; mở video mới sẽ tạo tracker mới.

## Cấu hình và giới hạn

- Mặc định confidence detection là **0,10** để tracker có thể nhận detection yếu; khác mức **0,25** trong thí nghiệm YOLO trước. Dùng `--conf 0.25` để xem theo ngưỡng cũ, nhưng các detection dưới ngưỡng này không thể hỗ trợ track.
- `--kpt-conf 0.5` chỉ lọc **khớp được vẽ**, không đổi detection hay tracker. `--device cpu` hoặc `--device 0` chọn CPU/GPU; mặc định chọn GPU nếu có. Thời gian suy luận có thể làm phát chậm; FPS Penn 15 chỉ là tốc độ hiển thị giả định.
- Nếu thư viện `lap` không có, công cụ dùng bộ ghép chi phí có ngưỡng dựa trên SciPy trong tiến trình. Kết quả có thể khác chi tiết/tie-breaking so với LAPJV chuẩn; tên `scipy-fallback` hiện ở đầu cửa sổ. Không tự tải hay cài gói khi mở công cụ.
- Đây là công cụ quan sát, chưa đo tỷ lệ ID switch hay độ chính xác của cách tự chọn người.

Dùng `--save-frame <anh.jpg>` cùng `--sequence` hoặc `--video` nếu muốn lưu ảnh tại frame được chọn, thay vì mở cửa sổ.
