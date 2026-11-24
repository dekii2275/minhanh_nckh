# Gán nhãn loại lỗi khung xương theo video (bước 1)

Công cụ gồm một cửa sổ OpenCV. Bên trái là video Penn Action (official test), vẽ chồng hai khung xương:

- **Annotation**: đường xanh lá, chấm xanh nhạt.
- **YOLO**: đường cam, đúng pose đã lưu mà pipeline đưa vào mạng (crop bbox GT, fixed swap), không chạy lại YOLO. Khớp có confidence ≥ 0,5 vẽ chấm đặc; khớp có confidence < 0,5 vẽ vòng rỗng màu vàng; khớp có confidence < 0,2 vẽ vòng rỗng màu đỏ.

Bên phải là bảng tick loại lỗi. Danh sách video lấy từ `outputs/error_attribution_20261124/phieu_gan_nhan_loi_buoc1.csv` (91 video gồm nhóm A, B và D đối chứng).

## Chạy

Mở PowerShell tại thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe .\tool\error_label_tool\error_label_tool.py
```

Một số cách chạy khác:

- `--nhom A`: chỉ làm 47 video nhóm A.
- `--start 0415`: mở thẳng một video.

Công cụ tự mở video đầu tiên **chưa xong**, nên tắt đi bật lại vẫn làm tiếp được.

## Thông tin trên màn hình

- **Header**:
  - Dự đoán của từng seed: YOLO và GT, với BiLSTM và CascadeFormer.
  - Trạng thái YOLO ở frame hiện tại: `matched` / `no_person` / `no_iou_match`.
  - Confidence trung bình và sai số trung bình so với GT.
- **Timeline** dưới video, mỗi ô là một frame. Bấm hoặc kéo trên timeline để tua.

  | Màu | Ý nghĩa |
  |---|---|
  | Xanh | Sai số < 0,1 đường chéo bbox |
  | Vàng | Sai số < 0,2 |
  | Đỏ | Sai số ≥ 0,2 |
  | Xám | YOLO mất người |
  | Nâu | Không có GT |

  Vạch xanh ngọc phía trên timeline là đoạn lỗi đã đánh dấu.

## Phím và chuột

| Thao tác | Phím / chuột |
|---|---|
| Phát / dừng | Space |
| Lùi / tiến 1 frame | A / D hoặc ← / → |
| Lùi / tiến 10 frame | J / L |
| Về frame đầu | R |
| Tốc độ phát (0,25×–2×) | + / − |
| Bật/tắt GT, YOLO | G, Y |
| Đường nối khớp GT ↔ YOLO (thấy độ lệch) | E |
| Tên khớp + confidence | C |
| Bbox GT | B |
| Xem song song GT \| YOLO | V |
| Tick loại lỗi (10 loại, gồm 9 = annotation/bbox GT lệch, 0 = người bị cắt khỏi khung) | 1–9, 0 hoặc **chuột trái** vào nút |
| Đặt làm **lỗi chính** | **Chuột phải** vào nút loại lỗi |
| Đặt frame bắt đầu / kết thúc đoạn lỗi = frame hiện tại | [ / ] (chuột phải vào nút để xóa) |
| Ghi chú tiếng Việt (hộp thoại Windows) | T |
| Lưu, đánh dấu xong và sang video tiếp | Enter / N (cần ít nhất 1 loại lỗi) |
| Video trước | P |
| Lưu và thoát | Q / Esc |

Nếu không chọn lỗi chính, loại lỗi tick đầu tiên được tính là lỗi chính. Mức độ và khớp bị ảnh hưởng chỉ cần bấm vào nút trên bảng.

## Duyệt nhãn nháp của Claude

Nhãn nháp cho cả 91 video nằm trong `outputs/error_attribution_20261124/nhan_loi_buoc1_claude.csv`. Ghi chú của nhãn nháp bắt đầu bằng `[Claude]`, và tất cả đều `da_xong = 0`. Mở công cụ trên file này để duyệt:

```powershell
.\.venv\Scripts\python.exe .\tool\error_label_tool\error_label_tool.py --out outputs\error_attribution_20261124\nhan_loi_buoc1_claude.csv
```

Mỗi video hiện sẵn nhãn nháp. Sửa nếu sai, rồi nhấn Enter để xác nhận. `da_xong = 1` nghĩa là anh/chị đã duyệt video đó.

## Kết quả

Mỗi lần thay đổi, công cụ ghi lại toàn bộ file `outputs/error_attribution_20261124/nhan_loi_buoc1.csv`. Các cột:

`id, nhom, action, loi_chinh, loi_phu, frame_bat_dau, frame_ket_thuc, khop_bi_anh_huong, muc_do_1_3, ghi_chu, da_xong, thoi_gian_utc`

Nhiều giá trị trong một ô được ngăn cách bằng `;`. File phiếu gốc không bị sửa.

## Lưu ý

- Pose YOLO là bản đã lưu ở `outputs/confidence_confirmation_penn_20261105/test_pose/poses`, áp fixed swap trái/phải giống pipeline. Công cụ không kiểm tra hash.
- Nếu số ảnh, số frame GT và số frame pose khác nhau, công cụ chỉ hiển thị phần chung và in cảnh báo ra terminal.
- Annotation GT cũng có thể sai. Nếu nghi ngờ GT, ghi vào ô ghi chú.
