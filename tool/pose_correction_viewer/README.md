# Xem pose YOLO trước và sau hiệu chỉnh

Phát lại kết quả đã lưu trên **253 video Penn Action validation**; công cụ không chạy lại YOLO hoặc huấn luyện mô hình.

```powershell
.\.venv\Scripts\python.exe .\tool\pose_correction_viewer\pose_correction_viewer.py
```

Ví dụ xem video 0018, frame 52 — frame không ghép được GT nhưng có khớp được nội suy:

```powershell
.\.venv\Scripts\python.exe .\tool\pose_correction_viewer\pose_correction_viewer.py --sequence 0018 --frame 52 --method interpolate3
```

## Bốn khung

1. **GT:** skeleton annotation và bbox GT.
2. **YOLO gốc:** skeleton của detection được GT hỗ trợ chọn, trước swap.
3. **Trước hiệu chỉnh:** cùng pose YOLO sau swap trái/phải cố định; đây là đối chứng trong thí nghiệm.
4. **Sau hiệu chỉnh:** pose sau một trong bốn phép đã thử, áp dụng trên cùng pose đã swap.

Xanh lam nhạt/tím là trái/phải theo thứ tự trong mã; vòng rỗng là GT, chấm đặc là YOLO. **Mũi tên vàng** chỉ khớp di chuyển ít nhất 1 pixel; **vòng cam** đánh dấu giá trị được nội suy, không phải detection mới. Ngưỡng 1 pixel chỉ ảnh hưởng hiển thị/tìm frame, không thay đổi kết quả. Frame không ghép được GT không có pose YOLO ở khung 2–3, nhưng nội suy có thể cung cấp pose ở khung 4.

Số sai số frame và video dùng cùng tập khớp hợp lệ của báo cáo. Khi frame không có khớp chung, sai số trước/sau là **N/A**; nếu pose sau có khớp, dòng **GT riêng** chỉ cho biết sai số riêng trên các khớp GT nhìn thấy và không so trực tiếp với N/A. Giá trị trung bình theo video tính trên các frame/khớp chung, không phải trung bình mọi frame bằng nhau.

## Điều khiển

- **Space:** phát/dừng; **A/D:** lùi/tiến 1 frame; **R:** về đầu.
- **M:** đổi giữa nội suy tối đa 3 frame, trung bình 5 frame, trung vị 5 frame và nội suy + trung bình 5 frame.
- **E:** đổi bộ lọc: có thay đổi, có nội suy, sai số thấp hơn, sai số cao hơn, không ghép GT, tất cả. **N/P** nhảy tới/lùi frame thỏa bộ lọc.
- **O:** ẩn/hiện vòng GT; **U:** ẩn/hiện mũi tên và vòng nội suy; **I:** in tọa độ từng khớp trong terminal; **Q/Esc:** đóng video.

Thanh thời gian đánh dấu **cam** nếu có khớp nội suy, **xanh** nếu có khớp di chuyển ít nhất 1 pixel và **xám** nếu không có thay đổi lớn. Các frame màu xám vẫn có thể có thay đổi dưới 1 pixel. Khi chuyển phương pháp, dự đoán hành động **cấp video** của ba checkpoint được in trong terminal; đó không phải dự đoán theo từng frame.

`--list` liệt kê video; `--method` nhận `interpolate3`, `mean5`, `median5`, `interpolate3_mean5`. Có thể lưu một frame bằng `--save-frame <duong_dan_anh>` khi dùng `--sequence`.

## Giới hạn

GT hỗ trợ chọn người theo từng frame; GT không được dùng để nội suy hay làm mượt. Swap cố định là đối chứng, không đúng với mọi clip. Hai phép làm mượt dùng frame tương lai và chỉ tác động khi đủ khớp confidence ≥ 0,3; nội suy chỉ lấp khoảng 1–3 frame có điểm hợp lệ hai đầu. Công cụ kiểm tra hash ảnh, annotation, pose nguồn và tái tính hiệu chỉnh/sai số trước khi hiển thị. Các kết quả là thăm dò trên validation, không phải hệ tự động đã kiểm định độc lập.
