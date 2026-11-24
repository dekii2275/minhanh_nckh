# Nghiên cứu nhận diện hành động từ pose

Dự án bắt đầu lại nghiên cứu về ảnh hưởng của hiệu chỉnh skeleton 2D theo thời gian đến nhận diện hành động. Xem [kế hoạch dự kiến](report/02112026_nckh/KE_HOACH_NGHIEN_CUU.md) và [tổng hợp tài liệu](report/02112026_nckh/TONG_HOP_BA_BAI_VA_HUONG_NGHIEN_CUU.md). Phương pháp và kết luận chưa được chốt.

Thí nghiệm gần đây so sánh nội suy khớp nhảy với pose gốc trên Penn Action: train toàn bộ official train và đánh giá official test. Official test đã tham gia nghiên cứu lịch sử; bbox GT vẫn hỗ trợ chọn người và quy tắc nội suy dùng frame tương lai. Báo cáo và kết quả mới chỉ được giữ trong workspace cục bộ.

- `report/`: tài liệu nghiên cứu theo ngày.
- `reports/`: báo cáo lịch sử từ phiên bản trước của repository.
- `outputs/`: kết quả và artifact; kiểm tra nguồn và giai đoạn trước khi sử dụng.
- `src/`, `configs/`, `tests/`, `data/` và `models/`: mã, cấu hình và dữ liệu sẵn có, được giữ lại để rà soát khi cần; không mặc định chúng phù hợp với hướng nghiên cứu mới.
- `luu_tru_nghien_cuu_truoc/`: tài liệu và artifact lịch sử có thể được lưu tại workspace để truy vết; không mặc định toàn bộ dữ liệu có trong Git.
- `third_party/`: mã và checkpoint tham khảo được tải riêng tại workspace, không đưa vào Git.
- `research_snapshots/2026-11-13/`: bản mã gốc trước chuẩn hóa để đối chiếu hash đã ghi trong artifact; xem [ghi chú snapshot](research_snapshots/2026-11-13/README.md).

Khi dùng lại kết quả hoặc mã cũ, cần kiểm tra nguồn, protocol, split và khả năng tái lập trước khi đưa vào nghiên cứu mới. Đường dẫn bên trong một số mã cũ vẫn trỏ tới `outputs/` và có thể cần khôi phục artifact hoặc điều chỉnh trước khi chạy lại.

## Cài đặt và kiểm tra

Tạo môi trường Python 3.10, cài `requirements-dev.txt`, sau đó chạy:

```bash
ruff check src tests tool notebooks report/02112026_nckh
ruff format --check src tests tool notebooks report/02112026_nckh
python -m pytest -q
```

Dữ liệu Penn Action, manifest thí nghiệm mới, checkpoint, cache pose, đặc trưng `.npz` và phần lớn hình/video không có trong Git. Một số kiểm thử tích hợp cần các artifact đó và sẽ bỏ qua khi chưa có. Phải chuẩn bị đúng dữ liệu/cache/checkpoint và GPU CUDA trước khi chạy lại pipeline. Kiểm tra protocol, hash và chọn thư mục output mới; các script thí nghiệm từ chối ghi đè kết quả hiện có.

## Script và công cụ

- Trên Windows, chạy `CHAY_E1_E2.bat` để chạy E1 và E2 với Python trong `.venv`. Nhật ký nằm ở `outputs/e2_pose_x_retrain_20261124/run_log.txt`. Xem cách chia dữ liệu và các artifact đầu vào trong [protocol Penn Action](report/13112026_nckh/PROTOCOL_THONG_NHAT_PENN_2026-11-15.md). Các bước dựng mô hình CascadeFormer cần mã tác giả ở `third_party/CascadeFormer`. [Hướng dẫn chuẩn bị CascadeFormer](report/02112026_nckh/CASCADEFORMER_DANH_GIA_VA_CHAY_THU.md)
- `tool/` có các trình xem dữ liệu, pose và tracking. [Hướng dẫn công cụ gán nhãn lỗi](tool/error_label_tool/README.md)
- `notebooks/` có sổ tay EDA và đối chiếu dự đoán. Cài `ipykernel` để chạy trong Jupyter. Sổ tay xem video cần thêm `ipywidgets`, FFmpeg trong `PATH` và các artifact được liệt kê trong sổ tay.
