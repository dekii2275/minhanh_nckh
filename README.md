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
ruff check src tests
ruff format --check src tests
python -m pytest -q
```

Dữ liệu Penn Action, manifest thí nghiệm mới, checkpoint, cache pose, đặc trưng `.npz` và phần lớn hình/video không có trong Git. Một số kiểm thử tích hợp cần các artifact đó và sẽ bỏ qua khi chưa có. Phải chuẩn bị đúng dữ liệu/cache/checkpoint và GPU CUDA trước khi chạy lại pipeline. Kiểm tra protocol, hash và chọn thư mục output mới; các script thí nghiệm từ chối ghi đè kết quả hiện có.
