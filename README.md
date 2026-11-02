# Nghiên cứu nhận diện hành động từ pose

Dự án bắt đầu lại nghiên cứu về ảnh hưởng của hiệu chỉnh skeleton 2D theo thời gian đến nhận diện hành động. Xem [kế hoạch dự kiến](report/02112026_nckh/KE_HOACH_NGHIEN_CUU.md) và [tổng hợp tài liệu](report/02112026_nckh/TONG_HOP_BA_BAI_VA_HUONG_NGHIEN_CUU.md). Phương pháp và kết luận chưa được chốt.

- `report/`: tài liệu nghiên cứu theo ngày.
- `reports/`: báo cáo lịch sử từ phiên bản trước của repository.
- `outputs/`: kết quả và artifact; kiểm tra nguồn và giai đoạn trước khi sử dụng.
- `src/`, `configs/`, `tests/`, `data/` và `models/`: mã, cấu hình và dữ liệu sẵn có, được giữ lại để rà soát khi cần; không mặc định chúng phù hợp với hướng nghiên cứu mới.
- `luu_tru_nghien_cuu_truoc/`: tài liệu và artifact lịch sử có thể được lưu tại workspace để truy vết; không mặc định toàn bộ dữ liệu có trong Git.
- `third_party/`: mã và checkpoint tham khảo được tải riêng tại workspace, không đưa vào Git.

Khi dùng lại kết quả hoặc mã cũ, cần kiểm tra nguồn, protocol, split và khả năng tái lập trước khi đưa vào nghiên cứu mới. Đường dẫn bên trong một số mã cũ vẫn trỏ tới `outputs/` và có thể cần khôi phục artifact hoặc điều chỉnh trước khi chạy lại.
