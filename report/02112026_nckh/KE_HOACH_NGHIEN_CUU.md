# Kế hoạch nghiên cứu dự kiến

**Mục tiêu:** Đánh giá hiệu chỉnh skeleton 2D theo thời gian ảnh hưởng thế nào đến độ chính xác pose, khả năng giữ chuyển động và nhận diện hành động.

**Dữ liệu:** Penn Action trước, sau đó J-HMDB; xác minh dữ liệu, split, khớp trái/phải và cách chọn người trước khi chạy.

**Đối chứng:** Cùng video và pose dự đoán, so sánh raw, nội suy khớp thiếu, lọc thời gian cố định và phương pháp thích ứng nếu phù hợp. Thêm pose annotation làm tham chiếu. So sánh classifier giữ nguyên và classifier được huấn luyện trên từng loại pose với cùng split, ngân sách và nhiều seed.

**Đánh giá:** PCK/sai số pose, sai lệch góc và vận tốc, Top-1 và macro-F1; báo chênh lệch theo video, độ bất định, biến thiên seed, chi phí và các trường hợp hiệu chỉnh làm kết quả xấu đi. Không coi giảm jitter là bằng chứng pose chính xác hơn.

**Quy trình:** Chốt và chọn cấu hình trên train/validation; khóa protocol trước test. Ghi nguồn dữ liệu, checkpoint, split và seed; nêu rõ test cũ đã ảnh hưởng đến thiết kế khi diễn giải. Hai bộ được đánh giá riêng, không mặc định là transfer giữa bộ.

**Thứ tự:** (1) Audit dữ liệu và pipeline; (2) chạy pilot baseline trên validation; (3) thiết kế và đánh giá hiệu chỉnh; (4) mở rộng sang J-HMDB và tổng hợp kết quả. Chưa khẳng định tính mới hoặc hiệu quả trước thí nghiệm và khảo sát bổ sung.
