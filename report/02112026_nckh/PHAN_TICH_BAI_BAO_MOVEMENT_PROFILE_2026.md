# Phân tích bài báo hồ sơ chuyển động có thể giải thích

Ngày khảo sát 02/11/2026. [Ma và cộng sự, Interpretable skeleton-based movement profiles for formative feedback in school physical activity education](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1925220/full), Frontiers in Public Health, 09/09/2026, DOI 10.3389/fpubh.2026.1925220. “Conceptual framework…” là chú thích Hình 1, không phải tên bài báo.

## Nội dung và thiết kế

Nghiên cứu so sánh LSTM, GRU, Transformer, ST-GCN và một bản CTR-GCN gọn, rồi dùng tọa độ khớp tính bốn đặc trưng hồ sơ vận động: biên độ, ổn định, phối hợp hai bên và tính nhất quán thời gian. Các chỉ số là mô tả tương đối theo phân phối cùng lớp, không được kiểm định là điểm chất lượng chuyển động; đây là proof-of-concept tính toán, chưa kiểm thử kết quả học tập, sức khỏe hoặc triển khai lớp học.

Ba bộ dữ liệu: NTU RGB+D 60 chọn 9 lớp, Penn Action 13 khớp 2D chọn 14 lớp thể thao (loại `strum_guitar`), FineGym 99 lớp với pose HRNet từ PYSKL. Không dùng J-HMDB. Penn dùng split official ban đầu 1.212 train, 1.020 held-out sau lọc lớp; chia stratified 15% từ train, seed 2026 thành 1.030 train và 182 development; test giữ 1.020. Tác giả nói chọn checkpoint bằng dev macro-F1, sau đó đánh giá held-out. Năm seed huấn luyện 0–4. Đây là mô tả của tác giả; chưa có code để kiểm tra độc lập.

Skeleton được căn tâm hông cho dữ liệu 2D, scale theo thân và lấy lại mẫu 64 frame. Các mô hình được so sánh trên cùng split và ngân sách train. CTR-GCN gọn không tự nhận tái lập chính xác mọi chi tiết của bản gốc. Metric chính macro-F1; accuracy được báo bổ sung.

## Sửa bảng tóm tắt

1. **0,893 → 0,905 là macro-F1, không phải accuracy.** Với GRU trên Penn, accuracy tương ứng 0,899 → 0,911. So sánh macro-F1 full so với tọa độ đơn thuần có p hiệu chỉnh Holm = 0,079, không đạt ngưỡng 0,05; không thể kết luận có cải thiện chắc chắn hoặc không có hiệu ứng. Tác giả báo cáo full so với tọa độ + vận tốc có p = 0,861, không chứng minh tương đương.
2. **Full feature không phải bốn điểm tổng hợp profile**: ablation dùng tọa độ + góc khớp + vận tốc + vị trí và tốc độ tâm thân (COM proxy). Không được xem đó là bằng chứng bốn điểm profile cải thiện nhận diện.
3. Với Penn, kiểm định được nêu là **CTR-GCN so với từng mô hình còn lại**; sau hiệu chỉnh chỉ so sánh với LSTM còn ý nghĩa. Không thể suy ra GRU/ST-GCN/Transformer đều vượt LSTM hay bốn mô hình tương đương. Tác giả dùng paired t-test trên 5 seed và Holm; Wilcoxon hai phía có p nhỏ nhất 0,0625 với 5 cặp khác 0.
4. **Biên độ là trung bình range of motion** (max góc trừ min góc) của 8 khớp, không phải trung bình góc. Tính phối hợp theo tương quan góc; phép tìm lag −15…15 frame dùng cho hành động luân phiên, không mặc định cho mọi hành động. Nhất quán thời gian dùng sự đều đặn các khoảng giữa đỉnh tốc độ và phương án dự phòng theo định nghĩa bài, không đơn thuần là tốc độ. Cần đọc đầy đủ công thức và điều kiện biên trước khi cài đặt.
5. Điểm bách phân vị phản ánh vị trí so với mẫu cùng lớp trong tập tham chiếu, không phải nhãn đúng/sai hay điểm chuyên gia, cũng không phải chuẩn độ tuổi. Không xác lập hiệu quả AQA hoặc giáo dục.

## Liên hệ với hiệu chỉnh skeleton

Bài không kiểm định trực tiếp phương pháp hiệu chỉnh skeleton hay thí nghiệm có kiểm soát với góc nhìn, che khuất, lỗi ước lượng pose. Tuy nhiên, chính bài chỉ ra biên độ nhạy với cực trị nhiễu; ổn định và nhịp nhạy với dao động theo frame; phối hợp nhạy với đổi trái/phải. Điều này giúp thiết kế đánh giá trước/sau hiệu chỉnh: so sánh raw, nội suy và bộ lọc trên cùng video, đo PCK/sai số và accuracy, đồng thời đo sai lệch các chỉ số chuyển động so với annotation trên đúng khớp hợp lệ. Phân tích có cặp theo video, xét khớp thiếu, chuyển động nhanh, độ phân giải và frame rate; không coi giảm jitter hay thay đổi percentile là cải thiện chất lượng động tác.

Đối chiếu 2D chiếu hình: góc 2D thay đổi theo góc nhìn và che khuất; normalize hông/thân có thể không ổn định nếu keypoint lỗi. Các cặp đối xứng và quy tắc khoảng trễ phải được xác định cho từng hành động và dữ liệu; các công thức, khớp tương đương và cách xử lý thiếu chưa xác nhận đủ để tuyên bố tái lập chính xác.

## Mã và tính tái lập

Trang bài không chỉ tới repository chính thức; tìm kiếm theo tiêu đề, DOI và tên tác giả chưa xác nhận được mã hay checkpoint công khai. Tuyên bố data availability dẫn tới bài, supplement và liên hệ tác giả. Không khẳng định mã không tồn tại. Chưa tải mã, chưa chạy demo và chưa triển khai lại công thức. Nếu tự triển khai, cần công bố đây là implementation độc lập; kiểm tra seed/split, quy tắc tính 2D, frame rate, ngưỡng, khớp không hợp lệ và dữ liệu dùng xây reference distribution.
