# Khởi đầu nghiên cứu nhận diện hành động từ skeleton

Ngày: 02/11/2026. Trạng thái: khảo sát ban đầu; câu hỏi và protocol chưa chốt.

## Mục tiêu

Nghiên cứu nhận diện nhãn hành động của một video từ chuỗi tọa độ khớp cơ thể 2D. Đề xuất trước mắt là tìm hiểu tác động của sai lệch pose đến độ chính xác nhận diện và so sánh các cách xử lý trong điều kiện kiểm soát. Đây là hướng khởi đầu để thảo luận, chưa phải cam kết rằng phương pháp mới hay hiệu quả hơn.

## Bối cảnh và tài liệu gần

- [ST-GCN, AAAI 2018](https://aaai.org/papers/12328-spatial-temporal-graph-convolutional-networks-for-skeleton-based-action-recognition/): mô hình hóa skeleton bằng đồ thị không gian–thời gian; là baseline kiến trúc quan trọng, đánh giá trên Kinetics và NTU-RGB+D, không thể lấy số đó so trực tiếp với Penn Action.
- [Revisiting Skeleton-Based Action Recognition, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.pdf): so sánh biểu diễn heatmap với GCN và kiểm tra nhiễu bằng cách bỏ keypoint trên FineGYM; tác giả cũng xét huấn luyện với nhiễu. Câu hỏi độ bền trước thiếu khớp vì vậy đã được nghiên cứu.
- [Occluded Skeleton-Based Human Action Recognition with Dual Inhibition Training, ACM MM 2023](https://doi.org/10.1145/3581783.3612170): nghiên cứu huấn luyện tăng độ bền khi che khuất và có nhiễu. Không thể tuyên bố ý tưởng mô phỏng thiếu khớp tự nó là mới.
- [Enhancing action recognition from low-quality skeleton data via part-level knowledge distillation](https://www.sciencedirect.com/science/article/pii/S0165168424001051): nghiên cứu skeleton chất lượng thấp và distillation, có dùng Penn Action cùng các bộ dữ liệu khác. Cần đọc kỹ split, nguồn pose, và thiết lập trước khi so sánh với dự án.
- [Dual-Domain Triple Contrast for Cross-Dataset Skeleton-Based Action Recognition, 2025](https://doi.org/10.1145/3715917): xem chuyển dịch phân phối giữa các bộ dữ liệu như bài toán domain adaptation. Đánh giá trong một bộ dữ liệu không đủ chứng minh khả năng transfer giữa các bộ.

Các nguồn trên chỉ là khảo sát định hướng, chưa phải tổng quan hệ thống hay kiểm tra đầy đủ tính mới. Số liệu giữa bài báo không được so sánh trực tiếp nếu split, nguồn pose, lớp hành động, mô hình pretrained hoặc metric khác nhau.

## Câu hỏi đề xuất

**Trong nhận diện hành động cấp video trên Penn Action, khi pose đầu vào được ước lượng từ video, các lựa chọn xử lý pose và huấn luyện ảnh hưởng thế nào đến độ chính xác so với các baseline dùng cùng nguồn pose và cùng split?**

Giả thuyết làm việc: huấn luyện và kiểm thử với phân phối pose phù hợp có thể giảm ảnh hưởng của sai lệch pose; cải thiện hình học pose không nhất thiết cải thiện accuracy. Đây là giả thuyết cần kiểm định, không phải kết luận.

Các câu hỏi hẹp hơn để chọn sau khi rà soát: (1) sai khác giữa pose annotation và pose dự đoán ảnh hưởng thế nào khi giữ nguyên classifier? (2) huấn luyện trên pose dự đoán so với annotation ảnh hưởng thế nào khi cùng đánh giá trên pose dự đoán? (3) một phép xử lý cụ thể có tạo lợi ích bổ sung so với hai baseline đó không? Lựa chọn phép xử lý phải dựa vào audit và literature, không mặc định tiếp tục calibration cũ.

## Thiết kế dự kiến

- Đơn vị phân tích: video; nhãn hành động cấp video. Kiểm tra nhãn, nguồn pose, số mẫu và split trước khi chốt.
- Đối chứng tối thiểu: huấn luyện trên annotation và đánh giá trên pose dự đoán; huấn luyện và đánh giá trên pose dự đoán; phép xử lý được đề xuất nếu có. Dùng cùng clip, classifier, ngân sách, mapping và điều kiện đánh giá để so sánh; ghi rõ khi không thể giữ giống nhau. Với mô hình mỗi điều kiện được huấn luyện riêng, dùng nhiều seed và báo cả biến thiên theo seed.
- Chọn kiến trúc, siêu tham số và phép xử lý bằng train/validation; không dùng test cho lựa chọn. Đánh giá chênh lệch accuracy theo cặp trên cùng video, theo điểm phần trăm và độ bất định thích hợp; phân tích theo lớp và chi phí nếu khả thi.
- Audit correspondence trái/phải, chọn người, thiếu khớp, confidence, nguồn checkpoint và phân bố pose trước khi đo; không dùng GT để chọn người khi suy luận nếu điều kiện triển khai không có GT. Nhiễu tổng hợp, nếu dùng, cần tách khỏi lỗi pose thực và không được mặc định là đại diện cho lỗi thực.
- Báo cáo nguồn dữ liệu và split, điều kiện trích pose, seed, cấu hình, checkpoint/hash, dự đoán và các thử nghiệm âm. Nếu test cũ đã ảnh hưởng tới thiết kế, không xem lại cùng test như kiểm định hoàn toàn độc lập.

## Những gì đã biết trong workspace

README xác nhận giai đoạn mới chưa chốt đề tài; các báo cáo trước là lịch sử. Hiện có thư mục dữ liệu Penn Action và J-HMDB, checkpoint `models/yolo11n-pose.pt` và mã nghiên cứu cũ; sự tồn tại của chúng chưa chứng minh dữ liệu đầy đủ hay pipeline tái lập được. Audit lịch sử về correspondence trên validation cho thấy cần xem xét convention trái/phải, nhưng không xác lập nguyên nhân và không thay thế kiểm định cho pipeline mới. Chưa chạy thí nghiệm mới trong giai đoạn này.

## Việc tiếp theo trước khi chạy

1. Kiểm kê tính đầy đủ dữ liệu, split, annotation, checkpoint, môi trường và tài nguyên tính toán; xác định test nào đã tham gia quyết định ở nghiên cứu cũ.
2. Đọc chi tiết các bài gần nhất, đặc biệt thiết lập Penn Action và phương pháp xử lý lỗi pose; lập bảng đối chiếu và thu hẹp câu hỏi.
3. Chốt protocol validation/test và baseline; kiểm tra pipeline trên mẫu nhỏ, sau đó mới ước lượng chi phí và chạy thí nghiệm chính.
