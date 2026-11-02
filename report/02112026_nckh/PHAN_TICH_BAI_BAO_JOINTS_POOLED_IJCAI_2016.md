# Phân tích Joints-Pooled 3D Deep Convolutional Descriptors

Ngày khảo sát 02/11/2026. Bài gốc: [Action Recognition with Joints-Pooled 3D Deep Convolutional Descriptors](https://www.ijcai.org/Proceedings/16/Papers/470.pdf), Congqi Cao và cộng sự, IJCAI 2016. Nghiên cứu kết hợp **video RGB và vị trí khớp**, không phải classifier chỉ nhận skeleton, và không đưa ra bộ hiệu chỉnh pose.

## Phương pháp và thiết lập

C3D tạo feature map không gian–thời gian từ clip video; vị trí khớp được ánh xạ lên feature map thông qua kích thước kernel, stride và padding. Sau pooling tại các vị trí khớp, các descriptor được tổng hợp thành đặc trưng video; trên Penn Action và sub-JHMDB tác giả dùng linear SVM. So sánh ratio scaling với coordinate mapping. Trên Penn và Composable Activities, clip dài 16 frame, chồng 8 frame; với sub-JHMDB, lấy ba clip 16 frame ở đầu, giữa và cuối. Không thể áp dụng quy tắc trượt Penn một cách mặc định cho sub-JHMDB.

Penn có 15 lớp, split official train/test. sub-JHMDB là tập con 316 video, 12 lớp và bài nói dùng ba split/3-fold. Bài cũng báo một thí nghiệm bổ sung trên JHMDB đầy đủ; chưa xác minh được đầy đủ các chi tiết protocol của kết quả bổ sung đó từ đoạn mô tả ngắn, không mặc định áp cùng quy trình. Composable Activities dùng leave-one-subject-out trên 14 người và fine-tune C3D; không dùng hai bộ dữ liệu đó như cùng một protocol. Trên Penn và sub-JHMDB, C3D không được fine-tune.

## Đối chiếu các con số

- **Penn Action:** bảng 4 báo C3D fc6 86,0%; JDD conv5b với khớp ước lượng 87,4%; JDD conv5b 94,3%; JDD fusion conv5b + conv4b 95,7%. Phần văn bản báo dùng temporal max+min thay trung bình đạt 97,1% với conv5b và 98,1% khi kết hợp conv5b/conv4b. Cần làm rõ điều kiện pose của con số 98,1% trước khi dùng như điểm khớp ước lượng; không suy ra chỉ từ con số đó rằng hiệu chỉnh hay dự đoán pose kém vẫn không ảnh hưởng.
- **sub-JHMDB:** bảng 2, cùng JDD conv5b, khớp GT 81,9%, khớp ước lượng 77,7%, giảm 4,2 điểm phần trăm; đây là đối chiếu trực tiếp GT/predicted mà tác giả báo. Bảng 3 báo fusion conv5b và tọa độ khớp 81,9%; fusion conv5b + conv4b 83,3%. Hai kết quả 81,9% thuộc hai cấu hình bảng khác nhau và không đồng nghĩa JDD luôn gồm tọa độ làm đầu vào bổ sung.
- **JHMDB đầy đủ:** tác giả ghi kết quả tốt nhất 72,63% và ánh xạ khớp ngoài ảnh vào biên gần nhất. Bài mô tả khoảng hai phần ba video có khớp annotation ngoài khung hình; đây không phải tỷ lệ lỗi nhận diện hoặc tỷ lệ video bị che khuất theo một định nghĩa khác.

Sự giảm 81,9 → 77,7 là bằng chứng trong thiết lập cụ thể của bài, không bảo đảm kết quả tương tự khi thay pose estimator, dataset, preprocessing hoặc classifier. RGB chứa thông tin bổ sung nên không so 98,1% hay các số trên trực tiếp với model skeleton-only.

## Ý nghĩa cho hướng hiệu chỉnh

Nghiên cứu cho thấy sự khác biệt giữa pose annotation và pose ước lượng có thể ảnh hưởng classifier, ngay cả khi có đặc trưng RGB. Gợi ý một đối chứng mở rộng: trên cùng video và cùng classifier, so các đầu vào khớp raw, hiệu chỉnh và GT; báo chênh lệch cặp theo video cùng chất lượng pose. Cần kiểm soát phép ánh xạ tọa độ sau crop/resize, khớp thiếu/ngoài frame, nguồn pose, lựa chọn người và nguồn pose khi train/test. So sánh này đo tác động trong hệ RGB+pose và không tự chứng minh lợi ích với skeleton-only. Ưu tiên skeleton-only nếu giữ nguyên phạm vi câu hỏi hiện tại.

## Code và khả năng chạy

Tìm theo tên bài và tác giả chưa xác nhận được repository/weights JDD chính thức; không khẳng định chúng không tồn tại. Có [một bản C3D PyTorch do bên thứ ba chuyển đổi](https://github.com/DavideA/c3d-pytorch) và checkpoint Sports-1M được liên kết ở repo đó, nhưng đây là backbone tham khảo, không phải code JDD chính thức và repository lưu ý chỉ kiểm thử định tính. Chưa tải hay chạy mã. Để tái lập cần kiểm tra nguồn checkpoint và tiền xử lý RGB, mapping receptive field, pooling, SVM, split và loại pose; cài lại độc lập không tự chứng minh kết quả tái lập.
