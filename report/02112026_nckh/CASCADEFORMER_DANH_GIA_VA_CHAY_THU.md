# CascadeFormer: đối chiếu bài báo và chạy thử

Ngày 02/11/2026. Nguồn: [bản arXiv v1](https://arxiv.org/html/2509.00692v1), [mã tác giả](https://github.com/Yusen-Peng/CascadeFormer), [checkpoint Penn Action 1.0](https://huggingface.co/YusenPeng/CascadeFormerCheckpoints/tree/main/CascadeFormer_Best_Checkpoints/CascadeFormer_1.0_best_checkpoints/BEST_PENN_ACTION_1.0). Repo cho biết bài đã có trong ICPR 2026. Bản v1 được khảo sát là *CascadeFormer: A Family of Two-stage Cascading Transformers for Skeleton-based Human Action Recognition*.

## Các điểm cần điều chỉnh trong bảng tóm tắt

- Số 81,5% của ST-GCN trong bài là **NTU RGB+D 60 cross-subject**, không phải Penn Action. Bảng Penn của bài ghi AOG 85,5%, HDM-BG 93,4%, CascadeFormer 1.0 94,66%. Không khẳng định các con số tương đương về input, split hay tiền xử lý khi chưa kiểm tra bài gốc của từng baseline.
- Với Penn, paper ghi loại skeleton bị che bằng visibility nhưng không làm rõ hết quy tắc. Mã tải ngày 02/11/2026, commit `847efac9bec015d2d1d8f2104d255a0258ad76d7`, trong `penn_utils.py` đặt **tọa độ của từng khớp có visibility = 0 thành (0,0)**; không xóa toàn frame. Cờ GT không khả dụng trong ứng dụng thực nếu không có annotation; không tự dùng GT visibility cho pose dự đoán khi đo pipeline triển khai.
- 94,66% là số tác giả báo trên pose annotation và cách xử lý của họ; không phải accuracy với pose YOLO. Paper báo random-joint masking 94,66%, full reconstruction 91,10%; tác giả nêu overfitting có thể là lý do, không chứng minh nguyên nhân.
- Paper nói chia chính thức train/test; mã hiện tại còn tách 5% official train làm validation với `train_test_split` stratified, seed 42. Chưa xác thực mã hiện tại trùng chính xác phiên bản tạo checkpoint. Các ablation trình bày điểm Penn; chưa xác định được đầy đủ quy trình chọn cấu hình/checkpoint chỉ bằng cách đọc paper, vì vậy không khẳng định test độc lập.
- Decoder tái tạo được dùng lúc pretrain và bỏ khi fine-tune/inference theo paper; đầu ra nhận diện không tự cung cấp skeleton đã hiệu chỉnh. Kết quả tốt khi masked pretraining không chứng minh robustness với che khuất thực tế hay tác dụng làm mượt.

## Tình trạng chạy thử

Đã clone mã vào [CascadeFormer](../../third_party/CascadeFormer) và tải bốn checkpoint fine-tuned của Penn vào thư mục `local_run/checkpoints/` bên trong đó. Thư mục `third_party/` chỉ có tại workspace và không được đưa vào Git; trên bản clone mới cần lấy mã và checkpoint từ các nguồn tác giả ở trên. Đã dùng PyTorch 2.7.1+cu128 trên RTX 3050 Laptop 6 GB; workspace có 2.326 tệp annotation Penn. Không cài thêm gói vào môi trường chung. Môi trường hiện tại thiếu `sklearn`, là dependency của script gốc.

Đã chạy [cascadeformer_smoke_test.py](../../third_party/CascadeFormer/local_run/cascadeformer_smoke_test.py): trích các class mô hình trực tiếp từ mã tác giả bằng AST (để tránh import dependency chỉ phục vụ training), nạp bốn state dict với `weights_only=True` và `strict=True`, dùng tiền xử lý tọa độ như mã. Một clip official train `0002.mat`, 80 frame × 26 tọa độ, chạy trên GPU và dự đoán `baseball_pitch`, trùng nhãn; logits hữu hạn. Chi tiết phiên bản, tên lớp, SHA-256 của checkpoint ở [kết quả JSON](../../third_party/CascadeFormer/local_run/cascadeformer_smoke_test_result.json). Đây là **kiểm tra kỹ thuật một mẫu train có thể đã được tác giả dùng huấn luyện**, không phải đánh giá accuracy hay tái lập 94,66%.

Mã inference của tác giả đệm theo batch nhưng mean pooling qua toàn bộ chiều thời gian, không thấy truyền padding mask trong đường suy luận được đọc. Vì vậy, cần kiểm tra độ nhạy kết quả theo thành phần batch và so sánh batch size 1; không sửa trước rồi gọi kết quả là tái lập nguyên bản. Thứ tự nhãn được suy ra bằng thứ tự xuất hiện khi duyệt tên tệp annotation đã sắp xếp; cần xác thực với mapping train checkpoint trước khi công bố metric.

## Cách khai thác cho hướng hiệu chỉnh

1. Dùng CascadeFormer là một classifier bổ sung bên cạnh classifier hiện có, cùng tập clip và cùng nguồn pose; kiểm tra ảnh hưởng raw/lọc/nội suy trên cùng video.
2. Kiểm tra mapping 13 khớp, convention trái/phải, định dạng tọa độ pixel, chọn người, confidence/missing và padding trước khi đưa pose YOLO vào classifier GT-trained. Không dùng visibility GT trong suy luận dự kiến chỉ có video.
3. Có thể nghiên cứu masked pretraining như baseline tăng độ bền, nhưng đó là huấn luyện representation chứ không mặc định là hiệu chỉnh tọa độ. Nếu muốn dùng decoder để impute, cần kiểm tra riêng đầu ra và huấn luyện/đánh giá lại.
4. Trước khi rút kết luận, kiểm tra tập tác giả đã dùng cho checkpoint: validation nội bộ dự án có thể đã nằm trong dữ liệu huấn luyện của tác giả. Huấn luyện lại bằng train dự án nếu cần validation độc lập; dùng các split cố định, nhiều seed và ghi nhận lịch sử test.
