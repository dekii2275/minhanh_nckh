# Protocol thống nhất: hiệu chỉnh skeleton và nhận diện hành động trên Penn Action

**Phiên bản:** 1.0, ngày 15/11/2026. **Trạng thái:** thiết kế cho phân tích hồi cứu/thăm dò; chưa triển khai toàn bộ ma trận test. Các phương án, ngưỡng, validation và một phần kết quả test đã được xem trước khi viết protocol này. Không gọi đây là đăng ký trước, kiểm định độc lập hoặc xác nhận mới.

## 1. Câu hỏi và phạm vi

Trên cùng chuỗi pose YOLO dự đoán, hiệu chỉnh theo thời gian thay đổi chất lượng pose, độ sai lệch chuyển động và nhận diện hành động thế nào so với pose gốc? Xét riêng khi (a) giữ classifier train trên pose gốc và (b) train classifier riêng trên mỗi loại pose.

Đơn vị nhận diện và bootstrap là **video**, không phải frame hoặc dự đoán của từng seed. Kết quả cho phép mô tả tác động trong điều kiện Penn, bbox GT hỗ trợ, YOLO11n-Pose và xử lý offline; không tự chứng minh hiệu quả trên hệ chọn người tự động, online hoặc real-time. Kết quả âm vẫn báo đầy đủ.

## 2. Split, nguồn và đầu vào chung

- Penn Action, 15 lớp; manifest `report/02112026_nckh/penn_audit_artifacts/video_manifest.csv`, SHA-256 `d54f5b402278a04debb0e7ccf569088afeb4465c5084bddf5cb1ca4e668ddc41`. Split seed0 cố định: 1.005 train, 253 validation, 1.068 official test; kiểm tra ID, nhãn, không giao nhau và hash annotation trước chạy.
- Train dùng để fit classifier; validation chọn checkpoint theo quy tắc ở mục 4. Official test dùng báo cáo mô tả hồi cứu. Test và validation đã từng ảnh hưởng phát triển, không được coi là mẫu độc lập. Không điều chỉnh phương án, ngưỡng hoặc checkpoint theo test; nếu có thay đổi sau khi xem test, tạo phiên bản protocol mới, ghi thời điểm và lý do, tiếp tục ghi tính thăm dò.
- Cùng cache YOLO11n-Pose, provenance train/test trùng nhau; bbox GT mở rộng 25% mỗi phía dùng crop, chọn detection IoU GT lớn nhất với IoU ≥0,3. Kiểm tra và lưu hash từng cache, nguồn model và annotation. Cùng 12 khớp, fixed left/right swap đúng một lần như mã hiện có. Không thay pose thiếu bằng tọa độ GT; giữ toàn bộ video kể cả video không phát hiện được người.
- Áp dụng mỗi phép xử lý lên cùng pose gốc theo thứ tự thời gian frame gốc; sau đó cùng chuẩn hóa và nội suy thời gian 64 bước, 24 tọa độ như `compare_person_selection.resample` và `data.action_dataset.normalize_skeleton_sequence`. Giá trị còn thiếu theo xử lý zero hiện hành; lưu chính sách này vì có thể ảnh hưởng kết quả. Phép sửa không dùng nhãn hành động, GT keypoint hay visibility; bbox GT được dùng riêng cho crop/chọn người và đánh giá.
- Pose annotation: cùng 12 khớp, visibility và chuẩn hóa theo `train_gt_baselines.load_data`, cho train/validation và áp dụng cùng quy tắc sang test; train classifier riêng. Đây là nhánh tham khảo nguồn đầu vào khác, không gọi là trần hay hiệu quả hiệu chỉnh và không chạy qua bộ sửa YOLO.

## 3. Phương án cố định

| Mã | Phương án | Quy tắc cố định |
|---|---|---|
| R | Pose gốc | Fixed swap; không hiệu chỉnh thời gian. |
| I | Nội suy khoảng thiếu | Khoảng thiếu hoặc confidence <0,30 dài ≤3 frame, hai đầu đáng tin cậy; không ngoại suy. Dùng đúng `evaluate_pose_corrections.py`. |
| M | Lọc trung bình | Cửa sổ ±2 frame; ít nhất 3 quan sát confidence ≥0,30; chỉ sửa target tin cậy. |
| D | Lọc trung vị | Cùng cửa sổ, mask và điều kiện như M. |
| J | Nội suy khớp nhảy | Với hai điểm gốc liên tiếp hợp lệ có confidence ≥0,5, đánh dấu điểm sau nếu độ dời >1,5 scale vai–hông median clip. Nội suy từng đoạn 1–3 điểm đánh dấu nếu có hai đầu gốc không đánh dấu, confidence ≥0,5 và khoảng cách hai đầu ≤2 scale. Không đủ điều kiện giữ nguyên; không lặp, không thay confidence, không điền riêng điểm thiếu. Dùng `evaluate_abrupt_joint_interpolation_20261113.correct`. |
| C | Thiếu + điểm nhảy ba frame | Phương án bổ sung, dùng `targeted_pose_pilot_20261113.correct`: xử lý khoảng thiếu/confidence thấp ≤3 và điểm giữa lệch >0,75 scale so với trung điểm hai đầu; quy tắc đồng thời từ pose gốc và điều kiện anchor như protocol gốc. C khác J, không gộp tên hoặc kết luận hai phương pháp. |
| G | Pose annotation | Nhánh tham khảo, classifier train/test trên annotation. Không tham gia so sánh hiệu chỉnh cùng nguồn YOLO. |

M/I/D dùng triển khai `src/gt_baselines/evaluate_pose_corrections.py` và protocol `outputs/gt_crop_corrections_20261103/protocol.json`; ưu tiên mã và hash lịch sử nếu khác biệt chi tiết. Ghi hash phiên bản mã thực tế, tham số, mapping và nguồn pose trước khi chạy. Các phương án dùng frame tương lai/clip nên offline. Không dùng nhánh torso/confidence làm một phép sửa tọa độ; nếu phân tích sau phải tách riêng và có đối chứng cùng số chiều.

## 4. Classifier và hai chế độ

BiLSTM + attention trong `src/models/action_bilstm.py`, input 64 × 24, cùng thứ tự lớp, seed 0/1/2. Mỗi nhánh train 50 epoch, batch 32, Adam lr 0,001, weight decay 0,0001; cosine T_max 50, eta_min 0,00001; gradient clip norm 1; cùng quy tắc shuffle và generator theo seed trong `train_gt_baselines.py`. Chọn epoch có accuracy validation cao nhất, hòa chọn epoch sớm nhất. Không chọn epoch bằng test. Training và inference mới dùng CUDA; ghi GPU, driver, torch/CUDA và kết quả seed.

1. **Frozen:** mỗi seed dùng đúng checkpoint R train trên pose gốc cho R/I/M/D/J/C; đầu vào test của từng nhánh tương ứng được biến đổi, không cập nhật trọng số. Lưu và kiểm tra hash checkpoint để xác nhận cùng mô hình.
2. **Retrained:** mỗi nhánh R/I/M/D/J/C có mô hình riêng train/validation trên chính biểu diễn đó. So với R trong cùng chế độ, seed, split và ngân sách. G cũng được train riêng theo cùng lịch, nhưng báo riêng vì khác nguồn pose.

Có thể dùng lại checkpoint 1.005 train hiện có **chỉ sau** khi kiểm tra hash, input feature, ID, mã, cấu hình train và quy tắc chọn checkpoint; tái hiện dự đoán validation đã lưu, ghi mọi sai khác backend. Nếu không khớp, tạo một lượt train mới cùng protocol và thư mục output mới. Frozen chọn một bộ R rõ ràng và dùng nhất quán; xác nhận checkpoint R frozen và retrained trùng hash nếu muốn cùng baseline trong hai chế độ. Không trộn với checkpoint train đủ 1.258 video/epoch 50; nếu báo lượt đó, đặt bảng riêng.

## 5. Chỉ số và phép so sánh

**Nhận diện chính:** accuracy top-1 ở cấp video; báo macro-F1, accuracy từng seed, trung bình và SD mẫu giữa seed. So sánh J−R cho frozen và retrained là hai đối chiếu trọng tâm mô tả; I/M/D/C−R là đối chứng/so sánh bổ sung. Không chọn phương án thắng để báo riêng; báo đầy đủ tất cả. GT G báo riêng. Tính số video đúng→sai và sai→đúng theo từng seed.

**Pose:** PCK@0,2, ngưỡng 0,2 đường chéo bbox GT theo frame; mẫu là các khớp-frame có GT visible, tọa độ hợp lệ và bbox GT hợp lệ. Pose thiếu tính sai trong PCK chính. Báo thêm coverage và sai số tọa độ trên tập khớp cùng quan sát hợp lệ giữa các phương án so sánh; nêu mẫu số và cách xử lý không detection. Đếm số khớp sửa gần GT hơn/xa GT hơn và sai→đúng/đúng→sai, tách điểm vốn thiếu; không bỏ qua điểm hỏng.

**Chuyển động:** tính sai lệch vector độ dời 2D giữa hai frame liên tiếp so với annotation, chia đường chéo bbox GT ở frame sau; dùng cùng tập cặp khớp-frame có GT và pose gốc hợp lệ ở cả hai frame và đủ điều kiện ở mọi phương án được so sánh. Báo mẫu số, trung bình gộp và theo video. Đây là sai lệch độ dời chuẩn hóa, không phải vận tốc vật lý. Góc, phân nhóm nhanh/chậm và các hình minh họa nếu thực hiện là phân tích bổ sung, định nghĩa mask và quy tắc trước khi tính; không gọi giảm jitter là tăng fidelity.

**Bất định:** với từng chênh accuracy, trung bình đúng/sai trên 3 seed theo từng video rồi bootstrap ghép cặp 10.000 mẫu video, seed bootstrap 20261115, lấy percentile 2,5/97,5%. Với PCK hoặc sai lệch chuyển động, bootstrap video, tính lại tổng tử/mẫu số ở từng mẫu; báo phân bố của chênh lệch và định nghĩa trọng số. Kết quả có điều kiện theo mô hình đã train; không bao gồm bất định do chọn phương án/checkpoint, lịch sử test hay đầy đủ biến thiên seed. Nhiều phương án/so sánh đã khảo sát: không diễn giải các khoảng riêng lẻ như bằng chứng khẳng định đã hiệu chỉnh đa kiểm định; khoảng chứa 0 không chứng minh tương đương.

## 6. Thứ tự triển khai và kiểm tra

1. Kiểm tra GPU/CUDA, không gian đĩa, dữ liệu, split, hash, checkpoint và cache; tạo thư mục output mới, không ghi đè. Ghi thời điểm và các kết quả đã biết trước chạy.
2. Audit tái sử dụng checkpoint theo mục 4; xác minh nguồn cache train/test và features R; kiểm tra correspondence, các video thiếu detection, shape/nhãn/ID; so sánh pred validation với artifact đã lưu.
3. Tạo features cho toàn bộ nhánh trên cùng test; kiểm thử bảo đảm R khớp bitwise cache cũ, mỗi phương án chỉ sửa điều kiện cho phép, không tạo nhãn/GT vào đầu vào; lưu hash và thống kê thay đổi từng video.
4. Chạy GPU frozen và retrained cho tất cả seed; ghi xác suất hoặc logits, dự đoán, nhãn, video, checkpoint/hash và trạng thái. Đối chiếu những ô test đã có; sai khác backend phải được điều tra và báo, không chỉnh kết quả tùy ý.
5. Tính pose, chuyển động, accuracy/macro-F1 và bootstrap theo mục 5, báo cả failure và mẫu số; tạo bảng và báo cáo. Ghi rõ test đã bị lộ trong lịch sử và kết quả mới/cũ, chế độ frozen/retrained, protocol và provenance từng dòng.
6. Một đánh giá trên J-HMDB hay tập khác cần audit data, mapping, split và viết protocol riêng trước; chưa coi là transfer hay kiểm định độc lập cho đến khi xác minh lịch sử sử dụng và thiết kế.

## 7. Đầu ra bắt buộc

Protocol bản sao và hash; manifest ID/split/nhãn và hash cache/annotation; phiên bản code, model, dependency và phần cứng; feature hoặc hash; lịch sử train và checkpoint/hash; dự đoán từng video/seed/chế độ/phương án; thống kê thay đổi pose, mặt nạ và mẫu số; summary/CI và báo cáo có các kết quả âm, giới hạn, phát sinh và thay đổi protocol. Nếu một bước không hoàn thành, ghi rõ trạng thái thay vì suy ra kết quả.

**Tài liệu liên quan:** [thí nghiệm bộ lọc](../03112026_nckh/HIEU_CHINH_YOLO_CROP_BILSTM_2026-11-03.md), [nội suy khớp nhảy](NOI_SUY_RIENG_KHOP_NHAY_DOT_NGOT_2026-11-13.md), [đánh giá test gần nhất](DANH_GIA_LAI_TEST_GPU_2026-11-15.md).
