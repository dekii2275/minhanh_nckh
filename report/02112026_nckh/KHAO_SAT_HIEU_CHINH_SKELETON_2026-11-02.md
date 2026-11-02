# Khảo sát hai hướng hiệu chỉnh skeleton

Ngày khảo sát: 02/11/2026. Phạm vi: tìm kiếm định hướng các nguồn gốc về hiệu chỉnh pose video và nhận diện hành động liên quan Penn Action, J-HMDB/Sub-JHMDB; chưa phải tổng quan hệ thống hay xác nhận tính mới. Các liên kết mã đã được kiểm tra ở mức trang repository, chưa cài đặt hoặc chạy lại.

## Dữ liệu và giới hạn

Workspace có `data/raw/Penn_Action/{frames,labels}` và `data/raw/jhmdb/{ReCompress_Videos,joint_positions,splits}`, cùng checkpoint `models/yolo11n-pose.pt`. Chưa kiểm tra tính đầy đủ, tính toàn vẹn hoặc khả năng tái lập của các bộ này. Penn Action có split train/test chính thức; J-HMDB và Sub-JHMDB không đồng nhất: Sub-JHMDB là tập con 316 video, 12 lớp; nhiều bài pose video báo trung bình ba split. Kiểm tra cách dựng tập con, annotation và split trước khi so sánh. [Nguồn mô tả Penn và Sub-JHMDB](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Motion_Adaptive_Pose_Estimation_From_Compressed_Videos_ICCV_2021_paper.pdf).

Kết quả dự án trước đã dùng test trong quá trình phát triển; không mặc định kết quả trên cùng test là kiểm định hoàn toàn độc lập. Báo cáo lịch sử J-HMDB split 1 ghi giảm jitter và tăng nhẹ PCK nhưng accuracy calibrated thấp hơn raw; đây là động lực kiểm tra hai mục tiêu riêng, không phải kết quả mới. Xem [protocol lịch sử](../../luu_tru_nghien_cuu_truoc/reports/EXPERIMENT_PROTOCOL.md) và [báo cáo J-HMDB](../../luu_tru_nghien_cuu_truoc/reports/JHMDB_EVALUATION_REPORT.md).

## Hướng 1 — Hiệu chỉnh theo thời gian có xét độ tin cậy và cấu trúc

**Câu hỏi:** Khi pose 2D được ước lượng theo từng frame, hiệu chỉnh có điều kiện theo độ tin cậy và chuyển động có cải thiện cả độ chính xác pose lẫn nhận diện hành động so với không hiệu chỉnh và các bộ lọc đơn giản không?

**Ý tưởng có thể thử:** đánh dấu khớp thiếu/độ tin cậy thấp, phát hiện bước nhảy bất thường sau khi theo dõi cùng một người, sau đó nội suy hoặc lọc thích ứng theo từng khớp. Có thể bổ sung ràng buộc mềm theo các đoạn xương, với mức can thiệp lựa chọn trên validation; cần tránh áp đặt chiều dài 2D cố định vì góc nhìn và che khuất thay đổi. Dùng frame tương lai thì ghi rõ xử lý offline.

**Đối chứng:** raw; nội suy thiếu khớp; median/One-Euro hoặc bộ lọc thời gian đơn giản; từng thành phần và kết hợp. So sánh cùng dự đoán, mapping, clip và classifier. Đánh giá cả classifier giữ nguyên và classifier huấn luyện lại trên pose theo từng điều kiện, gồm baseline train/test cùng phân phối. Không dùng GT để chọn người tại suy luận nếu điều kiện ứng dụng không có GT.

**Đo lường:** PCK chuẩn hóa theo bbox và sai số theo khớp/visibility nếu annotation cho phép; độ rung và sai số vận tốc chỉ là chỉ số bổ sung; accuracy/F1 theo video, thay đổi theo cặp và nhiều seed; thời gian xử lý. Phân tích ảnh hưởng lên chuyển động nhanh và clip bị che khuất. Pose mượt hơn không chứng minh pose đúng hơn, cũng không đảm bảo accuracy tăng.

**Công trình gần:** [RPSTN](https://arxiv.org/abs/2107.03591) khai thác quan hệ khớp và chuyển đặc trưng giữa frame cho Penn Action, Sub-JHMDB và PoseTrack; là mô hình ước lượng pose từ ảnh/heatmap chứ không phải bộ hậu xử lý tọa độ độc lập. [Mã RPSTN](https://github.com/YHDang/RPSTN) có hướng dẫn và liên kết checkpoint ngoài. [K-FPN, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620596.pdf) chọn keyframe và nội suy pose trên Penn Action/Sub-JHMDB; [mã chính thức](https://github.com/Yuexiaoxi10/Key-Frame-Proposal-Network-for-Efficient-Pose-Estimation-in-Videos). Cả hai cho thấy thời gian và nội suy đã được nghiên cứu; lựa chọn này không tự tạo tính mới.

**Khả thi:** cao cho pilot vì dùng chuỗi tọa độ sẵn có; rủi ro là làm mất chuyển động nhanh hoặc sửa sai người/convention. Tính mới tiềm năng chỉ có thể xác định sau đối chiếu sâu và chứng minh khác biệt cụ thể.

## Hướng 2 — Học hiệu chỉnh có giám sát, kết hợp mục tiêu pose và hành động

**Câu hỏi:** Một mô-đun học sửa sai tọa độ từ pose dự đoán, được giám sát bằng annotation và/hoặc tín hiệu nhận diện hành động, có tạo lợi ích bổ sung so với bộ lọc đơn giản và classifier được huấn luyện trực tiếp trên pose dự đoán không?

**Ý tưởng có thể thử:** từ chuỗi `(x, y, confidence, mask)` dự đoán một độ điều chỉnh theo từng khớp/frame; huấn luyện trên cặp pose dự đoán–annotation sau khi audit mapping và chọn đúng người. So sánh loss pose đơn thuần với loss pose kết hợp classification; giới hạn mức sửa và chỉ tính loss khớp có nhãn hợp lệ. Đây là đề xuất thiết kế, không phải phương pháp đã được xác nhận là mới. Nếu chỉ huấn luyện classification, tọa độ trung gian có thể là biểu diễn phục vụ classifier, không được tự động gọi là pose chính xác hơn.

**Đối chứng:** raw với classifier train trên raw; hướng 1; mô hình học sửa chỉ dùng loss pose; phiên bản kết hợp loss; nếu tài nguyên cho phép, thích nghi [PR-GCN](https://arxiv.org/abs/2010.07367) làm baseline liên quan. Tách lợi ích đến từ mô-đun sửa, số tham số và việc train lại. PR-GCN đã hiệu chỉnh trước nhận diện trên Kinetics-skeleton và NTU RGB+D, [mã chính thức](https://github.com/sj-li/PR-GCN); cần chuyển định dạng cho hai bộ dự án, không tuyên bố bài gốc dùng hai bộ này.

**Đo lường và rủi ro:** giống hướng 1, thêm chi phí train/suy luận và ablation; so sánh nhiều seed. Rủi ro overfit, loss nhận diện khiến khớp lệch GT, và mismatch annotation/detection. Phải tách theo video trước khi tạo cặp để tránh rò rỉ và không đưa annotation test vào inference. [Bài distillation trên skeleton chất lượng thấp](https://arxiv.org/abs/2404.18206) đã dùng Penn Action, trong đó có baseline train/test trên skeleton chất lượng thấp; nghiên cứu này là nhận diện, không phải bằng chứng module xuất tọa độ pose tốt hơn. Chưa xác nhận được mã chính thức của bài distillation trong lượt tìm kiếm này.

**Khả thi:** trung bình; cần cache dự đoán, cặp nhãn đáng tin cậy và huấn luyện nhiều mô hình. Đây là hướng sát mục tiêu nhận diện nhưng có prior work gần, cần đọc kỹ trước khi đề xuất đóng góp.

## Danh mục bài dùng dữ liệu dự án

| Bài | Dữ liệu và mục tiêu | Input / đánh giá cần phân biệt | Mã |
|---|---|---|---|
| [RPSTN](https://arxiv.org/abs/2107.03591) | Penn Action, Sub-JHMDB; pose video | Ảnh/đặc trưng pose, PCK; không trực tiếp kiểm tra accuracy hành động | [RPSTN](https://github.com/YHDang/RPSTN), checkpoint liên kết ngoài |
| [K-FPN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620596.pdf) | Penn Action, Sub-JHMDB; pose video | Keyframe, nội suy; PCK và chi phí; kiểm tra tiền xử lý bbox GT | [K-FPN](https://github.com/Yuexiaoxi10/Key-Frame-Proposal-Network-for-Efficient-Pose-Estimation-in-Videos) |
| [Motion Adaptive Pose, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Motion_Adaptive_Pose_Estimation_From_Compressed_Videos_ICCV_2021_paper.pdf) | Penn Action, Sub-JHMDB; pose video | Dùng video nén, motion vector/residual, bbox và PCK@0.2; không phải skeleton-only | Chưa xác nhận mã chính thức |
| [Multi-Modal Three-Stream Network](https://arxiv.org/abs/1909.03466) | Penn Action, J-HMDB và Sub-JHMDB; nhận diện | Có nhánh pose GT/dự đoán và nhánh RGB/optical flow; không so trực tiếp fusion với skeleton-only | Chưa xác nhận mã chính thức |
| [Action Recognition with Joints-Pooled 3D Deep Convolutional Descriptors, IJCAI 2016](https://www.ijcai.org/Proceedings/16/Papers/470.pdf) | Penn Action, Sub-JHMDB; nhận diện | RGB C3D pooling theo khớp, SVM; không phải skeleton-only | Chưa xác nhận mã chính thức |
| [Part-Level Knowledge Distillation](https://arxiv.org/abs/2404.18206) | Penn Action cùng NTU và SYSU; nhận diện skeleton chất lượng thấp | So sánh pose GT và pose từ PifPaf, có teacher–student; kiểm tra kỹ setup và split | Chưa xác nhận mã chính thức |

Mã tham khảo bổ sung: [PR-GCN](https://github.com/sj-li/PR-GCN) là baseline hiệu chỉnh–nhận diện, không được quảng cáo là chạy sẵn Penn/J-HMDB; [KGS-GCN](https://github.com/YuhanChen2024/KGS-GCN) cung cấp script đọc 13 khớp annotation Penn và official split, nhưng dùng như baseline phải xác minh paper, version và điều kiện huấn luyện. Chưa chạy thử bất kỳ repository nào.

## Đề xuất thứ tự

Ưu tiên hướng 1 để đo tác động từng thành phần với chi phí thấp, sau khi kiểm tra dữ liệu, split, correspondence và tracker. Hướng 2 phù hợp tiếp theo nếu hướng 1 cho thấy lỗi còn lại đủ đáng kể và dữ liệu ghép cặp đủ tin cậy. Khóa thiết kế và chọn cấu hình trên validation; báo test theo nguồn gốc đã sử dụng và không khẳng định độc lập nếu test từng ảnh hưởng thiết kế. Chưa có thí nghiệm mới trong khảo sát này.
