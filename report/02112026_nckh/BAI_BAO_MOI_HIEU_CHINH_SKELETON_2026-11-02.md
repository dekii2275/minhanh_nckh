# Bài báo gần đây: hiệu chỉnh skeleton và nhận diện hành động

Khảo sát ngày 02/11/2026, ưu tiên 2025–2026, từ các trang tác giả, tạp chí, hội nghị và arXiv, với truy vấn về pose refinement, temporal smoothing, missing-joint imputation và action recognition. Có bổ sung baseline cũ quan trọng. Đây là tìm kiếm định hướng, không phải tổng quan hệ thống và không bảo đảm bao phủ toàn bộ công trình mới nhất. Liên kết mã chỉ được kiểm tra ở mức trang công khai; chưa cài hay chạy.

## Sát câu hỏi nghiên cứu

1. **[Recognition-Guided Generative Skeleton Imputation for Robust Action Recognition Under Severe Occlusion in Construction Scenarios](https://www.mdpi.com/2227-9709/13/9/140), 2026.** Dùng nhận diện để kích hoạt/chọn phục hồi skeleton sinh ra trong các trường hợp che khuất; đánh giá dữ liệu xây dựng thu thập và NW-UCLA, dùng bốn classifier và chín kiểu che khuất. Bài báo cáo trung bình cải thiện trên hai tập, nhưng với điều kiện mất đoạn liên tục theo thời gian trên dữ liệu thu thập, accuracy giảm 73,24% xuống 71,47%. Tác giả nêu chưa ràng buộc rõ vận tốc, pha hay liên tục tại biên; đây là giải thích giả thuyết chứ không phải nguyên nhân đã được cô lập. Không dùng Penn/J-HMDB; chưa xác nhận code. Sát câu hỏi khi nào phục hồi gây hại, nhưng dùng sinh chuyển động 3D và bối cảnh khác pipeline 2D dự kiến. Kiểm tra kích thước mẫu, selection, công bố code và protocol trong bản đầy đủ trước khi tái lập.

2. **[A robust two-stage framework for human skeleton action recognition with GAIN and masked autoencoder](https://www.sciencedirect.com/science/article/pii/S0925231225001055), Neurocomputing 2025.** STGAIN điền khớp thiếu, FJMAE nhận diện. Tác giả đánh giá pipeline hai giai đoạn trên dữ liệu tương tác người–robot tự xây dựng; FJMAE được đánh giá thêm trên NTU RGB+D 60/120. Không suy diễn toàn bộ pipeline đã được kiểm định trên NTU từ abstract. Code được bài trỏ tới [repository](https://github.com/GD1201/A-two-stage-network-for-action-recognition), chưa kiểm tra chạy được. Là prior work gần về imputation + classification, không dùng Penn/J-HMDB theo thông tin đã kiểm tra.

3. **[A Graph Convolutional Network for Action Recognition in Occluded Skeleton Data](https://www.mdpi.com/2079-9292/15/11/2311), Electronics, 26/05/2026.** ROC-GCN giảm ảnh hưởng quan sát không tin cậy và bù đặc trưng bị che khuất; đánh giá NTU 60/120 với che khuất ngẫu nhiên, liên tục và một tập con có khớp thiếu thực. Bù tại đặc trưng, không mặc định xuất tọa độ đã sửa. Hữu ích làm đối chứng khái niệm: classifier bền với thiếu khớp có thể không cần dựng lại pose. Chưa xác nhận code; không phải Penn/J-HMDB.

4. **[Deep Imputation for Skeleton Data (DISK)](https://www.nature.com/articles/s41592-025-02893-y), Nature Methods, online 04/12/2025, số 2026.** Học quan hệ khớp và động học để điền khoảng thiếu 2D/3D mà không cần gán nhãn thủ công; so với nội suy tuyến tính và nhiều kiến trúc. Bài gồm nhiều tập động vật và CMU Human MoCap 3D; không đánh giá trực tiếp pose 2D Penn/J-HMDB. Tác giả nhận thấy hiệu quả nội suy tuyến tính phụ thuộc tỷ lệ chuyển động tĩnh. [Code chính thức](https://github.com/bozeklab/DISK); [code thí nghiệm bài báo](https://github.com/bozeklab/DISK_paper_code/). Baseline học cho imputation, chưa chứng minh chuyển trực tiếp sang YOLO 2D hay cải thiện action accuracy của dự án.

## Nghiên cứu gần dùng dữ liệu dự án

5. **[ListPose](https://doi.org/10.1007/s12204-025-2815-7), xuất bản online 04/04/2025, số 2026.** Mô hình pose video dùng TokenPose và thông tin các frame liền kề; đánh giá Penn Action và Sub-JHMDB. Đây là ước lượng pose từ hình ảnh/đặc trưng, không phải phương pháp sửa chuỗi tọa độ độc lập; abstract không xác nhận hiệu quả lên action recognition. Trang bài chỉ cho xem preview, chưa xác minh chi tiết split, metric và code.

6. **[Enhancing Action Recognition from Low-Quality Skeleton Data via Part-Level Knowledge Distillation](https://arxiv.org/abs/2404.18206), 2024.** Penn Action, NTU và SYSU; nhận diện trên pose chất lượng thấp và distillation từ pose chất lượng cao. Với Penn, annotation thủ công được dùng làm một nguồn skeleton chất lượng cao; baseline student không distillation được train/test trên pose chất lượng thấp. Không phải temporal smoothing; là prior work quan trọng cho baseline train/test cùng phân phối và sự khác biệt pose GT/predicted. Chưa xác nhận code chính thức.

## Baseline cũ quan trọng dù không mới nhất

7. **[SmoothNet, ECCV 2022](https://arxiv.org/abs/2112.13715)**: mạng hậu xử lý chuỗi pose theo thời gian để giảm jitter, được kiểm tra trên nhiều tác vụ pose 2D/3D; [code](https://github.com/cure-lab/SmoothNet). Cần kiểm tra cách train, checkpoint, topology và có dùng frame tương lai trước khi áp dụng. Cải thiện pose không tự chứng minh cải thiện nhận diện.

8. **[Revisiting Skeleton-Based Action Recognition, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.pdf)**: PoseC3D biểu diễn pose bằng heatmap và đánh giá độ bền trước bỏ keypoint trên FineGYM, có xét train với nhiễu. [Code/cấu hình MMAction2](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md). Không xem HMDB51 là J-HMDB.

## Hàm ý cho thiết kế

- Khoảng vấn đề có căn cứ: cùng lúc đo sai số pose, metric liên tục theo thời gian và accuracy hành động; phân tích theo dạng lỗi, độ dài đoạn thiếu và tốc độ. Bài năm 2026 nêu ví dụ completion gây hại cho một điều kiện, nên không tuyên bố đây là câu hỏi chưa ai xét.
- Baseline tối thiểu: raw; nội suy tuyến tính; bộ lọc thích ứng; classifier train/test trên pose dự đoán; nếu khả thi SmoothNet hoặc DISK. Giữ cùng clip, nguồn pose, cách chọn người và classifier; chọn cấu hình bằng validation.
- Nếu xét Penn Action và J-HMDB/Sub-JHMDB, xác thực nội dung và split workspace; khác topology, convention trái/phải, nhãn khớp và tiền xử lý phải audit. Không so sánh số trực tiếp giữa các bộ hay các bài khác protocol, modality hoặc pretrained.
- Bằng chứng lịch sử của dự án có test từng ảnh hưởng phát triển: không xem đánh giá lại cùng tập như kiểm định hoàn toàn độc lập. Chưa chạy thực nghiệm mới. Các bài chưa có code được xác nhận không đồng nghĩa tác giả không công bố.
