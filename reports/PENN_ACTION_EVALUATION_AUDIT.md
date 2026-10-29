# Audit correspondence: cập nhật kết luận

Cập nhật 29/10/2026. Báo cáo này thay thế khẳng định trước đây rằng đã chứng minh Penn Action viewer-centric toàn cục bằng khoảng cách tọa độ.

Bước 1 trên 30 clip official train cho 15 clip ủng hộ Fixed Swap nhất quán trên bằng chứng quan sát được, 4 clip đảo theo đoạn/ngắn/cục bộ, 11 chưa kết luận. Trong 15 clip nhất quán S, heuristic chọn đúng 11. Mẫu không xác nhận clip Identity nhất quán, nhưng Identity đúng ở một số thời điểm. Đây không phải audit toàn bộ frame hoặc hai annotator độc lập.

Heuristic dùng `median(pred_left_shoulder_x - pred_right_shoulder_x)>0` để swap sáu cặp cho toàn clip. Nhãn pose metadata không phải bằng chứng hướng của mọi frame. Cần phân biệt: đổi correspondence dự đoán, che/chồng khớp, sai target, quy ước nhãn thực sự và vị trí 2D. Swap cải thiện trung bình không chứng minh duy nhất cơ chế hay quy ước giải phẫu.

Phản ví dụ: 2062 quay lưng cần S nhưng heuristic I; 0864/2104 có các đoạn I/S khác nhau; 0757/2203 có đảo ngắn/cục bộ. 0137 và nhiều clip có nhầm target nên vote khoảng cách không đủ để xác nhận mapping. 1897 bị crop vai và có quyết định khác giữa raw và đầu vào stage4 production.

Không dùng GT để chọn permutation triển khai. So sánh I/S theo GT chỉ là chẩn đoán. Bước 2 dùng C3 theo quyết định production trên raw; Bước 3 đóng băng các cờ đó. Fixed Swap không được mặc định là chân lý toàn dataset.

Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
