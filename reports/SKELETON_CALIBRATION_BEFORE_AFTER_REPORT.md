# Trước/sau hiệu chỉnh: cách đọc hình và metric

Cập nhật diễn giải 29/10/2026 sau Bước 1–3. Các số dưới đây là artifact lịch sử, không phải lần chạy test mới trong đợt chẩn đoán. Kết luận hiện tại và phân biệt protocol xem [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md) và [protocol](EXPERIMENT_PROTOCOL.md).

Hình skeleton trông mượt hơn chỉ minh họa thay đổi tín hiệu. Cần kiểm tra riêng correspondence, đúng target, độ đúng tọa độ, lag và utility của classifier. Không dùng sự giống hình người hoặc độ dài xương ổn định để chứng minh kỹ thuật vận động đúng.

Theo [pose ablation lịch sử](SKELETON_CALIBRATION_ABLATION_STUDY.md), raw→Full giảm jitter 19,72→9,88px và bone std 7,32→5,64px; alignment-only→Full lại có PCK@20px 64,06→63,21%. Trên validation downstream mới, alignment-only→Full là 66,80→64,43%, Δ −2,37 điểm với CI [−5,14; +0,40]. Không gộp các split/mục tiêu thành bằng chứng một pipeline luôn tốt hơn.

Hình audit mới có GT vòng tròn/nét mảnh, YOLO dấu ×/nét đậm, cùng màu theo tên khớp, ba mapping trên cùng tọa độ raw. Cột heuristic minh họa permutation, không trộn thêm smoothing. Xem hình audit theo từng ID (`outputs/alignment_step1/figures`, chưa công bố trên nhánh này) và phạm vi visibility/confidence trong report Bước 1. Một số hình cho thấy sai target; không đánh giá chúng là sửa correspondence thành công.

Nguồn ưu tiên cho đóng góp từng thành phần: [audit correspondence](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md), [alignment versus Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md), [ablation 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
