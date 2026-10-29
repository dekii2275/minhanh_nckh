# Tổng hợp nghiên cứu sau ba bước kiểm chứng

Cập nhật 29/10/2026. Tài liệu này thay thế diễn giải trước đó về đóng góp của KineCalib; không có thực nghiệm mới trong lần chỉnh tài liệu.

**Kết luận hiện tại: gain Top-1 quan sát được chủ yếu đến từ alignment. Các module PCHIP, Bone và One-Euro chưa chứng minh lợi ích Top-1 bổ sung trong thiết lập đã kiểm tra.** Độ mượt của skeleton, độ đúng tọa độ và hiệu quả nhận diện là ba mục tiêu cần đo riêng.

Bài toán là kiểm tra chuỗi YOLO11n-Pose → xử lý skeleton → BiLSTM GT-trained. KineCalib là implementation hậu xử lý gồm nội suy PCHIP, ràng buộc độ dài đoạn xương 2D, One-Euro và một heuristic swap ở cấp clip; chưa coi đây là đóng góp thuật toán mới đã xác nhận.

Bước 1 kiểm tra correspondence trên 30 clip official train, 2 clip/hành động. 15 clip ủng hộ Fixed Swap nhất quán trên phần quan sát được; heuristic chọn đúng trong 11 clip thuộc nhóm này; 4 clip có đảo correspondence theo đoạn hoặc ngắn/cục bộ; 11 clip chưa kết luận vì nhầm người, mất detection hoặc che/chồng khớp. Không clip nào được xác nhận Identity nhất quán trong mẫu này, nhưng Identity đúng ở một số thời điểm. Các nhóm 15+4+11 không đồng nghĩa mọi frame đã được chấm tay. Audit chưa có hai annotator độc lập.

Dấu `median(x_left_shoulder - x_right_shoulder)` không đủ xác định correspondence cho toàn clip. Các phản ví dụ quay lưng cần swap, các đảo ngắn và lỗi chọn người khiến diễn giải Penn luôn viewer-centric/YOLO luôn có mapping đối lập không được xác lập. Fixed Swap cũng chưa được chứng minh đúng toàn dataset.

Bước 2 giữ cùng checkpoint, feature và video, tách alignment khỏi hiệu chỉnh tọa độ:

| Cấu hình | Top-1 (%) | Macro-F1 (%) | Δ so Raw (điểm phần trăm) |
|---|---:|---:|---:|
| C1 – Raw | 44.66 | 38.21 | +0.00 |
| C2 – Fixed Swap | 67.59 | 68.21 | +22.92 |
| C3 – Heuristic Only | 66.80 | 67.39 | +22.13 |
| C4 – Full KineCalib | 64.43 | 64.79 | +19.76 |

C3 dùng quyết định swap production sau các stage 1–3, rồi chỉ áp permutation lên raw. C4 dùng cùng quyết định trên tọa độ đã xử lý. Phân rã C1→C3→C4: **+22,13 điểm từ alignment**, rồi **−2,37 điểm từ calibration còn lại**, tổng +19,76. C4−C3 có CI [−5,14; +0,40]; C4−C2 là −3,16 [−7,11; +0,79]. Không có bằng chứng rõ về lợi ích bổ sung; chưa khẳng định calibration chắc chắn gây hại.

Bước 3 đóng băng 168 cờ swap / 85 cờ Identity của Bước 2, bỏ qua các stage bị tắt, giữ thứ tự PCHIP→Bone→One-Euro, áp permutation cuối. Không chạy lại heuristic:

| Cấu hình | Top-1 (%) | Macro-F1 (%) | Δ so Alignment Only (điểm phần trăm) |
|---|---:|---:|---:|
| A – Alignment Only | 66.80 | 67.39 | +0.00 |
| B – PCHIP | 66.40 | 67.16 | -0.40 |
| C – Bone | 65.61 | 65.80 | -1.19 |
| D – One-Euro | 65.22 | 65.84 | -1.58 |
| E – PCHIP + Bone | 65.22 | 65.46 | -1.58 |
| F – PCHIP + One-Euro | 66.40 | 67.75 | -0.40 |
| G – Bone + One-Euro | 64.43 | 64.81 | -2.37 |
| H – Full | 64.43 | 64.79 | -2.37 |

A và H tái tạo chính xác toàn bộ nhãn và posterior của C3/C4. PCHIP đơn lẻ cứu 2/làm sai 3 video; Bone 0/3; One-Euro 3/7; Full 4/10. CI của các so sánh chính so A đều chứa hoặc chạm 0. Bone có point estimate âm trong bốn ngữ cảnh; H−F là −1,98 [−3,95; −0,40], một tín hiệu khám phá chưa hiệu chỉnh đa kiểm định. PCHIP khi thêm vào One-Euro cho +1,19 [−0,79; +3,16], chưa đủ xác nhận lợi ích tương tác. F có Macro-F1 cao nhất nhưng thấp hơn A một video đúng.

Các module thực sự biến đổi đầu vào: PCHIP thay đổi 129 clip, lấp 1.922 giá trị xy NaN và giảm frame feature toàn 0 từ 184 xuống 99; Bone thay đổi 248 clip; One-Euro thay đổi 253 clip. Các chỉ số này không đo độ đúng pose hay chất lượng kỹ thuật của người tập. Có 42 clip đổi dự đoán, trong đó 18 clip đổi trạng thái đúng/sai giữa ít nhất hai cấu hình.

Kết quả nhận diện mới dùng một checkpoint GT-trained BiLSTM 24D Protocol v2, seed0, trên 253 video validation / 17.747 frame; checkpoint từng được chọn bằng validation GT này. Không train lại, tuning, sửa alignment hay đánh giá official test trong ba bước chẩn đoán. CI là paired bootstrap theo video, phân tầng 15 lớp, 20.000 lần, seed 20261029, chưa hiệu chỉnh nhiều phép so sánh. Không diễn giải CI chứa/chạm 0 thành tương đương; không coi kết quả này là xác nhận tổng quát hóa độc lập.

Các kết quả lịch sử được tách khỏi kết quả chẩn đoán:

| Nguồn | Có thể báo cáo | Không thể suy ra |
|---|---|---|
| Benchmark Protocol v2 nhiều seed đã lưu | Các metric test lịch sử, với split và modality cụ thể | Gain của từng module vì thiếu đối chứng alignment tương ứng |
| Pose/jitter/bone metrics lịch sử | Chỉ số tọa độ của implementation và mapping đó | Nhận diện tốt hơn, không độ trễ, tính đúng giải phẫu |
| J-HMDB split 1 lịch sử | Kết quả trong chính J-HMDB, gồm suy giảm accuracy | Transfer Penn→J-HMDB hoặc lợi ích phổ quát |
| KD lịch sử một seed | So sánh Teacher/Student đã lưu | KD phối hợp calibration tốt hơn mọi baseline |
| Movement profiler | Điểm proxy do công thức quy định | AQA được thẩm định, năng lực vận động hay hiệu quả huấn luyện |

Hướng giữ lại: Alignment Only và Fixed Swap làm đối chứng, artifact dự đoán ghép cặp, thiết kế tách correspondence khỏi hiệu chỉnh tọa độ. Bone cần ưu tiên xem lại giả định độ dài 2D trước foreshortening; nội suy/làm mượt cần kiểm tra gap, độ trễ và feature shift. Đây là giả thuyết tiếp theo, chưa phải cơ chế đã chứng minh. Chưa loại module khỏi production hoặc chạy bước tiếp theo.

Những tuyên bố được rút lại khỏi bản tổng hợp: pipeline full đã chứng minh vượt alignment-only; quy ước trái/phải toàn dataset đã được xác nhận; giảm jitter chứng minh tăng action utility; tốc độ một stage đại diện streaming end-to-end; AQA/khuyến nghị kỹ thuật đã được thẩm định; SOTA khi thiếu đối chứng cùng protocol. PCHIP dùng điểm tương lai và median clip nên không gọi toàn pipeline là causal streaming.

- [Bước 1: audit correspondence trên 30 clip train](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md).
- [Bước 2: Raw / Fixed Swap / Heuristic Only / Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md).
- [Bước 3: thiết kế ablation đầy đủ 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).
- [Protocol và phân cấp bằng chứng](EXPERIMENT_PROTOCOL.md).
- [Dữ liệu công bố và tái lập](REPRODUCIBILITY.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
