# Protocol, trạng thái bằng chứng và cách trích dẫn

Cập nhật 29/10/2026 sau Bước 1–3. Các số liệu lịch sử được giữ trong artifact; việc viết lại report không tạo thêm lần đánh giá test.

| Nhóm | Dữ liệu | Checkpoint/chọn mô hình | Trạng thái |
|---|---|---|---|
| Bước 1 | 30 clip official train, 15 hành động | YOLO11n-Pose, không classifier | Pilot audit, chưa xác nhận convention toàn dataset |
| Bước 2 | 253 validation, 17.747 frame | BiLSTM 24D seed0, chọn bằng validation GT | Chẩn đoán Raw/Swap/Heuristic/Full |
| Bước 3 | Cùng 253 validation | Cùng checkpoint, swap lưu từ Bước 2 | Ablation 2³; không tính lại heuristic |
| Protocol v2 nhiều seed lịch sử | 1.005 train / 253 val / 1.068 test mỗi seed | Validation GT; seeds 0–4 | Benchmark đã lưu, không tách đóng góp alignment |
| Kết quả Penn trước v2 | Official split cũ | Từng dùng test GT để chọn checkpoint | Không làm bằng chứng test độc lập |
| J-HMDB/KD/AQA lịch sử | Khác tập/chế độ/mục tiêu | Xem report riêng | Không gộp với CI validation Bước 2–3 |

Ba bước mới không dùng official test. Checkpoint seed0 từng có test result trong benchmark lịch sử không làm validation trở thành test độc lập, và không chứng minh cả lịch sử phát triển phương pháp hoàn toàn không chịu ảnh hưởng từ test. Không khẳng định tính độc lập tuyệt đối chỉ vì checkpoint được chọn bằng validation.

Trong Protocol v2, chia theo video trong từng lớp của official train; seed và validation_fraction được lưu trong JSON. Code từ chối ID trùng, train/test giao nhau hoặc ba test modality khác tập ID/nhãn. Train chỉ dùng GT; checkpoint chọn bằng validation GT, augmentation nếu có chỉ áp train. Baseline kiểm tra validation mỗi epoch; các thử nghiệm lịch sử có lịch đánh giá riêng. Giữ nguyên test cho báo cáo cuối, không dùng để chọn phương pháp tiếp theo.

Trong Bước 2, dùng đúng danh sách seed0 từ metadata, xác minh 1.005 train và 253 validation giao rỗng, tất cả thuộc official train. Tái hiện validation GT 94,46640316% tại checkpoint best epoch 27. Raw được trích YOLO một lần với imgsz=640/conf=0.25, chọn box confidence lớn nhất không theo GT. Không bỏ clip thiếu detection hoặc nhầm người.

C3 lấy quyết định alignment từ đầu vào stage4 production rồi áp lên raw; C4 áp cùng permutation lên tọa độ đã qua ba stage. Bước 3 đọc `production_swap` đã lưu: 168 True, 85 False, chạy PCHIP→Bone→One-Euro với các module bật, swap cuối. Classifier batch32, cùng thứ tự video, lengths/padding, float32, normalize 24D và `eval()`.

Kết quả nhận diện mới dùng một checkpoint GT-trained BiLSTM 24D Protocol v2, seed0, trên 253 video validation / 17.747 frame; checkpoint từng được chọn bằng validation GT này. Không train lại, tuning, sửa alignment hay đánh giá official test trong ba bước chẩn đoán. CI là paired bootstrap theo video, phân tầng 15 lớp, 20.000 lần, seed 20261029, chưa hiệu chỉnh nhiều phép so sánh. Không diễn giải CI chứa/chạm 0 thành tương đương; không coi kết quả này là xác nhận tổng quát hóa độc lập.

Các CI đo Δ Top-1 bằng điểm phần trăm, không dùng phần trăm tương đối. Các contrast tương tác được bootstrap theo cùng video. CI chưa bao gồm biến thiên giữa seed/train, phụ thuộc giữa clip cùng nguồn hay chọn cấu hình sau khi xem validation. Macro-F1 trong bảng là point estimate. Các cận bằng 0 vẫn chứa 0.

Các sửa diễn giải bắt buộc khi dùng số liệu cũ:

- Penn pre-v2 raw 45,51→67,23% là +21,72 điểm theo chế độ chọn checkpoint cũ, không thay thế số validation 44,66→64,43% mới.
- Test v2 nhiều seed raw→full là so hai pipeline hoàn chỉnh; không phân rã lợi ích PCHIP/Bone/One-Euro.
- Pose ablation cũ: alignment-only PCK@20px 64,06%, Full 63,21%. Không gán giảm jitter thành tăng fidelity.
- J-HMDB GT-trained raw 40,30→calibrated 37,69%; calibrated-trained raw 46,27→calibrated 44,03%. 46,27% không phải kết quả test calibrated; không có thí nghiệm transfer Penn→J-HMDB trong các artifact này.
- Penn domain-matched 88,81% thiếu artifact tái lập tương ứng; không dùng như kết quả đã xác nhận.
- Tốc độ YOLO hoặc hậu xử lý riêng không phải latency toàn pipeline; có nội suy dùng frame tương lai và median toàn clip.

Không chạy lại train/test để viết báo cáo. Nguồn và cách khôi phục môi trường nghiên cứu nằm trong [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Dataset, checkpoint và cache npz không được đưa lên Git; CSV dự đoán, metric, cờ swap, hash và hình audit được công bố.

- [Bước 1: audit correspondence trên 30 clip train](LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md).
- [Bước 2: Raw / Fixed Swap / Heuristic Only / Full](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md).
- [Bước 3: thiết kế ablation đầy đủ 2³](CALIBRATION_ABLATION_STEP3_2026-10-29.md).
- [Protocol và phân cấp bằng chứng](EXPERIMENT_PROTOCOL.md).
- [Dữ liệu công bố và tái lập](REPRODUCIBILITY.md).

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
