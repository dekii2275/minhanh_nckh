> **Trạng thái sau audit:** Đây là bản review lịch sử trước Bước 1–3, được giữ để truy vết câu hỏi nghiên cứu. Các mục “thiếu alignment audit/ablation” đã được bổ sung ở phạm vi pilot train và một checkpoint validation; không phải đã hoàn thành xác nhận test độc lập. Kết quả mới và những giới hạn còn lại nằm ở [báo cáo tổng hợp](COMPREHENSIVE_RESEARCH_REPORT.md). Không dùng nhận xét trong bản review này thay cho kết quả thực nghiệm mới.

# Phản biện nghiên cứu KineCalib và chiến lược phát triển bài báo

Ngày đánh giá: 29/10/2026. Đối tượng: mã nguồn, bản thảo, báo cáo và artifact hiện có trong repository `minhanh_nckh`.

Đây là phản biện mô phỏng theo các vai trò được yêu cầu, không phải đánh giá của những chuyên gia độc lập thực sự. Tôi đã đọc mã nguồn và đối chiếu các JSON/CSV liên quan; chưa chạy lại huấn luyện, suy luận YOLO, kiểm tra toàn bộ video hay tái lập số đo tốc độ. Vì vậy, “đã xác minh” dưới đây nghĩa là xác minh từ tài liệu/mã/artifact được nêu, không đồng nghĩa tái lập độc lập kết quả thực nghiệm.

Quy ước bằng chứng:

- **V — Verified Finding:** Có bằng chứng trực tiếp trong mã, artifact hoặc nguồn gốc được dẫn.
- **R — Reasoned Concern:** Lo ngại có cơ sở phương pháp luận; chưa chứng minh là nguyên nhân thực tế.
- **H — Hypothesis:** Giả thuyết đề xuất, cần kiểm định.
- **M — Missing Evidence:** Chưa đủ bằng chứng để kết luận.

## 1. Executive Research Assessment

**Kết luận sơ bộ: chưa sẵn sàng nộp dưới dạng full research paper với các tuyên bố hiện tại.** Pipeline có giá trị triển khai và kết quả Penn Action đáng nghiên cứu, nhưng novelty thuật toán yếu; đóng góp thực nghiệm chưa tách được tác động của quy ước trái/phải, làm mượt và lệch phân phối train–test. Một số tuyên bố về tổng quát hóa, AQA, SOTA và thời gian thực vượt quá bằng chứng.

Hướng khả thi nhất là **nghiên cứu thực nghiệm có kiểm soát về điều kiện hiệu chỉnh pose giúp hoặc làm hại nhận diện hành động**, kết hợp công cụ kiểm toán có thể tái lập. Chưa nên đặt trọng tâm vào một kiến trúc mới, biomechanics hay chấm điểm chất lượng vận động.

### Diễn giải nghiên cứu theo góc nhìn reviewer

| Câu hỏi | Diễn giải và giới hạn |
|---|---|
| Bài toán gì? | Giảm khoảng cách giữa skeleton gán nhãn và skeleton ước lượng từ video khi dùng cho nhận diện hành động. |
| Vì sao đáng nghiên cứu? | Skeleton dự đoán có mất khớp, rung, nhầm bên và sai người; độ chính xác pose và giá trị cho tác vụ sau không nhất thiết đồng biến. |
| Phương pháp hiện tại? | Temporal filtering/refinement, biểu diễn heatmap, augmentation, huấn luyện với pose dự đoán và distillation. Đây là các hướng đã có, không phải một vấn đề chưa ai nghiên cứu. |
| Hạn chế nào quan trọng? | Chưa biết trong thiết lập cụ thể này phần suy giảm nào do lỗi quy ước, lỗi tọa độ hay mismatch phân phối; baseline thực dụng chưa đầy đủ. |
| Đề xuất giải quyết gì? | Kết hợp PCHIP, giới hạn độ dài đoạn xương 2D, One-Euro, hoán đổi trái/phải và đặc trưng 64D. |
| Cơ chế cải thiện? | Sửa correspondence khớp, lấp chỗ thiếu, giảm biến thiên; song tác động riêng lên accuracy chưa được chứng minh. |
| Kết quả mong đợi hợp lệ? | Cải thiện accuracy/F1 so với baseline đã chuẩn hóa quy ước, với chi phí nhỏ và điều kiện thất bại được mô tả. |
| Ai hưởng lợi? | Người xây dựng hệ thống phân tích video chi phí thấp và người tái sử dụng pose giữa các nguồn dữ liệu. Lợi ích cho huấn luyện viên/bệnh nhân chưa được kiểm chứng. |

**Một câu mô tả đóng góp hiện có:** “Một nghiên cứu thực nghiệm ban đầu cho thấy hậu xử lý skeleton kết hợp căn quy ước và hiệu chỉnh quỹ đạo có thể phục hồi nhận diện trên Penn Action trong thiết lập train bằng GT, trong khi kết quả J-HMDB cho thấy lợi ích không phổ quát.”

### Bằng chứng tích cực có thể giữ

**V:** Có 25 JSON cho 5 cấu hình × 5 seed của Protocol v2. Tôi kiểm tra ID train/validation/test không giao nhau trong metadata của từng run. Mã chọn checkpoint theo validation GT. Điều này hỗ trợ tính hợp lệ của bước chọn checkpoint hiện tại, nhưng không chứng minh toàn bộ quá trình phát triển chưa từng dùng test để lựa chọn thiết kế.

| Penn Action, Protocol v2 | GT Top-1 | Raw Top-1 | Calibrated Top-1 |
|---|---:|---:|---:|
| BiLSTM 24D | 91,63 ± 0,40 | 45,51 ± 3,16 | 65,47 ± 1,34 |
| Kinematics 64D | 92,15 ± 0,37 | 45,00 ± 3,16 | 68,54 ± 1,45 |
| Augmentation | 93,00 ± 0,56 | 44,53 ± 3,51 | 65,15 ± 1,66 |
| Spatial–temporal attention | 93,16 ± 0,90 | 44,81 ± 6,44 | 63,56 ± 3,49 |
| Combined | 93,33 ± 0,47 | 46,61 ± 1,50 | 69,83 ± 1,54 |

Đơn vị: %, mean ± sample standard deviation, 5 seed; code tổng hợp dùng `ddof=1`. Baseline tăng **19,96 điểm phần trăm**, không nên viết mơ hồ “tăng 19,96%”. Đây là kết quả so với raw trong cùng thiết lập GT-trained, chưa phải lợi ích so với baseline triển khai tốt nhất. Nguồn: [báo cáo đa seed](MULTI_SEED_BENCHMARK_REPORT.md), artifact tổng hợp (`outputs/action_recognition/protocol_v2/multi_seed/multi_seed_benchmark_summary.json`, chưa công bố trên nhánh này).

## 2. Research Contribution Scorecard

Điểm là đánh giá chất lượng bằng chứng hiện tại cho bài báo, không phải dự đoán xác suất được nhận. 10 là mạnh, 1 là rất yếu.

| Criterion | Score | Justification | Evidence Needed |
|---|---:|---|---|
| Problem Significance | 7 | Pose nhiễu và mismatch ảnh hưởng tác vụ thực tế; chưa lượng hóa nhu cầu triển khai cụ thể. | Bối cảnh sử dụng và mục tiêu độ chính xác/độ trễ. |
| Research Gap Clarity | 4 | Gap chung đã có prior work; gap có điều kiện chưa được phát biểu/kiểm chứng rõ. | Phân rã lỗi và đối chiếu trực tiếp prior work. |
| Scientific Novelty | 3 | Thành phần phần lớn là kỹ thuật đã biết; chưa chứng minh cơ chế mới. | Hiểu biết thực nghiệm mới hoặc quy tắc can thiệp vượt baseline. |
| Methodological Rigor | 5 | V2 cải thiện chia tập; thiếu kiểm soát convention và train modality. | Ma trận train–test, ablation downstream, kiểm soát tuning. |
| Experimental Strength | 5 | Có 5 seed Penn; J-HMDB là bằng chứng phụ, baseline hạn chế. | Ba split J-HMDB, baseline cạnh tranh, CI, lỗi từng video. |
| Practical Applicability | 4 | Chạy được pipeline offline; chưa chứng minh yêu cầu vận hành. | End-to-end benchmark và dữ liệu thực địa. |
| Reproducibility | 6 | Có source, metadata ID, artifact; tài liệu còn sai khác. | Commit/cache/checkpoint hashes, môi trường khóa phiên bản, predictions. |
| Publication Readiness | 3 | Các claim quan trọng còn thiếu hoặc trái bằng chứng. | Sửa toàn bộ claim và hoàn thành thí nghiệm quyết định. |

Không đủ dữ liệu để chấm độ chính xác AQA so với chuyên gia, hiệu quả lâm sàng, hay hiệu năng trên điện thoại: **N/A**. Chưa có nhãn chuyên gia, cohort hoặc benchmark thiết bị tương ứng.

## 3. Critical Weaknesses

### C1 — Critical: Chưa tách sửa correspondence khỏi cải thiện pose

**V:** module trái/phải (`outputs/research_snapshot/src/skeleton_calibrator.py`, chưa công bố trên nhánh này) lấy `median(x_left_shoulder - x_right_shoulder)` của toàn clip, rồi hoán đổi sáu cặp khớp cho toàn clip nếu trung vị dương. Đây là heuristic ở cấp clip; không phải mô hình giải phẫu, tracking hay tối ưu continuity theo frame. Điều kiện tọa độ vai không đủ xác định duy nhất hướng cơ thể trong mọi tư thế.

**V:** Báo cáo tổng hợp nói Penn viewer-centric/YOLO subject-centric, trong khi introduction bản thảo diễn giải ngược lại. Audit pilot trên vài chuỗi không đủ chứng minh quy tắc gán nhãn toàn dataset.

**R:** Phần lớn gain có thể là khôi phục correspondence đang bị so khớp sai. Nếu đúng, so full pipeline với raw chưa chuẩn hóa sẽ phóng đại đóng góp denoising.

Reviewer có thể hỏi: “Sau khi baseline được ánh xạ đúng, phương pháp còn hơn gì một bộ lọc chuẩn?”

**Khắc phục:** Kiểm tra độc lập mẫu train theo góc nhìn, cả tư thế quay lưng/quay người; xác định mapping từ tài liệu hoặc train, đóng băng trước test. So identity, fixed global swap, heuristic hiện tại, alignment-only và alignment + từng module trên cùng classifier. Oracle chọn theo GT chỉ được báo như chẩn đoán/cận tham chiếu, không làm phương pháp triển khai. Chi phí thấp–trung bình; đây là thí nghiệm bắt buộc.

### C2 — Critical: Chấm “chất lượng vận động” bằng chính đại lượng được làm đẹp

**V:** profiler (`outputs/research_snapshot/src/movement_profiling.py`, chưa công bố trên nhánh này) thưởng cho giảm sai phân quỹ đạo và giảm biến thiên độ dài đoạn xương; calibration trực tiếp tối ưu các đại lượng tương tự. Video người thật không đổi khi hậu xử lý skeleton, nhưng điểm “AQA” tăng từ 51,12 lên 64,01.

**R:** Đây là bằng chứng cải thiện chỉ số chất lượng tín hiệu theo định nghĩa nội bộ, không chứng minh người thực hiện động tác tốt hơn. Smoothness thấp có thể là động tác bùng nổ đúng kỹ thuật; độ dài xương 2D thay đổi có thể do phối cảnh.

Reviewer: “Tại sao làm mượt đo đạc lại được diễn giải thành cải thiện kỹ thuật của vận động viên?”

**Khắc phục:** Đổi tên thành *pose trajectory quality diagnostics*. Tách lỗi detector khỏi lỗi con người. Nếu giữ AQA, cần nhãn chấm độc lập của chuyên gia, độ nhất quán giữa người chấm, tương quan Spearman/MAE và baseline AQA phù hợp. Các nhãn “nguy cơ chấn thương”, “phân bổ lực tối ưu” trong code hiện không có cơ sở xác thực. Chi phí đổi phạm vi thấp; xác thực AQA thực sự cao hơn nhiều.

### M1 — Major: Chưa có raw-trained → raw-tested baseline

**V:** Protocol v2 Penn hiện train trên GT, chọn checkpoint trên validation GT và test ba modality. Hàm chuẩn bị tập (`outputs/research_snapshot/src/experiment_protocol.py`, chưa công bố trên nhánh này) yêu cầu train GT và ba test modality; chưa cung cấp thí nghiệm Penn raw-trained/calibrated-trained được tái lập cho số 88,81%.

**R:** Mismatch train–test có thể là yếu tố chính. Không có cơ sở quy toàn bộ chênh GT→raw cho chất lượng detector.

**Khắc phục:** Chạy ma trận GT/raw/calibrated train × GT/raw/calibrated test, với validation tương ứng từng điều kiện triển khai và một quy tắc tuning công bằng. Tách rõ zero-adaptation và domain-matched. Chi phí trung bình, tận dụng cache và BiLSTM hiện có.

### M2 — Major: Kết luận phổ quát trái với J-HMDB

**V:** Artifact/báo cáo ghi GT-trained raw 40,30% → calib 37,69% (**−2,61 điểm**); calib-trained raw 46,27% → calib 44,03% (**−2,24 điểm**). Không được gọi 46,27% là accuracy trên calibrated test. [Báo cáo protocol](EXPERIMENT_PROTOCOL.md) đã ghi nhận lỗi này.

**Khắc phục:** Báo kết quả âm, chạy đủ ba split, nhiều seed và phân tích tốc độ, góc nhìn, dropout. J-HMDB hiện là đánh giá trong dataset thứ hai, không phải train Penn rồi transfer sang J-HMDB. Chi phí trung bình; chưa cần dataset lớn mới.

### M3 — Major: Làm mượt và ràng buộc xương chưa chứng minh fidelity tốt hơn

**V:** [CSV ablation gốc](../outputs/calibration/ablation_summary.csv):

| Metric | Raw | Alignment-only | Full |
|---|---:|---:|---:|
| Temporal jitter, px theo định nghĩa hiện tại | 19,719 | 19,719 | 9,885 |
| Bone-length std, px | 7,321 | 7,321 | 5,635 |
| Mean error, px | 53,416 | 32,761 | 32,186 |
| Median error, px | 35,297 | 11,561 | 12,498 |
| PCK@20px, % | 22,869 | 64,057 | 63,214 |
| PCK-torso@0.20, % | 2,716 | 27,705 | 26,392 |

Full mượt hơn nhưng **PCK và median error kém alignment-only**. Không thể suy từ giảm jitter ra tăng độ chính xác hình học hoặc action utility. Các số 2,89→1,45 px và một số PCK-torso trong báo cáo tổng hợp không khớp CSV này.

**Khắc phục:** Tái sinh bảng từ artifact; báo fidelity, độ trễ pha, tốc độ đỉnh và accuracy song song. Thêm GT-clean → filtered-GT để đo mức phá hủy chuyển động thật. Chi phí thấp–trung bình.

### M4 — Major: “Real-time” chưa được chứng minh ở cấp hệ thống

**V:** PCHIP sử dụng điểm hợp lệ tương lai; giới hạn xương và quyết định swap dùng trung vị toàn clip; BiLSTM và attention hiện dùng ngữ cảnh hai chiều/toàn chuỗi. Mã mô hình (`outputs/research_snapshot/src/models/action_bilstm.py`, chưa công bố trên nhánh này).

Đây là các phép xử lý offline hợp lệ, không tự động là leakage nhãn test. Tuy nhiên chúng không chứng minh streaming nhân quả. FPS tính toán cao không loại bỏ thời gian chờ đủ clip hoặc chờ khớp xuất hiện trở lại.

**Khắc phục:** Chọn rõ offline hay online. Với online: cố định look-ahead, thay thống kê toàn clip bằng thống kê quá khứ, dùng cửa sổ nhân quả và đo độ trễ quyết định. Báo warm-up, batch size, GPU synchronization, p50/p95 và peak memory toàn pipeline. Chi phí đo thấp; chuyển online trung bình–cao.

### M5 — Major: Related work và claim SOTA chưa đáng tin cậy

**V:** CascadeFormer trong bảng được dẫn [6], nhưng [6] là Video Swin Transformer. Một số tên bài/tác giả/venue sai so với nguồn xác minh ở mục 5. Chưa có thực nghiệm cùng protocol với phương pháp mạnh; 69,83% là tốt nhất trong cấu hình nội bộ, không đủ gọi SOTA.

**Khắc phục:** Dùng reference manager, kiểm tra từng trích dẫn từ nguồn gốc, bỏ số không truy nguồn được; tách bảng bối cảnh literature khỏi bảng so sánh trực tiếp. Chi phí thấp, bắt buộc trước nộp.

### M6 — Major: KD chưa tạo thêm bằng chứng downstream tích cực

**V:** KD artifact (`outputs/action_recognition/kd_experiment/kd_summary.json`, chưa công bố trên nhánh này), seed 0: teacher calibrated 64,888%, student 64,232%, chênh **−0,655 điểm**. Student raw tăng 42,041→43,258%, nhưng chỉ một seed. “+20,97” là raw→calib của student, không phải KD cải thiện so teacher.

**V:** student dataset (`outputs/research_snapshot/src/run_kd_experiment.py`, chưa công bố trên nhánh này) dùng GT cộng nhiễu/dropout tổng hợp; chưa phải paired GT–YOLO thật. Chưa có CE-only cùng augmentation để cô lập KD.

**Khắc phục:** Giữ KD là exploratory; thêm CE-only/same noise, nhiều seed, rồi mới xem xét GT–detected pairs. Không cần đưa vào contribution chính nếu không vượt baseline.

### Các vấn đề Moderate và Minor

- **Moderate:** Một seed điều khiển cả chia validation lẫn khởi tạo, nên độ lệch chuẩn gộp hai nguồn biến thiên. Tách split seed/model seed hoặc giải thích rõ estimand.
- **Moderate:** Mean ± std không tự chứng minh “statistically significant”. Cần predictions theo video, bootstrap/paired test thích hợp và CI cho chênh lệch. Không coi hàng chục nghìn frame là các mẫu độc lập.
- **Moderate:** Chuẩn hóa (`outputs/research_snapshot/src/data/action_dataset.py`, chưa công bố trên nhánh này) dùng khoảng cách vai từng frame; ở góc nghiêng có thể khuếch đại nhiễu và tạo dao động scale. So với torso/robust clip scale, giữ cùng mapping.
- **Moderate:** Chọn người có box confidence cao nhất từng frame có thể đổi danh tính trong video nhiều người. Cần phân tích sai người và so tracking đơn giản.
- **Moderate:** Chưa kiểm toán tuning hyperparameter calibration trên test; V2 sửa checkpoint selection, không tự xóa mọi khả năng test-guided design trước đó.
- **Moderate:** `requirements.txt` dùng cận dưới phiên bản; chưa đủ khóa môi trường/cache/checkpoint để tái lập chặt.
- **Minor:** Đặc trưng 64D thực tế là 24 position + 24 sai phân bậc nhất + 16 bone-vector; không có acceleration, lực hay độ lớn dịch chuyển riêng như một số đoạn bản thảo. Sai phân chưa chia thời gian nên cần gọi đúng đơn vị.
- **Minor:** Bảng AQA theo lớp trong báo cáo không khớp JSON ở nhiều dòng. Ví dụ baseball_pitch báo 44,38→54,93 nhưng JSON là 50,49→63,56. Tổng hợp JSON theo trọng số 225 clip khớp 51,12→64,01; cần sửa bảng báo cáo và lưu dữ liệu từng clip.

## 4. Novelty & Contribution Analysis

### Kiểm toán thành phần

| Thành phần | Đã có/phổ biến hay đề xuất mới? | Giá trị khoa học hiện có |
|---|---|---|
| YOLO11n-Pose | Mô hình có sẵn; lựa chọn triển khai. | Nguồn pose, không phải contribution thuật toán. |
| One-Euro | Bộ lọc đã công bố năm 2012. | Baseline cần có; dùng lại không mới. |
| PCHIP và confidence threshold | Nội suy chuẩn + heuristic missingness. | Cần chứng minh tốt hơn linear/hold-last trong lỗi detector thật. |
| Chiếu giới hạn độ dài đoạn xương | Ràng buộc hình học theo median clip. | Chưa phải mô hình cơ sinh học; cần xử lý foreshortening. |
| Swap theo median vai | Heuristic cụ thể của implementation. | Có thể hữu ích để audit; novelty và đúng đắn toàn dataset chưa xác lập. |
| BiLSTM + attention | Kiến trúc quen thuộc. | Công cụ đo downstream, không đủ đóng góp kiến trúc. |
| Pos/velocity/bone 64D | Họ đặc trưng động học phổ biến. | Bằng chứng cải thiện nội bộ; không đủ claim thuật toán mới. |
| Protocol v2, 5 seed | Thực hành khoa học cần thiết. | Tăng độ tin cậy; không tự tạo benchmark mới. |
| KineCalib tích hợp | Tích hợp kỹ thuật đã biết. | Systems/empirical contribution tiềm năng nếu chứng minh lợi ích và giới hạn. |
| AQA heuristic | Thang điểm tự thiết kế. | Chưa có construct validity, không phải quality assessment đã xác thực. |

Theoretical: chưa có. Methodological/Algorithmic/Architectural: yếu. Empirical: có tín hiệu đáng phát triển. Dataset/Benchmark: chưa có benchmark mới được xác lập. Systems: prototype tiềm năng. Applied research: chưa có đánh giá người dùng/hiện trường độc lập.

### Sáu tình huống cần tránh nhầm lẫn

1. **Kết hợp công nghệ có sẵn — áp dụng rõ.** Có thể thành systems paper nếu có trade-off mới, ràng buộc triển khai thực và so sánh toàn hệ thống công bằng.
2. **Đổi mô hình — áp dụng một phần ở attention.** GT tăng nhưng calibrated giảm khi chỉ thêm ST attention; chưa có lý do coi kiến trúc phức tạp hơn là tốt hơn.
3. **Tối ưu triển khai — mới có số tốc độ thành phần.** Cần Pareto accuracy–latency và thiết bị đích trước khi claim systems efficiency.
4. **Ứng dụng vào miền mới — sports không phải miền chưa được nghiên cứu.** Cần nhu cầu cụ thể và dữ liệu kiểm chứng riêng; nhắc “rehabilitation” không tạo contribution y sinh.
5. **Tốt hơn trên một dataset — áp dụng.** Cần baseline mạnh, độ biến thiên và điều kiện tổng quát hóa; kết quả dataset thứ hai hiện phản bác claim phổ quát.
6. **Pipeline hoàn chỉnh — áp dụng rõ.** Completeness là lợi ích kỹ thuật; khoa học nằm ở hiểu biết/cơ chế/bằng chứng mới mà pipeline cho phép đo.

**Q1 — What is genuinely new?** Chưa xác lập một thuật toán mới. Ứng viên mạnh nhất là bằng chứng phân rã lỗi quy ước–hình học–chuyển động trong một thiết lập có kiểm soát; hiện chưa đủ thí nghiệm để khẳng định là đóng góp mới so prior work.

**Q2 — Why does it matter?** Nếu biết rõ khi nào smoothing gây mất thông tin, người triển khai có thể tránh xử lý làm đẹp skeleton nhưng giảm action accuracy, và tránh coi nhầm mapping là lỗi detector.

**Q3 — How can it be proven?** Giữ convention cố định, ma trận train–test, ablation downstream, nhiều tốc độ/FPS và detector, kiểm tra trên tập giữ kín, so baseline đơn giản và báo đầy đủ kết quả âm.

### Simpler Alternative Challenge

| Phương án đơn giản hơn | Vì sao có thể cạnh tranh? | Bằng chứng KineCalib phải cung cấp |
|---|---|---|
| Mapping đúng + raw-trained classifier | Loại mismatch mà không thêm nhiều hậu xử lý. | Gain bổ sung trên test tương ứng, cùng tuning budget. |
| Mapping + linear interpolation + One-Euro | Giữ phần có lý do rõ, bỏ bone projection/PCHIP phức tạp. | Accuracy/F1 tốt hơn hoặc fidelity/latency tốt hơn có ý nghĩa. |
| Confidence/missing mask + augmentation ở train | Cho classifier học độ tin cậy, tránh nội suy quá mức. | Tốt hơn dưới dropout thực, không chỉ Gaussian noise. |
| Savitzky–Golay/EMA/Kalman + TCN nhỏ | Temporal baseline rẻ và quen thuộc. | Pareto accuracy–latency tốt hơn với cùng look-ahead. |

## 5. Related Work & Research Gap Analysis

Tra cứu có mục tiêu dưới đây giúp bác bỏ novelty quá rộng; đây chưa phải systematic literature review đầy đủ đến mọi công bố năm 2026.

| Công trình đã xác minh | Liên quan trực tiếp | Hệ quả cho định vị |
|---|---|---|
| [Casiez et al., One-Euro, CHI 2012](https://gery.casiez.net/1euro/) | Bộ lọc thích ứng theo tốc độ, cân bằng jitter và lag. | Không nhận smoothing thích ứng là mới; phải đo lag thực tế. |
| [Zeng et al., SmoothNet, ECCV 2022](https://ailingzeng.site/smoothnet) | Refinement theo thời gian, nhiều estimator/dataset, cả 2D và 3D. | Claim model-agnostic denoising đã có prior art mạnh; cần so chất lượng/chi phí. |
| [Duan et al., PoseConv3D, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html) | Học từ 2D pose heatmap và xem xét chất lượng trích xuất pose cho action recognition. | Không thể nói benchmark action gần như chỉ dùng GT hoặc pose chất lượng cao. |
| [Cormier et al., WACV 2024 RWS Workshop](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Cormier_Enhancing_Skeleton-Based_Action_Recognition_in_Real-World_Scenarios_Through_Realistic_Data_WACVW_2024_paper.html) | Realistic augmentation; thư viện Skelbumentations. | Robustness với pose thật đã được nghiên cứu; Gaussian noise đơn thuần là baseline yếu. |
| [Liu et al., 2024, Part-Level Knowledge Distillation](https://arxiv.org/abs/2404.18206) | Teacher chất lượng cao hướng dẫn student chất lượng thấp; có Penn Action. | GT→noisy KD không phải ý tưởng mới và cần so đúng công trình. |
| [Ma et al., Frontiers in Public Health 2026](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1925220/full) | Interpretable movement profiles và formative feedback, có dữ liệu skeleton. | Hướng profiling đã có; phải phân biệt mô tả tín hiệu với chấm chất lượng kỹ thuật. |

### Lỗi thư mục tài liệu cần sửa ngay

- [1] trong bản thảo ghi sai tên bài Penn Action. Nguồn đúng là [From Actemes to Action: A Strongly-Supervised Representation for Detailed Action Understanding](https://openaccess.thecvf.com/content_iccv_2013/html/Zhang_From_Actemes_to_2013_ICCV_paper.html).
- Skelbumentations không khớp citation “A. Khokhlova … CVPRW” đang dùng; nguồn xác minh là Cormier et al. tại WACV 2024 workshop nêu trên.
- Citation Liu [13] không khớp tên và tác giả của công trình low-quality skeleton KD xác minh được. Không nên giữ metadata hiện tại khi chưa kiểm tra publisher.
- Article number 1925220 xác minh được, nhưng title và danh sách tác giả Ma [19] trong bản thảo sai. Nguồn chính thức là **Ma Q, Lei X, Li F, Yang C, Du J, Zhu J, Zhao D, Jia Z**, title như liên kết trên.
- [CascadeFormer](https://arxiv.org/abs/2509.00692) có nguồn thật, nhưng citation [6] trong bảng của bạn sai. Chưa xác minh lại con số 94,66% và FPS; tạm bỏ các con số này khỏi bảng so sánh chính.
- Chưa đủ cơ sở xác nhận metadata/số liệu Williams [18], Dai & Zhao [20] trong phạm vi tra cứu này. Đánh dấu “chưa xác minh”, không kết luận chúng giả chỉ vì chưa tìm được.

### Gap nào có thể bảo vệ?

| Loại gap | Đánh giá hiện tại | Điều kiện đủ mạnh |
|---|---|---|
| Knowledge gap | **H:** Chưa biết khi nào giảm jitter làm mất tín hiệu hành động trong thiết lập này. | Quy luật lặp lại trên nhiều classifier/dataset, gắn với tốc độ/occlusion và kiểm chứng ngoài tập dùng xây giả thuyết. |
| Methodological gap | **M:** Chưa chứng minh thiếu phương pháp phù hợp. | Chỉ ra baseline hiện có thất bại ở điều kiện cụ thể và một cơ chế khắc phục vượt baseline. |
| Empirical gap | Có khả năng mạnh nhất: convention, denoising và train-domain đang bị trộn. | Thiết kế factorial, predictions mở và phân rã hiệu ứng đáng tin cậy. |
| Application gap | Chưa có đánh giá hiện trường. | Bài toán người dùng rõ, ground truth độc lập và so quy trình thực dụng. |
| Engineering gap | Hậu xử lý nhẹ có thể hữu ích. | End-to-end trade-off và yêu cầu tài nguyên thực, không chỉ FPS module. |

Gap đề xuất nên viết có điều kiện: “Chúng tôi nghiên cứu ảnh hưởng riêng và tương tác của correspondence, hậu xử lý thời gian và phân phối huấn luyện đến nhận diện từ pose dự đoán chi phí thấp.” Không viết “chưa có công trình nào” nếu chưa có review hệ thống hỗ trợ.

## 6. Experimental Validation Plan

Tất cả hàng dưới là **thí nghiệm đề xuất**, trừ các kết quả hiện có đã trích ở mục 1 và 3. “Expected Evidence” là dạng bằng chứng cần thu, không phải kết quả dự đoán sẽ đạt được.

| Experiment | Research Question | Hypothesis | Experimental Setup | Baseline | Metrics | Expected Evidence |
|---|---|---|---|---|---|---|
| A1. Convention audit | GT/detector có thực sự khác quy ước? | H1: Một phần lỗi là correspondence. | Mẫu train theo lớp/góc nhìn, hai người kiểm tra độc lập; đóng băng mapping rồi test. | Identity, fixed swap, heuristic. | Tỷ lệ nhầm bên, đồng thuận người kiểm, PCK. | Phạm vi quy ước và lỗi nhãn, tránh tổng quát từ pilot. |
| A2. Ma trận train–test 3×3 | Calibration giúp ngoài mismatch train–test không? | H2: Gain giảm sau domain matching. | GT/raw/calib train × GT/raw/calib test; cùng ID; val tương ứng. | Raw→raw; aligned raw→aligned raw. | Top-1, macro-F1, Δ và CI. | Giá trị tăng thêm thực tế của calibration. |
| B1. Baseline đơn giản | Full có cần thiết? | H3: Một cấu hình nhỏ đạt tương đương. | Mapping cố định; linear/PCHIP; EMA/One-Euro/SG; cùng tuning budget. | Alignment-only, linear+One-Euro. | Accuracy, fidelity, lag, runtime. | Có/không có lợi ích của độ phức tạp. |
| B2. Classifier cạnh tranh | Gain phụ thuộc BiLSTM không? | H4: Hiệu ứng thay đổi theo biểu diễn. | TCN nhỏ và một ST-GCN hoặc PoseConv3D phù hợp tài nguyên; cùng dữ liệu. | BiLSTM hiện tại. | Top-1/F1, params, latency. | Phạm vi hiệu lực qua classifier. |
| C1. Ablation downstream | Module nào thực sự tạo gain? | H5: Alignment chi phối gain Penn. | Alignment-only, +interpolation, +bone, +smooth; leave-one-out; 24D/48D/40D/64D. | Cùng checkpoint và mapping. | Paired Δaccuracy/F1; PCK; lag. | Tác động riêng/tương tác, không suy từ jitter. |
| C2. Clean-motion control | Filter phá hủy chuyển động thật không? | H6: Fast motion dễ bị làm sai. | GT vs filtered GT; synthetic corruption có nhiễu thật được mô phỏng. | Identity GT, One-Euro. | GT action accuracy, vị trí/timing đỉnh vận tốc. | Giá phải trả cho smoothing. |
| D1. Robustness | Các lỗi nào làm phương pháp thất bại? | H7: Long dropout và yaw rotation gây hại. | Mức dropout liên tục, nhiễu, FPS/thay frame, crop/scale có kiểm soát. | Không hiệu chỉnh và baseline đơn giản. | Đường cong robustness, AUC, error theo mức. | Miền hoạt động và ngưỡng thất bại. |
| E1. J-HMDB ba split | Kết quả âm có lặp lại? | H8: Tác động phụ thuộc dataset/loại động tác. | Ba split chuẩn; nhiều seed; cùng quy tắc validation. | Raw-trained và alignment-only. | Mean theo split/seed, F1, CI cho Δ. | Kết luận tổng quát có giới hạn, gồm kết quả âm. |
| E2. Transfer calibration | Policy được chọn ở Penn có chuyển được? | H9: Policy cố định vẫn hữu ích ở một phần J-HMDB. | Khóa tham số ở Penn; giữ J-HMDB để đánh giá; classifier train theo protocol đích. | Calibration tham số mặc định và raw. | Accuracy/fidelity Δ. | Transfer policy; không gọi là transfer classifier. |
| E3. Detector thứ hai | Có thật model-agnostic? | H10: Hiệu ứng không riêng YOLO11n. | Pose estimator nhẹ thứ hai trên subset rồi mở rộng nếu có tín hiệu. | Raw/aligned output mỗi detector. | Δaccuracy, PCK, cost. | Giới hạn hoặc khả năng khái quát qua detector. |
| F1. Efficiency | Có đáp ứng ngân sách triển khai? | H11: Bản nhỏ nằm trên Pareto frontier. | Cố định thiết bị, input, batch=1, warm-up; đo decode→prediction; nếu online cố định look-ahead. | Raw pipeline và cấu hình nhỏ. | p50/p95, throughput, peak RAM/VRAM, wait time. | Chi phí thực và trade-off accuracy–latency. |
| G1. Failure analysis | Vì sao có clip bị hại? | H12: Foreshortening/đổi người/fast motion giải thích một phần. | Lấy toàn bộ clip raw đúng→calib sai và ngược lại; phân nhóm mù theo quy tắc. | Raw predictions cùng clip. | Số clip, loại lỗi, confusion matrix, error severity. | Failure taxonomy có số lượng và ví dụ truy vết. |
| KD optional | KD hơn augmentation không? | Chưa có hướng lợi ích ổn định. | CE-only/noise giống student; KD; nhiều seed; chỉ sau A–C. | CE-only same noise. | Δraw/calib accuracy và CI. | Phần tăng thực do loss KD. |
| AQA optional | Điểm có phản ánh chuyên gia? | Chưa biết. | Thu nhãn độc lập, blind raw/calib, lặp người chấm; held-out subject. | Chỉ số đơn giản và mô hình AQA phù hợp. | Inter-rater reliability, Spearman, MAE. | Construct validity; không dùng score tăng sau smoothing làm ground truth. |

### Quy tắc thiết kế và thống kê

1. Giữ official test; chia theo video, kiểm tra duplicate/nguồn video hoặc subject nếu metadata cho phép. Disjoint ID không đủ loại mọi near-duplicate.
2. Với so sánh zero-adaptation, giữ cùng GT-trained checkpoint. Với câu hỏi triển khai, train/val đúng modality. Không trộn hai estimand trong một kết luận.
3. Chọn calibration và hyperparameter bằng train/validation. Nếu đã dùng test để thiết kế, công khai lịch sử và dùng một tập ngoài để xác nhận giả thuyết mới.
4. Lưu dự đoán/xác suất từng video, không chỉ confusion matrix. Bootstrap paired ở đơn vị video; nếu có group nguồn/subject, bootstrap theo group. Với nhiều seed trên cùng test, không coi seed×video là các mẫu độc lập hoàn toàn; báo riêng variation do seed và bất định test hoặc dùng hierarchical bootstrap.
5. Báo mean/std, CI của chênh lệch, effect size và điều chỉnh đa so sánh khi thử nhiều giả thuyết xác nhận. Nêu trước primary comparison; thí nghiệm khám phá được đánh dấu rõ.
6. PCK pixel chỉ so trực tiếp khi độ phân giải/scale phù hợp; thống nhất torso definition, visibility mask và mẫu số missing predictions. Jitter sai phân chưa chia thời gian có đơn vị theo frame, không phải acceleration vật lý đã chuẩn hóa.
7. Ground-truth skeleton là điều kiện tham chiếu của classifier, không phải upper bound toán học: học từ noisy pose đôi khi có thể vượt nó do regularization hoặc thông tin biểu diễn khác.

## 7. Real-World Applicability Analysis

**Mức hiện tại: Proof of Concept có kiểm chứng trên benchmark offline.** Chưa đủ gọi Validated Prototype trong môi trường mục tiêu, càng chưa phải Real-World Validated System.

| Khía cạnh | Đánh giá | Bằng chứng cần bổ sung |
|---|---|---|
| Problem–solution fit | Có ích cho tái sử dụng pose khác nguồn; nhu cầu smart coach chưa được định lượng. | Chọn một workflow cụ thể và chi phí lỗi. |
| Deployment feasibility | Huấn luyện mô hình nhỏ trên RTX 3050 là khả thi theo artifact; chưa có đo điện thoại/CPU toàn hệ thống. | Thiết bị đích và benchmark đầy đủ. |
| Operational constraints | Offline, cần clip và có future context. | Mục tiêu clip-level hoặc causal look-ahead rõ. |
| Scalability | Nhiều stream tăng chi phí pose; attention toàn chuỗi có chi phí theo bình phương độ dài chuỗi. | Concurrent streams, queueing, giới hạn độ dài clip. |
| Robustness | J-HMDB đã cho tín hiệu giới hạn; chưa có đánh giá camera/góc nhìn mới. | Held-out camera/subject và stress tests. |
| Cost–benefit | Calibration có thể rẻ, nhưng maintenance mapping và lỗi sau làm mượt chưa định lượng. | Accuracy–latency–failure trade-off so raw-trained. |
| Practical baseline | Thiếu baseline raw-trained, tracking và interpolation đơn giản. | So cùng pipeline đầu vào/thiết bị/budget. |
| Deployment gap | Video benchmark được cắt sẵn; thực tế có nhiều người, hành động nền/unknown và thời điểm bắt đầu–kết thúc không biết. | Continuous video, rejection/abstention, tracking, onset/offset nếu phạm vi yêu cầu. |

Các bước nâng cấp: (1) xác định một tác vụ clip-level rõ; (2) thu pilot nhỏ từ camera mục tiêu với người chưa xuất hiện ở train; (3) đo lợi ích và lỗi bằng nhãn độc lập; (4) chỉ sau đó mở rộng người dùng/thiết bị. Không cần triển khai app hoàn chỉnh để chứng minh câu hỏi khoa học ban đầu.

**Giá trị cho tổ chức hiện nay:** công cụ kiểm toán mapping và chất lượng quỹ đạo có thể giúp phát hiện lỗi tích hợp trước huấn luyện. Chưa chứng minh nó chấm kỹ thuật, dự báo chấn thương hoặc cải thiện kết quả huấn luyện tốt hơn giải pháp sẵn có. Phiên bản mạnh hơn cần chứng minh giảm lỗi downstream/chi phí điều tra lỗi với baseline thực dụng.

## 8. Simulated Peer Reviews

Điểm dưới đây theo chuẩn một full research paper quốc tế về computer vision/action recognition. Đây là mô phỏng có chung hồ sơ bằng chứng, không phải ba phản biện độc lập đã được mời.

### Reviewer #1 — Novelty-Focused

**Summary:** KineCalib tích hợp hậu xử lý chuẩn để cải thiện nhận diện từ pose dự đoán; contribution mới so prior art chưa rõ.

**Strengths:** Bài toán thực; có multi-seed; audit correspondence có thể tạo insight thực nghiệm.

**Major Weaknesses:** Phần lớn thành phần đã biết; chưa so trực tiếp SmoothNet/realistic augmentation/low-quality KD; không tách gain do mapping; “SOTA” không được hỗ trợ.

**Minor Weaknesses:** Citation sai, 64D diễn giải không khớp code, contribution list quá dài và phân tán.

**Questions for Authors:**

1. Sau khi mapping hai miền đã chuẩn hóa, đâu là kết quả chứng minh tính mới còn lại?
2. Phát hiện nào không thể dự đoán từ prior work về smoothing, noisy skeleton và train-domain mismatch?
3. Vì sao kết hợp bốn bước này là một phương pháp nghiên cứu mới thay vì một cấu hình engineering?
4. Có bằng chứng nào cho thấy quy tắc median vai đúng khi người xoay, nghiêng hoặc quay lưng?

**Missing Evidence:** Novelty matrix dựa nguồn gốc, alignment-only downstream, baseline cùng protocol.

**Recommendations:** Định vị empirical paper; ưu tiên tìm quy luật có điều kiện và phản ví dụ, giảm claim thuật toán.

**Overall Score:** 3/10. **Confidence:** 4/5 cho hạn chế hồ sơ hiện tại; độ phủ literature chưa đủ để khẳng định không có novelty nào.

**Preliminary Recommendation:** Reject.

### Reviewer #2 — Methodology-Focused

**Summary:** Protocol v2 cải thiện checkpoint selection, nhưng thiết kế chưa xác định cơ chế của gain và có metric/task mismatch.

**Strengths:** Giữ official test; split theo video; 5 seed; artifact cho phép kiểm tra kết quả âm.

**Major Weaknesses:** Thiếu train-domain matrix, ablation chỉ số pose chưa đủ chứng minh action gain, thống kê significance chưa báo, kết luận J-HMDB bị thổi mạnh, AQA đánh giá vòng tròn.

**Minor Weaknesses:** Seed gộp split và initialization; bảng cũ/mới lẫn nhau; thiếu predictions từng video và môi trường khóa phiên bản.

**Questions for Authors:**

1. Raw-trained→raw-tested và alignment-only đạt bao nhiêu với cùng validation/tuning budget?
2. Vì sao full có PCK/median error kém alignment-only, và điều này tác động action accuracy thế nào?
3. Có bao nhiêu lựa chọn thiết kế/hyperparameter được quyết định sau khi xem official test?
4. Với J-HMDB, kết quả âm còn giữ khi chạy cả ba split và nhiều seed không?

**Missing Evidence:** Predictions paired, CI cho chênh lệch, calibration tuning provenance, factorial ablations và CE-only KD.

**Recommendations:** Làm sạch bảng số liệu trước; hoàn thành ma trận train–test và ablation downstream; dừng suy luận từ smoothness sang motion quality.

**Overall Score:** 4/10. **Confidence:** 4/5, giới hạn bởi chưa tái lập train và kiểm tra dữ liệu gốc.

**Preliminary Recommendation:** Reject.

### Reviewer #3 — Application-Focused

**Summary:** Hệ thống offline nhỏ có tiềm năng làm công cụ audit pose; chưa chứng minh smart coaching hoặc vận hành thời gian thực.

**Strengths:** Có pipeline chạy được; tính toán nhẹ ở thành phần calibration; có khả năng cung cấp diagnostics dễ diễn giải.

**Major Weaknesses:** FPS module thay cho end-to-end latency; future context; chưa đánh giá video liên tục/nhiều người; thiếu nhãn chất lượng động tác và baseline thực dụng.

**Minor Weaknesses:** Chưa nêu thiết bị/người dùng đích, tiêu chí chấp nhận lỗi và chính sách từ chối khi pose kém.

**Questions for Authors:**

1. Hệ thống phải chờ bao lâu từ lúc hành động bắt đầu đến lúc có quyết định với độ tin cậy dùng được?
2. Khi điểm AQA tăng chỉ vì hậu xử lý cùng một video, cơ sở nào để nói vận động viên có kỹ thuật tốt hơn?
3. Giá trị tăng thêm so với raw-trained classifier + tracking + interpolation đơn giản là gì?
4. Lỗi detector, phối cảnh và lỗi người tập được phân biệt bằng ground truth độc lập nào?

**Missing Evidence:** Benchmark thiết bị đích, hiện trường, nhãn chuyên gia và đo giá trị workflow.

**Recommendations:** Giới hạn claim thành offline pose auditing; nếu chọn applications paper, xác thực một use case hẹp trước khi mở rộng.

**Overall Score:** 3/10. **Confidence:** 4/5 về thiếu bằng chứng deployment; chưa thể chấm utility người dùng đáng tin cậy.

**Preliminary Recommendation:** Reject.

## 9. Research Improvement Roadmap

Ưu tiên dựa tỷ lệ giá trị khoa học/chi phí. Thời gian và nhân lực thực chưa được cung cấp; mức effort dưới đây là tương đối, không phải cam kết lịch.

| Priority | Problem | Proposed Improvement | Scientific Benefit | Effort | Validation |
|---|---|---|---|---|---|
| P0 / Level 1 | Claim và số liệu không nhất quán | Một nguồn bảng từ artifact; sửa citation; bỏ SOTA, cross-dataset transfer, AQA đã xác thực. | Khôi phục khả năng kiểm toán; không tăng novelty riêng. | Thấp | Mỗi claim có artifact/nguồn và protocol. |
| P0 / Level 1 | Convention bị trộn với denoising | Audit độc lập + alignment-only + downstream ablation. | Xác định cơ chế thật của gain. | Thấp–TB | Gain còn lại sau mapping và CI. |
| P0 / Level 1 | Thiếu practical baseline | Ma trận train–test, raw-trained và calibrated-trained. | Tách mismatch khỏi calibration utility. | TB | Cùng split/budget; prediction paired. |
| P1 / Level 1 | Kết luận quá rộng | Ba split J-HMDB, báo kết quả âm, failure groups. | Giới hạn tổng quát hóa có thể bảo vệ. | TB | Hiệu ứng theo split/seed/tốc độ. |
| P1 / Level 2 | Thiếu insight mới | Đo motion loss và xây policy chọn raw/filter từ train/val. | Contribution phương pháp/empirical có thể đo. | TB | Vượt always-filter và never-filter trên held-out. |
| P1 / Level 2 | Pipeline phức tạp chưa cần thiết | Pareto với linear+One-Euro, confidence mask, TCN; estimator thứ hai ở subset. | Quy tắc lựa chọn đơn giản, có giá trị triển khai. | TB | Accuracy/F1/latency và detector transfer. |
| P2 / Level 2 | Tái lập chưa hoàn chỉnh | Release cache schema, hashes, predictions, configs và script tái sinh báo cáo. | Artifact đáng sử dụng và kiểm chứng. | Thấp–TB | Chạy từ môi trường sạch trên subset. |
| P2 / Level 3 | Cần đóng góp sâu hơn | Benchmark lỗi pose có taxonomy và protocol can thiệp cố định. | Benchmark/knowledge contribution tiềm năng. | Cao | Nhiều detector/task, kiểm chứng nhãn lỗi và usefulness. |
| P3 / Level 3 | Muốn AQA thực sự | Thu nhãn kỹ thuật động tác, xác thực chuyên gia, tách lỗi đo. | Applied science nếu nhãn và kết quả đủ tin cậy. | Cao | Reliability, external test, utility độc lập. |

### Đánh giá giá trị, nguồn lực và rủi ro

| Hướng | Scientific Value / Novelty Potential | Practical Value | Difficulty / Resources | Experimental Requirements | Risk / Expected Impact |
|---|---|---|---|---|---|
| Level 1: clean evidence + causal separation | Giá trị cao về hợp lệ, novelty thấp–TB. | Cao cho sửa tích hợp. | Một người + cache + GPU hiện có; không cần model lớn. | A1, A2, C1. | Có thể lộ gain biến mất; vẫn cho quyết định nghiên cứu đúng. |
| Level 2: motion-aware selective filtering | Cao nếu có quy luật/policy vượt baseline. Novelty chưa xác lập. | Cao nếu tránh hại fast motion. | Trung bình; thống kê đơn giản trước, chưa cần transformer. | C2, D1, E1/E2 và held-out policy. | Proxy không dự đoán được harm; tác động tốt nếu rule chuyển miền. |
| Level 2: reproducible robustness study | Cao về bằng chứng; novelty phụ thuộc phát hiện mới. | Cao cho người chọn pipeline. | Trung bình; mở rộng có chọn lọc detector/classifier. | B1/B2, E3, F1/G1. | Chỉ thành bảng benchmark nếu không có insight. |
| Level 3: error benchmark | Tiềm năng cao, cần độ phủ. | Cao cho cộng đồng nếu dùng lại được. | Cao về annotation/curation hơn GPU. | Independent labels, nhiều nguồn, protocol release. | Nhãn không tin cậy/benchmark quá hẹp. |
| Level 3: expert-validated AQA | Giá trị ứng dụng cao; novelty chưa biết. | Có thể cao cho use case hẹp. | Cần chuyên gia và dữ liệu có chất lượng thực, không chỉ compute. | Nhãn độc lập, reliability và external validity. | Scope phình, construct validity yếu; không nên là bước kế tiếp mặc định. |

**Điểm quyết định:** Nếu alignment-only hoặc raw-trained đạt tương đương full, hãy công bố kết quả phân rã/negative finding nếu đủ mới và chặt, hoặc đổi đóng góp thành toolkit audit. Không tiếp tục thêm module để cứu câu chuyện “pipeline mới”. Nếu policy đơn giản không vượt baseline trên held-out, không nâng nó thành contribution.

## 10. Conference Targeting Strategy

Research domain: computer vision, skeleton-based action recognition, pose refinement và empirical robustness. Community: human motion understanding, pattern recognition, applied vision. Paper type phù hợp nhất sau chỉnh sửa: empirical research/benchmark methodology; systems/applications chỉ khi có đo deployment. Audience: nhà nghiên cứu và kỹ sư dùng pose dự đoán cho tác vụ downstream.

Các lựa chọn dưới là đánh giá fit có điều kiện, không phải dự đoán được nhận. Scope đã đối chiếu website chính thức của kỳ được dẫn; kỳ tương lai phải kiểm tra CFP mới. Không suy deadline hoặc track chưa công bố.

| Conference | Relevant Scope/Track | Expected Contribution | Fit | Main Weakness to Address | Required Evidence |
|---|---|---|---|---|---|
| **A. WACV main conference**, kỳ còn phù hợp lịch | Computer vision systems/applications và algorithms; [CFP chính thức](https://wacv.thecvf.com/Conferences/2026/CallForPapers). | Insight/method đáng kể với validation mạnh. | Ambitious; hiện chưa đạt. | Novelty và lợi ích so practical baselines. | Nhiều detector/classifier, controlled study, efficiency và external validation. |
| **A/B. ICPR main conference**, kỳ kế tiếp phù hợp | Pattern recognition/computer vision; [website chính thức](https://icpr2026.org/). | Nghiên cứu thực nghiệm có đóng góp rõ, protocol chặt. | Có thể phù hợp sau nâng cấp đáng kể; không mặc định dễ. | Chưa có insight mới và benchmark so sánh đủ mạnh. | Factorial study, 3 split J-HMDB, statistics và reproducible artifacts. |
| **B. KSE full research paper**, kỳ tương lai | Image processing/pattern recognition, ML applications, systems; [CFP](https://kse2026.kse-conferences.org/call-for-papers/). | Bài toán hẹp, đóng góp empirical/system rõ. | Tương đối phù hợp nếu hoàn thành P0/P1. | Tích hợp có sẵn và claim thiếu chứng cứ. | Alignment control, modality matrix, baseline nhỏ và kết quả tái lập. |
| **B/C. RIVF research paper**, kỳ tương lai | Computing/communication technologies; cần xác nhận phiên CFP và scope AI/vision cụ thể; [website chính thức](https://rivf2026.org/). | Applied/empirical study có chất lượng. | Có điều kiện, cần đọc CFP kỳ nộp. | Utility và rigor chưa đủ. | Controlled comparisons và use case rõ; nếu systems cần end-to-end measurements. |
| **C. Workshop paper về human motion/robust vision** | Chỉ chọn sau khi CFP workshop thực tế phù hợp; [WACV 2024 RWS precedent](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Cormier_Enhancing_Skeleton-Based_Action_Recognition_in_Real-World_Scenarios_Through_Realistic_Data_WACVW_2024_paper.html). | Phát hiện hẹp về failure mode hoặc benchmark/toolkit. | Có thể phù hợp hơn full paper với bằng chứng ban đầu được sửa. | Thiếu insight có kiểm chứng và đối chiếu realistic augmentation. | Ablation đúng, giới hạn trung thực và artifact. Không giả định workshop đó sẽ mở lại. |

Tại **29/10/2026**, các kỳ 2026 nêu trên không được xem là cửa nộp mới mặc định: ICPR 2026 đã diễn ra theo lịch chính thức; CFP WACV 2026 là kỳ cũ; KSE 2026 công bố hạn full paper 03/08/2026; RIVF 2026 công bố hạn 15/09/2026. Nguồn: [ICPR](https://icpr2026.org/), [WACV](https://wacv.thecvf.com/Conferences/2026/CallForPapers), [KSE](https://kse2026.kse-conferences.org/), [RIVF](https://rivf2026.org/). Chưa xác minh CFP tương lai đủ để chốt lịch nộp.

Short paper, workshop và demo/system paper là các loại bài khác nhau; chỉ chọn loại được CFP cụ thể hỗ trợ. Demo cần hệ thống hoạt động và bằng chứng utility; không thay thế được contribution của full paper. Các tên IEEE TMM, IEEE Access và Journal of Sports Analytics trong manuscript là **tạp chí**, cần tách khỏi chiến lược hội nghị.

Chiến lược: hoàn thành P0 trước khi chọn venue; nếu chỉ còn insight hẹp, nhắm workshop phù hợp hoặc full paper phạm vi hẹp tại venue có scope phù hợp; nếu có quy luật chuyển miền hoặc policy vượt baseline đáng tin cậy, cân nhắc ICPR/WACV. Không dùng tên venue để hợp lý hóa chất lượng bằng chứng hiện tại.

## 11. Revised Scientific Contributions

### Refined Research Problem

“Trong nhận diện hành động từ pose 2D dự đoán, tác động của căn correspondence và hậu xử lý quỹ đạo thay đổi như thế nào theo phân phối huấn luyện, động học chuyển động và đặc điểm lỗi detector?”

### Research Motivation

Pipeline pose nhẹ có thể tạo tín hiệu mượt nhưng làm sai timing/biên độ chuyển động. Người triển khai cần biết can thiệp nào có ích cho tác vụ cuối, thay vì tối ưu chỉ số hình thức của skeleton. Kết quả Penn dương và J-HMDB âm là động lực điều tra, chưa phải bằng chứng đã giải thích được nguyên nhân.

### Validated or Proposed Research Gap

Đã xác minh trong dự án: thiếu kiểm soát correspondence, train-domain và downstream module effects. Gap của toàn ngành còn **đề xuất**, cần review literature sâu hơn: thiếu quy tắc can thiệp có khả năng chuyển miền để cân bằng giảm nhiễu và bảo toàn thông tin hành động trong thiết lập chi phí thấp.

### Research Questions và Hypotheses

| RQ | Giả thuyết có thể bác bỏ | Cách kiểm định |
|---|---|---|
| RQ1. Gain nào do correspondence? | H1: Alignment chiếm phần lớn gain Penn so raw trong thiết lập GT-trained. | Mapping control và alignment-only downstream. |
| RQ2. Calibration còn hữu ích khi train/test cùng modality? | H2: Gain nhỏ hơn sau domain matching. | Ma trận 3×3 và CI cho chênh lệch. |
| RQ3. Khi nào smoothing làm hại? | H3: Nhóm fast-motion/foreshortening có tỷ lệ harm cao hơn. | Định nghĩa nhóm từ train/val, kiểm tra held-out; GT-filter control. |
| RQ4. Can thiệp chọn lọc có chuyển được? | H4: Policy đơn giản từ confidence/dynamics vượt always-filter và never-filter trên tập ngoài. | Khóa policy ở source, đánh giá target không tuning test. |

### Tối đa ba đóng góp cốt lõi đề xuất

Đây là cách viết mục tiêu contribution; chưa được sử dụng như kết quả đã hoàn thành.

**C1 — Controlled error attribution.** “Chúng tôi xây dựng đánh giá có kiểm soát để phân biệt tác động của correspondence, nhiễu tọa độ và lệch phân phối huấn luyện trong nhận diện từ pose 2D dự đoán.”

- Mới ở đâu: phân rã và insight thực nghiệm cụ thể, nếu review xác nhận chưa được giải quyết tương đương.
- Quan trọng: tránh quy nhầm gain do sửa mapping thành denoising.
- Khác prior work: nhấn mạnh can thiệp kiểm soát đồng thời ba yếu tố, thay vì chỉ báo final accuracy.
- Chứng minh: A1/A2/C1 trên nhiều điều kiện.
- Thiếu: alignment-only downstream, domain-matched Penn và audit quy ước độc lập.

**C2 — Conditions for beneficial calibration.** “Chúng tôi khảo sát điều kiện mà hậu xử lý thời gian cải thiện hoặc suy giảm nhận diện, và đánh giá liệu một quy tắc can thiệp đơn giản có thể bảo toàn thông tin chuyển động tốt hơn xử lý cố định hay không.”

- Mới ở đâu: quy luật có thể kiểm chứng hoặc policy có transfer; chưa thể nhận là mới ở hiện trạng.
- Quan trọng: tập trung action utility và giới hạn smoothing.
- Khác prior work: cần chỉ rõ bằng đối chiếu noise augmentation/refinement, không chỉ gọi policy motion-aware.
- Chứng minh: clean-motion control, stress tests và external validation.
- Thiếu: cơ chế, policy, baseline fair và bằng chứng held-out.

**C3 — Reproducible evaluation resource.** “Chúng tôi cung cấp protocol, dữ liệu dự đoán và công cụ audit để tái lập so sánh correspondence, refinement và train-domain dưới ngân sách tính toán giới hạn.”

- Mới ở đâu: tài nguyên và protocol có giá trị tái sử dụng; 5 seed tự nó không mới.
- Quan trọng: giúp kiểm tra claim và so sánh công bằng.
- Khác prior work: cần scope lỗi/annotation/schema cụ thể, không lặp lại toolbox đã có.
- Chứng minh: release schema, hashes, per-video predictions, tái sinh bảng từ môi trường sạch.
- Thiếu: đóng gói artifact, provenance thống nhất, điều kiện phân phối dữ liệu và người dùng ngoài thử tái lập.

### Proposed Paper Positioning

Tên bài đề xuất: **When Does 2D Pose Calibration Help Action Recognition? A Controlled Study of Joint Conventions, Temporal Filtering, and Training Domains.**

Nên đưa KineCalib thành một cấu hình can thiệp trong nghiên cứu, không để tên pipeline thay cho contribution. Giữ 64D như một control biểu diễn; KD và AQA chỉ ở phụ lục/exploratory nếu không có bằng chứng đủ mạnh.

### Research Narrative

**Problem:** pose dự đoán làm giảm action accuracy → **Existing Limitations:** lỗi correspondence, nhiễu và train-domain thường dễ bị trộn trong đánh giá → **Proposed Gap:** chưa có bằng chứng đủ rõ về từng yếu tố trong bối cảnh chi phí thấp đang xét → **Key Insight cần kiểm định:** đẹp hơn về hình học/thời gian không đảm bảo tốt hơn cho phân loại → **Method:** factorial intervention với baseline đơn giản, có thể thêm selective filtering nếu được chứng minh → **Evidence:** multi-seed, nhiều split/detector, fidelity–utility–latency và kết quả âm → **Contribution:** hiểu biết có điều kiện + protocol tái lập → **Practical Implications:** lựa chọn xử lý pose dựa trên tác vụ và điều kiện vận hành.

## 12. Final Area Chair Verdict

**Q1. Hiện có đóng góp khoa học thực sự không?** Có kết quả thực nghiệm ban đầu và tài nguyên kỹ thuật có giá trị; chưa xác lập contribution thuật toán mới. Claim empirical mạnh nhất vẫn thiếu thí nghiệm phân rã nguyên nhân.

**Q2. Đủ cho hội nghị quốc tế không?** Chưa đủ để bảo vệ full paper với bản thảo hiện tại. Một bài empirical phạm vi hẹp có triển vọng sau các sửa chữa P0/P1; workshop phù hợp cũng cần sửa tính hợp lệ và citation. Không thể gán xác suất nhận bài đáng tin cậy.

**Q3. Có thể reject vì gì?** Weak novelty; mapping confound; thiếu raw-trained baseline; unsupported SOTA/generalization/real-time/AQA; kết quả âm bị diễn giải sai; related work và số liệu không nhất quán.

**Q4. Thí nghiệm quan trọng nhất?** Alignment-only downstream, ma trận train–test và module ablation có motion-fidelity control; sau đó ba split J-HMDB và practical baselines.

**Q5. Chỉ thay ba điều?** (1) Chuẩn hóa toàn bộ claim/số liệu/citation theo artifact; (2) phân rã correspondence–denoising–training domain bằng đối chứng quyết định; (3) định vị thành empirical study về lợi ích và tác hại có điều kiện, xác nhận trên J-HMDB và baseline đơn giản.

**Q6. Hướng cân bằng novelty, ứng dụng và nguồn lực?** Controlled robustness study, rồi selective filtering đơn giản nếu dữ liệu chứng minh cần. Dùng cache và mô hình nhỏ hiện có; ưu tiên đo đúng hơn thêm kiến trúc lớn.

**Q7. Nếu thiếu novelty thuật toán?** Xây đóng góp empirical về failure conditions, benchmark lỗi có nhãn đáng tin, protocol audit tái lập hoặc systems study có Pareto và utility thực. Mỗi hướng vẫn cần kiến thức/bằng chứng mới; một pipeline chạy được không tự đủ.

**Q8. Tiếp tục hay đổi hướng?** **Tiếp tục nhưng điều chỉnh phạm vi và câu chuyện khoa học.** Giữ nền tảng pose→action và công cụ hiện có. Tạm bỏ AQA/coaching/rehabilitation khỏi claim chính, không đầu tư thêm KD/attention trước khi biết gain còn lại sau mapping và domain matching. Nếu gain biến mất, theo đuổi kết luận thực nghiệm trung thực hoặc toolkit audit thay vì cố bảo vệ KineCalib như thuật toán mới.

### Thông tin còn thiếu có ảnh hưởng trực tiếp đến kế hoạch

1. Nhân lực, thời hạn, thời gian GPU và ngân sách annotation thực tế; RTX 3050 chỉ được suy ra từ báo cáo/artifact.
2. Các tham số calibration và lựa chọn kiến trúc đã từng được quyết định dựa trên tập nào, kể cả giai đoạn trước V2.
3. Bối cảnh đích: nhận diện clip offline, nhận diện online, hay chấm kỹ thuật động tác. Đây là ba mục tiêu cần loại bằng chứng khác nhau.

Các thiếu hụt này ảnh hưởng phạm vi và lịch thực hiện; không làm thay đổi kết luận rằng bản thảo hiện tại cần sửa lớn trước nộp.

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
