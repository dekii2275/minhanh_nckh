# Bước 3 — Ablation PCHIP, Bone Constraint, One-Euro trên Penn Action validation

**Không có cấu hình calibration nào vượt Alignment Only về Top-1 trong run này.** A đạt 66,80%; H đạt 64,43%. Bone có point estimate âm trong cả bốn ngữ cảnh; PCHIP và One-Euro có tác động phụ thuộc ngữ cảnh. F (PCHIP + One-Euro) có Macro-F1 cao nhất 67,75%, nhưng Top-1 66,40%, thấp hơn A một video. Đây là chẩn đoán trên một checkpoint/validation, chưa phải kết luận tổng quát hoặc chứng minh mọi calibration vô ích.

| Cấu hình | Accuracy (%) | Macro-F1 (%) | Δ so A (điểm phần trăm) | Đúng / 253 |
|---|---:|---:|---:|---:|
| A – Alignment Only | 66.80 | 67.39 | +0.00 | 169 |
| B – + PCHIP | 66.40 | 67.16 | -0.40 | 168 |
| C – + Bone | 65.61 | 65.80 | -1.19 | 166 |
| D – + One-Euro | 65.22 | 65.84 | -1.58 | 165 |
| E – PCHIP + Bone | 65.22 | 65.46 | -1.58 | 165 |
| F – PCHIP + One-Euro | 66.40 | 67.75 | -0.40 | 168 |
| G – Bone + One-Euro | 64.43 | 64.81 | -2.37 | 163 |
| H – Full KineCalib | 64.43 | 64.79 | -2.37 | 163 |

Tác động riêng của từng thành phần, so với A; Full là tác động gộp:

| Thành phần | Δ Accuracy (điểm phần trăm) | Paired bootstrap 95% CI | Sai→đúng | Đúng→sai |
|---|---:|---:|---:|---:|
| PCHIP (B−A) | -0.40 | [-1.98; +1.19] | 2 | 3 |
| Bone Constraint (C−A) | -1.19 | [-2.77; +0.00] | 0 | 3 |
| One-Euro (D−A) | -1.58 | [-3.95; +0.79] | 3 | 7 |
| Full KineCalib (H−A) | -2.37 | [-5.14; +0.40] | 4 | 10 |

CI cho tất cả cấu hình so với A:

| So sánh | Δ Accuracy (điểm phần trăm) | 95% CI | Sai→đúng | Đúng→sai |
|---|---:|---:|---:|---:|
| B−A | -0.40 | [-1.98; +1.19] | 2 | 3 |
| C−A | -1.19 | [-2.77; +0.00] | 0 | 3 |
| D−A | -1.58 | [-3.95; +0.79] | 3 | 7 |
| E−A | -1.58 | [-3.56; +0.00] | 1 | 5 |
| F−A | -0.40 | [-3.16; +2.37] | 7 | 8 |
| G−A | -2.37 | [-4.74; +0.00] | 2 | 8 |
| H−A | -2.37 | [-5.14; +0.40] | 4 | 10 |

Tác động khi thêm một thành phần vào mỗi ngữ cảnh; đây là 12 cạnh của thiết kế đầy đủ 2³:

| Thành phần được thêm | So sánh | Δ Accuracy (điểm phần trăm) | 95% CI | Sai→đúng | Đúng→sai |
|---|---|---:|---:|---:|---:|
| PCHIP | B-A | -0.40 | [-1.98; +1.19] | 2 | 3 |
| PCHIP | E-C | -0.40 | [-1.58; +0.79] | 1 | 2 |
| PCHIP | F-D | +1.19 | [-0.79; +3.16] | 5 | 2 |
| PCHIP | H-G | +0.00 | [-1.58; +1.58] | 2 | 2 |
| Bone | C-A | -1.19 | [-2.77; +0.00] | 0 | 3 |
| Bone | E-B | -1.19 | [-2.77; +0.40] | 1 | 4 |
| Bone | G-D | -0.79 | [-2.37; +0.79] | 1 | 3 |
| Bone | H-F | -1.98 | [-3.95; -0.40] | 0 | 5 |
| OneEuro | D-A | -1.58 | [-3.95; +0.79] | 3 | 7 |
| OneEuro | F-B | +0.00 | [-2.77; +2.77] | 6 | 6 |
| OneEuro | G-C | -1.19 | [-3.16; +0.79] | 2 | 5 |
| OneEuro | H-E | -0.79 | [-2.77; +1.19] | 3 | 5 |

**Q1 — Thành phần nào cải thiện accuracy?** Không thành phần đơn lẻ B/C/D và không tổ hợp E/F/G/H nào có Top-1 cao hơn A trên mẫu này. Không có so sánh chính nào có CI nằm hoàn toàn trên 0. Không thể suy từ đó rằng các phương án tương đương hoặc không thể có lợi trên checkpoint/dataset khác. F có Macro-F1 cao hơn A **+0,36 điểm**, nhưng đây là point estimate, chưa có kiểm định/CI Macro-F1 hay dữ liệu độc lập xác nhận.

**Q2 — Thành phần nào có xu hướng giảm accuracy?** Bone: C−A −1,19; E−B −1,19; G−D −0,79; H−F −1,98 điểm. C−A không cứu video nào và làm sai 3 video. H−F làm sai thêm 5, không cứu video nào; CI [−3,95; −0,40] nằm dưới 0 trong phép so sánh chưa hiệu chỉnh đa kiểm định. Đây là tín hiệu ưu tiên xem lại Bone, không phải chứng minh Bone gây hại phổ quát. One-Euro đơn lẻ: D−A −1,58, cứu 3/làm sai 7, CI chứa 0. PCHIP đơn lẻ: B−A −0,40, cứu 2/làm sai 3, CI chứa 0.

**Q3 — Có thành phần chỉ có ích khi kết hợp?** Có gợi ý về tương tác PCHIP × One-Euro, chưa có bằng chứng chắc chắn. Thêm PCHIP vào One-Euro (F−D) tăng +1,19 điểm, cứu 5/làm sai 2, nhưng CI [−0,79; +3,16] chứa 0. Thêm One-Euro vào PCHIP (F−B) giữ nguyên Top-1, cứu 6/làm sai 6; vì vậy cùng accuracy không có nghĩa cùng dự đoán. F vẫn thấp hơn A về Top-1.

Tương tác trên thang accuracy khi Bone tắt là `(F−D)−(B−A)` = +1,58 điểm, CI [0,00; +3,56], còn chạm 0. Tương tác không đồng nghĩa lợi ích tuyệt đối. Khi Bone bật, tương tác PCHIP × One-Euro chỉ +0,40 [0,00; +1,19]. Tương tác PCHIP × Bone khi One-Euro tắt là 0,00 [−1,19; +1,19]. Các tương tác hai chiều còn lại và tương tác ba chiều được lưu đầy đủ trong CSV; không chọn riêng tương tác thuận lợi để kết luận.

**Q4 — Có thể đơn giản hóa không?** Với mục tiêu Top-1 của checkpoint và validation hiện tại, A là baseline đơn giản đáng giữ để đối chứng: 169/253 đúng, so với H 163/253. Chưa có bằng chứng cần thêm bất kỳ module nào để nâng Top-1. Không thay đổi implementation production ở bước này. F chỉ là ứng viên nếu nghiên cứu tiếp ưu tiên Macro-F1; việc chọn F hay A sau khi xem validation cần được khóa trước đánh giá độc lập, không gọi là phương pháp thắng đã xác nhận. Fixed Swap của Bước 2 không phải A: A vẫn dùng đúng các quyết định heuristic production đã lưu.

**Q5 — Gợi ý hướng nghiên cứu có căn cứ:**

- Tách sửa correspondence và chọn/tracking đúng người khỏi hiệu chỉnh hình học. Tất cả cấu hình giữ cố định cùng YOLO và alignment; các module hiện tại không giải quyết nhầm người đã phát hiện ở Bước 1.
- Nghiên cứu sự đánh đổi giữa khôi phục dữ liệu thiếu, fidelity quỹ đạo và utility nhận diện. PCHIP thực sự thay đổi đầu vào, nhưng không tự động tăng Top-1; cần kiểm tra tại các đoạn gap, độ bất định và biên chuyển động, thay vì chỉ chứng minh skeleton mượt hơn.
- Xem lại giả định độ dài xương 2D theo median toàn clip trước foreshortening/xoay người. Tín hiệu âm của Bone ở cả bốn ngữ cảnh tạo giả thuyết có thể kiểm chứng bằng fidelity và phân tích view; run này chưa chứng minh đó là cơ chế gây lỗi.
- Với smoothing, kiểm tra độ trễ/biên độ và khả năng giữ thời điểm chuyển động. So sánh trên checkpoint GT-trained cũng có thể phản ánh lệch phân phối feature, chưa đủ kết luận One-Euro làm mất thông tin vận động. Có thể nghiên cứu calibration có điều kiện theo độ tin cậy/chuyển động, nhưng chưa triển khai hoặc tuning trong Bước 3.
- Scientific contribution khả dĩ là một nghiên cứu có đối chứng về khi nào calibration giúp/hại, với benchmark đã khóa correspondence, paired per-video evidence và xác nhận độc lập. Tám cấu hình trên cùng validation chưa đủ để tuyên bố novelty thuật toán, SOTA hay generalization.

Chẩn đoán đầu vào, không dùng GT joints:

| Cấu hình | Video đổi tọa độ so A | Video đổi feature so A | Giá trị xy NaN còn lại | Giá trị xy NaN được lấp | Frame feature toàn 0 |
|---|---:|---:|---:|---:|---:|
| A | 0 | 0 | 4416 | 0 | 184 |
| B | 129 | 129 | 2494 | 1922 | 99 |
| C | 248 | 248 | 4416 | 0 | 184 |
| D | 253 | 253 | 4416 | 0 | 184 |
| E | 248 | 248 | 2494 | 1922 | 99 |
| F | 253 | 253 | 2494 | 1922 | 99 |
| G | 253 | 253 | 4416 | 0 | 184 |
| H | 253 | 253 | 2494 | 1922 | 99 |

Các số trên đếm **giá trị tọa độ x/y**, không phải số khớp hoặc độ chính xác pose. PCHIP thay đổi 129/253 video, lấp 1.922 giá trị NaN trong 4.416, làm giảm frame feature toàn 0 từ 184 xuống 99. Bone thay đổi tọa độ/feature ở 248 video, One-Euro ở 253 video. Vì vậy kết quả không xuất phát từ module hoàn toàn không chạy. Không thêm tọa độ NaN mới so A ở bất kỳ cấu hình nào. Đây là bằng chứng về hoạt động của code, không chứng minh các tọa độ mới đúng giải phẫu.

Code PCHIP hiện hành chỉ nội suy khi có ít nhất hai điểm hợp lệ, có điểm trước/sau và khoảng cách hai chỉ số không vượt `max_gap=15`; không ngoại suy biên. `invalid_mask` có xét confidence<0.35, nhưng code không tự đặt mọi tọa độ confidence thấp thành NaN: giá trị không được lấp có thể giữ nguyên. Không sửa hành vi này trong ablation. Tên PCHIP trong báo cáo chỉ implementation hiện có, không đại diện mọi thuật toán khôi phục khớp thiếu.

Phân tích lớp chỉ mang tính mô tả: F cải thiện số đúng của bowl từ 2 lên 4 và bench_press từ 4 lên 5, nhưng giảm jumping_jacks từ 10 xuống 8, baseball_swing từ 20 xuống 19, squat từ 9 xuống 8 và tennis_serve từ 15 xuống 14. Các số là số video đúng, không phải accuracy lớp hay bằng chứng cơ chế. Xem bảng từng lớp để đánh giá đầy đủ.

Danh sách video thay đổi dự đoán giữa ít nhất hai cấu hình (bao gồm cả đổi từ một nhãn sai sang nhãn sai khác):

`0041, 0092, 0127, 0137, 0165, 0299, 0351, 0359, 0447, 0455, 0490, 0509, 0517, 0536, 0542, 0576, 0581, 0584, 0586, 0600, 0605, 0607, 0628, 0681, 0683, 0686, 0777, 1089, 1146, 1424, 1425, 1625, 1685, 1738, 1742, 1746, 1760, 2125, 2156, 2211, 2237, 2243`

Có **42 video đổi nhãn dự đoán**, trong đó **18 video đổi trạng thái đúng/sai** ở ít nhất một cặp cấu hình. Đây là hai khái niệm khác nhau; không dùng 42 làm số video được cải thiện. changed_videos.csv (`outputs/calibration_step3/changed_videos.csv`, chưa công bố trên nhánh này) chứa đủ 8 dự đoán cho các ID này. changed_prediction_events.csv (`outputs/calibration_step3/changed_prediction_events.csv`, chưa công bố trên nhánh này) ghi từng sự kiện thay đổi giữa 28 cặp, gồm nhãn trước/sau và `improved`, `harmed`, `both_wrong`.

Ví dụ thay đổi trạng thái đúng/sai so A (danh sách đầy đủ, không chỉ ví dụ thuận lợi):

| Cấu hình | Sai→đúng: ID | Đúng→sai: ID |
|---|---|---|
| B | 0447, 0681 | 0542, 1746, 1760 |
| C | — | 0165, 0777, 1625 |
| D | 0041, 1425, 2237 | 0165, 0299, 1089, 1146, 1760, 2156, 2211 |
| E | 0447 | 0165, 0542, 0777, 1625, 1746 |
| F | 0041, 0447, 0490, 0681, 0686, 1425, 2237 | 0165, 0299, 0542, 1089, 1146, 1746, 2156, 2211 |
| G | 1425, 2237 | 0165, 0299, 0777, 1089, 1146, 1625, 2156, 2211 |
| H | 0447, 0490, 1425, 2237 | 0165, 0299, 0542, 0777, 1089, 1146, 1625, 1746, 2156, 2211 |

Điều kiện thực nghiệm và kiểm chứng:

- **253 video / 17.747 frame**, toàn bộ validation từ Bước 2, giao rỗng với 1.005 ID train của checkpoint. Không loại clip khó/thiếu detection; không đọc official test hoặc chạy YOLO lại. Script dùng nhãn hành động đã lưu để đánh giá/stratify bootstrap; không đọc GT joints để biến đổi input hay chọn swap.
- Checkpoint GT-trained **BiLSTM 24D Protocol v2 seed0** giữ nguyên. Hash checkpoint, code model, feature, calibration và tất cả 253 file raw YOLO được kiểm tra trước/sau run. Batch classifier=32, cùng toàn bộ frame, thứ tự video, normalize/midhip/shoulder fallback/NaN→0 và float32 như Bước 2. Model `eval()`, không train hoặc tuning.
- Bật/tắt đúng ba stage, giữ thứ tự **PCHIP → Bone → One-Euro**. `FullCalibrationPipeline(lr_mode=None)` cung cấp các stage mặc định và tắt stage heuristic. Sau các stage được bật, áp sáu cặp swap theo **cờ Bước 2 đã lưu**; không tính median/heuristic cho bất kỳ cấu hình nào. 168 clip swap, 85 clip Identity.
- Calibration giữ nguyên confidence=0.35, gap=15, bone ratio=0.40–1.20, 30Hz, min cutoff=1.0, beta=0.05, d cutoff=1.0. Không tuning theo kết quả mới.
- A tái tạo C3 và H tái tạo C4 **cả 253 nhãn dự đoán và toàn bộ posterior 15 lớp**, max absolute difference A=0.0, H=0.0. Lần kiểm tra ban đầu ở 4 luồng CPU đã khớp nhãn nhưng lệch posterior tối đa khoảng 1,76×10⁻⁶; đã xác minh YOLO Bước 2 gọi `select_device('cpu')` và tự đổi PyTorch sang 8 luồng. Run báo cáo khớp lại 8 luồng, sai khác posterior bằng 0; không đổi checkpoint/thuật toán/tham số và không nới ngưỡng kiểm tra để vượt qua.
- Hai test kiểm soát ép heuristic ném lỗi nếu bị gọi: tất cả 8 cấu hình đều chạy qua, giữ đúng cờ swap cho cả True/False, và không sửa input raw. Không đánh đồng việc tái tạo số liệu A/H với việc heuristic đã đúng giải phẫu.
- CI: **paired bootstrap theo video, phân tầng 15 lớp**, giữ số video mỗi lớp, 20.000 lần, seed=20261029, percentile 95%, cùng chỉ số lấy mẫu cho mọi cấu hình và contrast. Mỗi video là một đơn vị, không bootstrap frame. Δ tính theo điểm phần trăm; 1 video = 100/253 ≈0,3953 điểm. `(sai→đúng − đúng→sai)/253×100` khớp mọi chênh lệch cặp.
- Các CI chưa hiệu chỉnh nhiều phép so sánh; CI chạm 0 được coi là còn chứa 0. Tương tác là difference-of-differences trên accuracy, không phải bằng chứng hiệp lực cơ học. Bootstrap có điều kiện trên checkpoint và thành phần lớp này, không gồm biến thiên train/seed hay phụ thuộc giữa clip cùng nguồn. Checkpoint đã được chọn bằng validation GT này; phân tích nhiều cấu hình trên cùng validation không tạo ra test độc lập.

Artifact:

- per_video_predictions.csv (`outputs/calibration_step3/per_video_predictions.csv`, chưa công bố trên nhánh này): 253 ID, đủ 8 dự đoán/nhãn hành động, đúng/sai, outcome so A, cờ alignment cố định, số frame và thông tin detection Bước 2.
- configuration_metrics.csv (`outputs/calibration_step3/configuration_metrics.csv`, chưa công bố trên nhánh này), component_marginal_effects.csv (`outputs/calibration_step3/component_marginal_effects.csv`, chưa công bố trên nhánh này), interaction_contrasts.csv (`outputs/calibration_step3/interaction_contrasts.csv`, chưa công bố trên nhánh này), all_pairwise_comparisons.csv (`outputs/calibration_step3/all_pairwise_comparisons.csv`, chưa công bố trên nhánh này): metric, CI, tác động riêng/kết hợp, toàn bộ 28 cặp.
- per_class_accuracy.csv (`outputs/calibration_step3/per_class_accuracy.csv`, chưa công bố trên nhánh này), transformation_diagnostics.csv (`outputs/calibration_step3/transformation_diagnostics.csv`, chưa công bố trên nhánh này), probabilities.csv (`outputs/calibration_step3/probabilities.csv`, chưa công bố trên nhánh này), reproduction_checks.json (`outputs/calibration_step3/reproduction_checks.json`, chưa công bố trên nhánh này), provenance.json (`outputs/calibration_step3/provenance.json`, chưa công bố trên nhánh này), verification.json (`outputs/calibration_step3/verification.json`, chưa công bố trên nhánh này).
- Code evaluate_calibration_step3.py (`outputs/research_snapshot/src/evaluate_calibration_step3.py`, chưa công bố trên nhánh này), report_calibration_step3.py (`outputs/research_snapshot/src/report_calibration_step3.py`, chưa công bố trên nhánh này), verify_calibration_step3.py (`outputs/research_snapshot/src/verify_calibration_step3.py`, chưa công bố trên nhánh này), test_calibration_step3.py (`outputs/research_snapshot/tests/test_calibration_step3.py`, chưa công bố trên nhánh này).

Tái lập:

```bash
MPLCONFIGDIR=/tmp/alignment-step3-mpl /home/dekii2275/miniconda3/envs/penn_pose/bin/python src/evaluate_calibration_step3.py
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/report_calibration_step3.py
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/verify_calibration_step3.py
```

Dừng ở Bước 3. Chưa loại bỏ module khỏi production, chưa cải tiến thuật toán, chưa train và chưa đánh giá official test.

Bản cục bộ: code thực nghiệm ở trên là snapshot giữ nguyên byte, không sửa production. Snapshot, CSV, hình, dataset, checkpoint và cache raw npz chưa được đưa lên nhánh tài liệu; xem [phạm vi công bố và khôi phục môi trường](REPRODUCIBILITY.md). Posterior được xuất CSV tại máy để kiểm tra.

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
