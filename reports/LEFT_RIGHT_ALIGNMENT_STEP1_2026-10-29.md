# Bước 1 — Audit Left/Right Penn Action, 29/10/2026

**Kết quả: heuristic hiện tại không phải quy tắc correspondence đáng tin cậy ở cấp clip.** Có phản ví dụ quay lưng cần swap nhưng heuristic giữ nguyên; có clip cần I và S tại các thời điểm khác nhau. Chưa sửa alignment, chưa train, chưa chạy downstream/ablation hay tuning.

| Tiêu chí | Kết quả |
|---|---:|
| Tổng số video kiểm tra | 30 (2149 frame chạy YOLO) |
| Số video Identity đúng nhất quán trên bằng chứng quan sát được | 0 |
| Số video Fixed Swap đúng nhất quán trên bằng chứng quan sát được | 15 |
| Số video Current Heuristic đúng nhất quán trên bằng chứng quan sát được | 11 |
| Số video không thể kết luận ở cấp clip | 11 |
| Số video có hiện tượng đổi hướng/xoay thân quan sát được | 11 |
| Có I và S đúng ở thời điểm khác nhau | 4: 2 có đoạn nhiều frame, 2 đảo ngắn/cục bộ |
| Alignment thất bại rõ, ngoài nhóm chưa kết luận | 4 clip heuristic giữ I trong nhóm cần S; thêm 4 clip có đảo cục bộ/đổi correspondence |

“Đúng” ở đây là **đúng correspondence trên các khớp phân biệt được qua mẫu frame và chẩn đoán**, không phải mọi khớp đều chính xác về vị trí hoặc mọi frame đã được xem thủ công. Các clip chỉ thấy tay không chứng minh correspondence chân. Số 0 của Identity không có nghĩa Identity luôn sai ở mọi frame. 11 chưa kết luận không tính đúng hoặc sai cho bất kỳ phương án nào. 15 S + 4 mixed/cục bộ + 11 chưa kết luận = 30; 11 heuristic đúng là tập con của 15 S.

Cách làm: chọn ngẫu nhiên phân tầng 2 clip/hành động, seed `20261029`, từ 1.258 clip có `train==1`; clip thứ hai ưu tiên pose khác, nếu lớp chỉ có một pose thì lấy ID khác. Không dùng clip test (`train==-1`) và không đọc dự đoán test. Đây là mẫu từ official train, không chia lại train/validation; không dùng kết quả này như validation downstream. Pose metadata: {'front': np.int64(11), 'back': np.int64(7), 'right': np.int64(6), 'left': np.int64(6)}. Không diễn giải pose metadata là hướng của mọi frame.

Chạy YOLO11n-Pose có sẵn, CPU, imgsz=640, box confidence=0.25; chọn người có box confidence cao nhất như code hiện hành, không chọn theo GT. Chạy toàn bộ frame để lấy đúng median clip. Xem 5 frame cách đều/clip và thêm frame có bằng chứng I/S mạnh nhất. Đối chiếu thêm hình phóng to tại các trường hợp bất đồng. Đây là review bằng hình của trợ lý, chưa có hai annotator độc lập hay kiểm tra đồng thuận.

Chẩn đoán GT chỉ hỗ trợ tìm frame cần xem: mỗi cặp phải có cả hai GT visible, hai YOLO confidence ≥0.5, tọa độ hợp lệ, độ tách hai phía ≥max(5px, 0.15×torso). So trung bình khoảng cách Euclidean hai đầu của cặp dưới I/S, chỉ vote khi chênh lệch >max(3px, 0.15×độ tách GT). Frame rõ khi ≥2 cặp cùng vote và không có vote ngược; còn lại `?`. Các ngưỡng phục vụ sàng lọc audit, không tuning calibration, không phải định nghĩa chân lý. Vote trên nhầm người vẫn có thể trông thuyết phục nên phải kiểm tra RGB và giữ U. Không biến oracle theo GT thành alignment triển khai.

Đổi hướng được đánh dấu bằng RGB khi có xoay thân rõ giữa các frame, không tính riêng gập/ngồi/nhảy. 11 ID: 0099, 0137, 0219, 0287, 0604, 0801, 0864, 2062, 2104, 2192, 2203. Đây là số quan sát sơ bộ, không phải phân loại góc quay tự động hay danh sách đầy đủ mọi chuyển hướng giữa frame lấy mẫu.

Các phát hiện cần review:

- **2062**: quay lưng/chéo lưng, S khớp GT ở f1/32/47/63; heuristic I (`median dx=-17.08px`) gây nối sai hai phía. Chẩn đoán 63/63 frame ủng hộ S. **2192**, **0801** cũng cần S nhưng heuristic giữ I. Vì vậy không thể dùng “quay lưng ⇒ không swap” để giải thích convention.
- **1897**: S khớp tay GT, nhưng vai bị crop/độ tin cậy thấp. Median vai raw `+170.88px`, sau các stage 1–3 là `-25.23px`, khiến heuristic production chọn I. Đây là lỗi quyết định phụ thuộc đầu vào xử lý trước; không quy hết cho góc nhìn.
- **0864**: S ở f27/29, I trên đoạn f41–43 và f45–47. Median raw `-0.23px`, production `+0.89px`: dấu không ổn định. **2104**: I ở f21–24 và f37–38, S ở các đoạn khác (ví dụ f61). Hai clip này không thể được căn đúng xuyên suốt bằng một permutation duy nhất.
- **0757**: S 219 frame chẩn đoán nhưng I tại f176,183, cả 6 cặp. Đây là đảo ngắn của dự đoán/correspondence quan sát được, không có bằng chứng người quay từ trước ra sau. **2203**: I ngắn ở f29 trên 2 cặp phân biệt được, S ở phần lớn đoạn khác; không kết luận toàn thân đảo nhãn. Hai ca ngắn này tách riêng khỏi đoạn mixed nhiều frame.
- **0604**: có gợi ý I tại f36 và S ở f15, nhưng f41 chọn nhầm người; giữ U, không nâng thành mixed đã xác nhận. Các vote I/S của **0137** cũng không chứng minh đổi correspondence vì target sai.
- **0137, 0433, 0443, 1316, 1597, 1622, 1787**: có sai target/đổi target trên RGB. Không thể sửa bằng đổi nhãn khớp. **1484, 1377, 1691** thiếu bằng chứng vì dropout, nghiêng và che khuất. Cùng 0604 tạo nhóm 11 U.

Code hiện tại ở src/skeleton_calibrator.py (`outputs/research_snapshot/src/skeleton_calibrator.py`, chưa công bố trên nhánh này), bản chụp nguyên class tại current_alignment_snapshot.py (`outputs/alignment_step1/current_alignment_snapshot.py`, chưa công bố trên nhánh này). Phần quyết định chính:

```python
valid_mask = ~df_out[lx_col].isna() & ~df_out[rx_col].isna()
if valid_mask.sum() < 2:
    return df_out
delta_x = df_out.loc[valid_mask, lx_col] - df_out.loc[valid_mask, rx_col]
median_delta = float(np.median(delta_x))
if median_delta > 0.0:
    for left_kpt, right_kpt in paired_joints:
        for suffix in ["x", "y", "conf"]:
            col_l = f"pred_{left_kpt}_{suffix}"
            col_r = f"pred_{right_kpt}_{suffix}"
            if col_l in df_out.columns and col_r in df_out.columns:
                tmp = df_out[col_l].values.copy()
                df_out[col_l] = df_out[col_r].values.copy()
                df_out[col_r] = tmp
```

Sáu cặp: shoulder, elbow, wrist, hip, knee, ankle. Clip ≤2 frame hoặc thiếu cột vai trả nguyên; chỉ dùng tính hợp lệ NaN, không confidence hay margin. `mode` được lưu nhưng không kiểm soát nhánh trong class. Trong `FullCalibrationPipeline`, alignment là stage 4 sau interpolation → bone constraint → smoothing. Audit dùng quyết định đúng tại đầu vào stage 4 làm **Current Heuristic**, rồi áp permutation đó lên raw để so hình công bằng với Identity/Fixed Swap. Cột raw heuristic được lưu riêng; raw và production khác quyết định tại **0864, 1597, 1897**. Không diễn giải khác biệt này thành ablation accuracy.

Hình minh họa (xanh lam=L, tím=R; GT vòng tròn/nét mảnh, YOLO dấu ×/nét đậm; số 0–5 tương ứng vai, khuỷu, cổ tay, hông, gối, cổ chân; YOLO chỉ vẽ confidence ≥0.5):

Hình cục bộ: 2062: quay lưng vẫn cần Fixed Swap (`outputs/alignment_step1/figures/evidence_2062.jpg`, chưa công bố trên nhánh này)

Hình cục bộ: 0864: bốn cột GT, Identity, Fixed Swap, Current Heuristic (`outputs/alignment_step1/figures/0864.jpg`, chưa công bố trên nhánh này)

Hình cục bộ: 0757: đảo ngắn tại f176 và f183 (`outputs/alignment_step1/figures/evidence_0757.jpg`, chưa công bố trên nhánh này)

Bộ 30 hình bốn phương án (`outputs/alignment_step1/figures`, chưa công bố trên nhánh này) có 5–7 frame/clip, gồm cột Current Heuristic; `evidence_*.jpg` là hình phóng to 3 cột. GT luôn là nhãn Penn nguyên bản, không swap GT. Hình heuristic chỉ thể hiện alignment, không thêm các hiệu chỉnh tọa độ khác.

Bảng từng video: I=Identity, S=Fixed Swap, M=thay đổi theo thời gian/cục bộ, U=chưa kết luận. “Đạt” chỉ nghĩa nhất quán theo phạm vi quan sát nêu trên; M không được tính đạt toàn clip. Cột I/S/? là **số frame chẩn đoán tự động**, chưa loại nhầm target, không phải số frame chấm tay.

| ID | Hành động | Pose | I/S/? | Heuristic | Review | Ghi chú |
|---|---|---|---|---|---|---|
| 0099 (`outputs/alignment_step1/figures/0099.jpg`, chưa công bố trên nhánh này) | baseball_pitch | front | 0/84/27 | S | S; H đạt | S rõ ở người mục tiêu; xoay thân khi ném. |
| 0137 (`outputs/alignment_step1/figures/0137.jpg`, chưa công bố trên nhánh này) | baseball_pitch | back | 6/6/64 | I | U | Chọn người STAFF ở tiền cảnh; vote I/S không có giá trị cho người GT. |
| 0219 (`outputs/alignment_step1/figures/0219.jpg`, chưa công bố trên nhánh này) | baseball_swing | right | 0/30/5 | S | S; H đạt | S; xoay thân khi đánh bóng. |
| 0287 (`outputs/alignment_step1/figures/0287.jpg`, chưa công bố trên nhánh này) | baseball_swing | back | 0/27/10 | S | S; H đạt | S dù nhãn pose=back; median vai dương nên heuristic tình cờ chọn S. |
| 0433 (`outputs/alignment_step1/figures/0433.jpg`, chưa công bố trên nhánh này) | bench_press | front | 0/7/34 | S | U | Có frame chọn người hỗ trợ đứng sau thay vì người nằm ghế. |
| 0443 (`outputs/alignment_step1/figures/0443.jpg`, chưa công bố trên nhánh này) | bench_press | left | 0/44/80 | S | U | Chọn người hỗ trợ đứng sau ở các frame kiểm tra. |
| 0604 (`outputs/alignment_step1/figures/0604.jpg`, chưa công bố trên nhánh này) | bowl | back | 1/19/35 | I | U | S ở f15; I gợi ý tại f36; f41 chọn người khác, không kết luận clip. Không xếp mixed đã xác nhận. |
| 0543 (`outputs/alignment_step1/figures/0543.jpg`, chưa công bố trên nhánh này) | bowl | front | 0/57/2 | S | S; H đạt | S rõ khi đi về phía camera. |
| 0757 (`outputs/alignment_step1/figures/0757.jpg`, chưa công bố trên nhánh này) | clean_and_jerk | front | 2/219/0 | S | M; không đạt clip | S 219 frame chẩn đoán; I tại f176 và f183 (6/6 cặp); hai lỗi ngắn, không có xoay hướng rõ. |
| 0775 (`outputs/alignment_step1/figures/0775.jpg`, chưa công bố trên nhánh này) | clean_and_jerk | right | 0/139/110 | S | S; H đạt | S trên các cặp phân biệt được; nhiều người nền nhưng target đúng ở frame kiểm tra. |
| 0864 (`outputs/alignment_step1/figures/0864.jpg`, chưa công bố trên nhánh này) | golf_swing | back | 6/24/23 | S | M; không đạt clip | S f27/29; I f41–43,45–47; xoay thân. Raw chọn I, đầu vào production chọn S. |
| 0801 (`outputs/alignment_step1/figures/0801.jpg`, chưa công bố trên nhánh này) | golf_swing | front | 0/11/45 | I | S; H sai | S trên khớp nhìn thấy; heuristic giữ I vì median vai âm, gần nghiêng. |
| 0996 (`outputs/alignment_step1/figures/0996.jpg`, chưa công bố trên nhánh này) | jump_rope | front | 0/38/0 | S | S; H đạt | S rõ. |
| 0976 (`outputs/alignment_step1/figures/0976.jpg`, chưa công bố trên nhánh này) | jump_rope | right | 0/31/12 | S | S; H đạt | S trên các cặp phân biệt được. |
| 1053 (`outputs/alignment_step1/figures/1053.jpg`, chưa công bố trên nhánh này) | jumping_jacks | front | 0/22/0 | S | S; H đạt | S rõ. |
| 1112 (`outputs/alignment_step1/figures/1112.jpg`, chưa công bố trên nhánh này) | jumping_jacks | left | 0/28/0 | S | S; H đạt | S rõ dù pose metadata=left. |
| 1316 (`outputs/alignment_step1/figures/1316.jpg`, chưa công bố trên nhánh này) | pullup | back | 0/0/54 | S | U | YOLO chọn người đứng bên trái thay vì người kéo xà. |
| 1170 (`outputs/alignment_step1/figures/1170.jpg`, chưa công bố trên nhánh này) | pullup | front | 0/79/0 | S | S; H đạt | S rõ. |
| 1484 (`outputs/alignment_step1/figures/1484.jpg`, chưa công bố trên nhánh này) | pushup | right | 0/0/31 | S | U | 25/31 frame không phát hiện; người nhỏ, nghiêng và chồng khớp. |
| 1377 (`outputs/alignment_step1/figures/1377.jpg`, chưa công bố trên nhánh này) | pushup | left | 0/0/73 | S | U | 31/73 frame không phát hiện; một bên bị che/chồng, thiếu bằng chứng hai phía. |
| 1597 (`outputs/alignment_step1/figures/1597.jpg`, chưa công bố trên nhánh này) | situp | left | 0/35/59 | I | U | Đổi target: có frame chọn người tập, có frame chọn người đứng; không kết luận toàn clip. |
| 1622 (`outputs/alignment_step1/figures/1622.jpg`, chưa công bố trên nhánh này) | situp | right | 0/0/51 | S | U | Chọn người giữ chân thay vì người GT nằm tập. |
| 1787 (`outputs/alignment_step1/figures/1787.jpg`, chưa công bố trên nhánh này) | squat | front | 0/44/41 | S | U | Đổi target giữa người tập và huấn luyện viên. |
| 1691 (`outputs/alignment_step1/figures/1691.jpg`, chưa công bố trên nhánh này) | squat | left | 0/0/58 | S | U | Nghiêng, che khuất, hai phía chồng nhau; không phân biệt chắc I/S. |
| 1914 (`outputs/alignment_step1/figures/1914.jpg`, chưa công bố trên nhánh này) | strum_guitar | front | 0/45/0 | S | S; H đạt | S rõ ở tay; không suy luận cho chân không quan sát được. |
| 1897 (`outputs/alignment_step1/figures/1897.jpg`, chưa công bố trên nhánh này) | strum_guitar | front | 0/34/14 | I | S; H sai | S rõ ở tay; crop mất vai, production median âm nên heuristic I; raw median dương. |
| 2062 (`outputs/alignment_step1/figures/2062.jpg`, chưa công bố trên nhánh này) | tennis_forehand | back | 0/63/0 | I | S; H sai | S rõ cả f1,32,47,63; lưng/chéo lưng, heuristic I sai. |
| 2104 (`outputs/alignment_step1/figures/2104.jpg`, chưa công bố trên nhánh này) | tennis_forehand | left | 6/29/26 | S | M; không đạt clip | I tại f21–24,37–38; S ở đoạn khác, xem f61. Có xoay thân. |
| 2203 (`outputs/alignment_step1/figures/2203.jpg`, chưa công bố trên nhánh này) | tennis_serve | right | 1/59/31 | S | M; không đạt clip | S chiếm ưu thế; I ngắn tại f29 (2 cặp), các cặp khác mơ hồ; không gọi là đổi toàn thân chắc chắn. |
| 2192 (`outputs/alignment_step1/figures/2192.jpg`, chưa công bố trên nhánh này) | tennis_serve | back | 0/38/30 | I | S; H sai | S trên cặp thấy rõ; lưng/nghiêng, heuristic I sai. |

Artifact tái lập: sample.json (`outputs/alignment_step1/sample.json`, chưa công bố trên nhánh này), video_review.csv (`outputs/alignment_step1/video_review.csv`, chưa công bố trên nhánh này), frame_diagnostics.csv (`outputs/alignment_step1/frame_diagnostics.csv`, chưa công bố trên nhánh này), provenance.json (`outputs/alignment_step1/provenance.json`, chưa công bố trên nhánh này) (hash model/code, phiên bản thư viện), dự đoán raw từng clip trong `outputs/alignment_step1/*.npz`. Script audit_left_right_step1.py (`outputs/research_snapshot/src/audit_left_right_step1.py`, chưa công bố trên nhánh này) tái lập lấy mẫu, inference, chẩn đoán và các contact sheet; `video_review.csv` là quyết định review thủ công tách riêng, không được script tự suy ra từ vote.

```bash
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/audit_left_right_step1.py
```

**Kết luận giới hạn cho Bước 1:** có mismatch correspondence rõ và heuristic bỏ sót/tạo lỗi tại các đoạn có bằng chứng I, nhưng chưa định lượng phần đóng góp của nó vào sai số toàn dataset hoặc downstream accuracy. Mẫu này ủng hộ Fixed Swap làm baseline kiểm soát cần xem xét, không chứng minh fixed swap đúng mọi clip hay Penn dùng một quy ước viewer-centric toàn cục. Cần review các ngoại lệ và lỗi chọn người trước khi coi alignment đã hợp lệ cho downstream. Thuật toán được giữ nguyên. Bước 2–3 đã hoàn tất sau audit này; xem [Bước 2](LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md) và [Bước 3](CALIBRATION_ABLATION_STEP3_2026-10-29.md). 30 clip là pilot tìm lỗi, chưa đủ khái quát toàn dataset.

Bản cục bộ: code thực nghiệm ở trên là snapshot giữ nguyên byte, không sửa production. Snapshot, CSV, hình, dataset, checkpoint và cache raw npz chưa được đưa lên nhánh tài liệu; xem [phạm vi công bố và khôi phục môi trường](REPRODUCIBILITY.md). Posterior được xuất CSV tại máy để kiểm tra.

Phạm vi nhánh GitHub này chỉ gồm tài liệu. CSV, hình và code snapshot mới giữ cục bộ; xem [phạm vi công bố](REPRODUCIBILITY.md).
