from pathlib import Path
import pandas as pd, json
out=Path('outputs/alignment_step1');df=pd.read_csv(out/'diagnostics.csv',dtype={'id':str})
S=set('0099 0219 0287 0543 0775 0801 0996 0976 1053 1112 1170 1914 1897 2062 2192'.split())
M=set('0757 0864 2104 2203'.split())
turn=set('0099 0137 0219 0287 0604 0864 0801 2062 2104 2203 2192'.split())
notes={
'0099':'S rõ ở người mục tiêu; xoay thân khi ném.',
'0137':'Chọn người STAFF ở tiền cảnh; vote I/S không có giá trị cho người GT.',
'0219':'S; xoay thân khi đánh bóng.',
'0287':'S dù nhãn pose=back; median vai dương nên heuristic tình cờ chọn S.',
'0433':'Có frame chọn người hỗ trợ đứng sau thay vì người nằm ghế.',
'0443':'Chọn người hỗ trợ đứng sau ở các frame kiểm tra.',
'0604':'S ở f15; I gợi ý tại f36; f41 chọn người khác, không kết luận clip. Không xếp mixed đã xác nhận.',
'0543':'S rõ khi đi về phía camera.',
'0757':'S 219 frame chẩn đoán; I tại f176 và f183 (6/6 cặp); hai lỗi ngắn, không có xoay hướng rõ.',
'0775':'S trên các cặp phân biệt được; nhiều người nền nhưng target đúng ở frame kiểm tra.',
'0864':'S f27/29; I f41–43,45–47; xoay thân. Raw chọn I, đầu vào production chọn S.',
'0801':'S trên khớp nhìn thấy; heuristic giữ I vì median vai âm, gần nghiêng.',
'0996':'S rõ.', '0976':'S trên các cặp phân biệt được.',
'1053':'S rõ.', '1112':'S rõ dù pose metadata=left.',
'1316':'YOLO chọn người đứng bên trái thay vì người kéo xà.',
'1170':'S rõ.',
'1484':'25/31 frame không phát hiện; người nhỏ, nghiêng và chồng khớp.',
'1377':'31/73 frame không phát hiện; một bên bị che/chồng, thiếu bằng chứng hai phía.',
'1597':'Đổi target: có frame chọn người tập, có frame chọn người đứng; không kết luận toàn clip.',
'1622':'Chọn người giữ chân thay vì người GT nằm tập.',
'1787':'Đổi target giữa người tập và huấn luyện viên.',
'1691':'Nghiêng, che khuất, hai phía chồng nhau; không phân biệt chắc I/S.',
'1914':'S rõ ở tay; không suy luận cho chân không quan sát được.',
'1897':'S rõ ở tay; crop mất vai, production median âm nên heuristic I; raw median dương.',
'2062':'S rõ cả f1,32,47,63; lưng/chéo lưng, heuristic I sai.',
'2104':'I tại f21–24,37–38; S ở đoạn khác, xem f61. Có xoay thân.',
'2203':'S chiếm ưu thế; I ngắn tại f29 (2 cặp), các cặp khác mơ hồ; không gọi là đổi toàn thân chắc chắn.',
'2192':'S trên cặp thấy rõ; lưng/nghiêng, heuristic I sai.'}
rows=[]
for r in df.to_dict('records'):
 sid=r['id'];cat='S' if sid in S else 'M' if sid in M else 'U'
 r.update(review=cat,identity_correct=False if cat!='U' else None,fixed_swap_correct=True if cat=='S' else False if cat=='M' else None,heuristic_correct=bool(r['heuristic_swap']) if cat=='S' else False if cat=='M' else None,body_turn_observed=sid in turn,review_note=notes[sid]);rows.append(r)
review=pd.DataFrame(rows);review.to_csv(out/'video_review.csv',index=False)
summary={'videos':len(df),'frames':int(df.frames.sum()),'identity_consistent':0,'fixed_swap_consistent':len(S),'heuristic_consistent':int(review.heuristic_correct.fillna(False).sum()),'inconclusive':sum(review.review=='U'),'mixed_or_local_reversal':len(M),'sustained_mixed':['0864','2104'],'brief_reversal':['0757','2203'],'turn_observed':len(turn)}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
text=f'''# Bước 1 — Audit Left/Right Penn Action, 29/10/2026

**Kết quả: heuristic hiện tại không phải quy tắc correspondence đáng tin cậy ở cấp clip.** Có phản ví dụ quay lưng cần swap nhưng heuristic giữ nguyên; có clip cần I và S tại các thời điểm khác nhau. Chưa sửa alignment, chưa train, chưa chạy downstream/ablation hay tuning.

| Tiêu chí | Kết quả |
|---|---:|
| Tổng số video kiểm tra | 30 ({summary['frames']} frame chạy YOLO) |
| Số video Identity đúng nhất quán trên bằng chứng quan sát được | 0 |
| Số video Fixed Swap đúng nhất quán trên bằng chứng quan sát được | 15 |
| Số video Current Heuristic đúng nhất quán trên bằng chứng quan sát được | 11 |
| Số video không thể kết luận ở cấp clip | 11 |
| Số video có hiện tượng đổi hướng/xoay thân quan sát được | 11 |
| Có I và S đúng ở thời điểm khác nhau | 4: 2 có đoạn nhiều frame, 2 đảo ngắn/cục bộ |
| Alignment thất bại rõ, ngoài nhóm chưa kết luận | 4 clip heuristic giữ I trong nhóm cần S; thêm 4 clip có đảo cục bộ/đổi correspondence |

“Đúng” ở đây là **đúng correspondence trên các khớp phân biệt được qua mẫu frame và chẩn đoán**, không phải mọi khớp đều chính xác về vị trí hoặc mọi frame đã được xem thủ công. Các clip chỉ thấy tay không chứng minh correspondence chân. Số 0 của Identity không có nghĩa Identity luôn sai ở mọi frame. 11 chưa kết luận không tính đúng hoặc sai cho bất kỳ phương án nào. 15 S + 4 mixed/cục bộ + 11 chưa kết luận = 30; 11 heuristic đúng là tập con của 15 S.

Cách làm: chọn ngẫu nhiên phân tầng 2 clip/hành động, seed `20261029`, từ 1.258 clip có `train==1`; clip thứ hai ưu tiên pose khác, nếu lớp chỉ có một pose thì lấy ID khác. Không dùng clip test (`train==-1`) và không đọc dự đoán test. Đây là mẫu từ official train, không chia lại train/validation; không dùng kết quả này như validation downstream. Pose metadata: {dict(df.pose.value_counts())}. Không diễn giải pose metadata là hướng của mọi frame.

Chạy YOLO11n-Pose có sẵn, CPU, imgsz=640, box confidence=0.25; chọn người có box confidence cao nhất như code hiện hành, không chọn theo GT. Chạy toàn bộ frame để lấy đúng median clip. Xem 5 frame cách đều/clip và thêm frame có bằng chứng I/S mạnh nhất. Đối chiếu thêm hình phóng to tại các trường hợp bất đồng. Đây là review bằng hình của trợ lý, chưa có hai annotator độc lập hay kiểm tra đồng thuận.

Chẩn đoán GT chỉ hỗ trợ tìm frame cần xem: mỗi cặp phải có cả hai GT visible, hai YOLO confidence ≥0.5, tọa độ hợp lệ, độ tách hai phía ≥max(5px, 0.15×torso). So trung bình khoảng cách Euclidean hai đầu của cặp dưới I/S, chỉ vote khi chênh lệch >max(3px, 0.15×độ tách GT). Frame rõ khi ≥2 cặp cùng vote và không có vote ngược; còn lại `?`. Các ngưỡng phục vụ sàng lọc audit, không tuning calibration, không phải định nghĩa chân lý. Vote trên nhầm người vẫn có thể trông thuyết phục nên phải kiểm tra RGB và giữ U. Không biến oracle theo GT thành alignment triển khai.

Đổi hướng được đánh dấu bằng RGB khi có xoay thân rõ giữa các frame, không tính riêng gập/ngồi/nhảy. 11 ID: {', '.join(sorted(turn))}. Đây là số quan sát sơ bộ, không phải phân loại góc quay tự động hay danh sách đầy đủ mọi chuyển hướng giữa frame lấy mẫu.

Các phát hiện cần review:

- **2062**: quay lưng/chéo lưng, S khớp GT ở f1/32/47/63; heuristic I (`median dx=-17.08px`) gây nối sai hai phía. Chẩn đoán 63/63 frame ủng hộ S. **2192**, **0801** cũng cần S nhưng heuristic giữ I. Vì vậy không thể dùng “quay lưng ⇒ không swap” để giải thích convention.
- **1897**: S khớp tay GT, nhưng vai bị crop/độ tin cậy thấp. Median vai raw `+170.88px`, sau các stage 1–3 là `-25.23px`, khiến heuristic production chọn I. Đây là lỗi quyết định phụ thuộc đầu vào xử lý trước; không quy hết cho góc nhìn.
- **0864**: S ở f27/29, I trên đoạn f41–43 và f45–47. Median raw `-0.23px`, production `+0.89px`: dấu không ổn định. **2104**: I ở f21–24 và f37–38, S ở các đoạn khác (ví dụ f61). Hai clip này không thể được căn đúng xuyên suốt bằng một permutation duy nhất.
- **0757**: S 219 frame chẩn đoán nhưng I tại f176,183, cả 6 cặp. Đây là đảo ngắn của dự đoán/correspondence quan sát được, không có bằng chứng người quay từ trước ra sau. **2203**: I ngắn ở f29 trên 2 cặp phân biệt được, S ở phần lớn đoạn khác; không kết luận toàn thân đảo nhãn. Hai ca ngắn này tách riêng khỏi đoạn mixed nhiều frame.
- **0604**: có gợi ý I tại f36 và S ở f15, nhưng f41 chọn nhầm người; giữ U, không nâng thành mixed đã xác nhận. Các vote I/S của **0137** cũng không chứng minh đổi correspondence vì target sai.
- **0137, 0433, 0443, 1316, 1597, 1622, 1787**: có sai target/đổi target trên RGB. Không thể sửa bằng đổi nhãn khớp. **1484, 1377, 1691** thiếu bằng chứng vì dropout, nghiêng và che khuất. Cùng 0604 tạo nhóm 11 U.

Code hiện tại ở [src/skeleton_calibrator.py](../src/skeleton_calibrator.py#L338), bản chụp nguyên class tại [current_alignment_snapshot.py](../outputs/alignment_step1/current_alignment_snapshot.py). Phần quyết định chính:

```python
valid_mask = ~df_out[lx_col].isna() & ~df_out[rx_col].isna()
if valid_mask.sum() < 2:
    return df_out
delta_x = df_out.loc[valid_mask, lx_col] - df_out.loc[valid_mask, rx_col]
median_delta = float(np.median(delta_x))
if median_delta > 0.0:
    for left_kpt, right_kpt in paired_joints:
        for suffix in ["x", "y", "conf"]:
            col_l = f"pred_{{left_kpt}}_{{suffix}}"
            col_r = f"pred_{{right_kpt}}_{{suffix}}"
            if col_l in df_out.columns and col_r in df_out.columns:
                tmp = df_out[col_l].values.copy()
                df_out[col_l] = df_out[col_r].values.copy()
                df_out[col_r] = tmp
```

Sáu cặp: shoulder, elbow, wrist, hip, knee, ankle. Clip ≤2 frame hoặc thiếu cột vai trả nguyên; chỉ dùng tính hợp lệ NaN, không confidence hay margin. `mode` được lưu nhưng không kiểm soát nhánh trong class. Trong `FullCalibrationPipeline`, alignment là stage 4 sau interpolation → bone constraint → smoothing. Audit dùng quyết định đúng tại đầu vào stage 4 làm **Current Heuristic**, rồi áp permutation đó lên raw để so hình công bằng với Identity/Fixed Swap. Cột raw heuristic được lưu riêng; raw và production khác quyết định tại **0864, 1597, 1897**. Không diễn giải khác biệt này thành ablation accuracy.

Hình minh họa (xanh lam=L, tím=R; GT vòng tròn/nét mảnh, YOLO dấu ×/nét đậm; số 0–5 tương ứng vai, khuỷu, cổ tay, hông, gối, cổ chân; YOLO chỉ vẽ confidence ≥0.5):

![2062: quay lưng vẫn cần Fixed Swap](../outputs/alignment_step1/figures/evidence_2062.jpg)

![0864: bốn cột GT, Identity, Fixed Swap, Current Heuristic](../outputs/alignment_step1/figures/0864.jpg)

![0757: đảo ngắn tại f176 và f183](../outputs/alignment_step1/figures/evidence_0757.jpg)

Bộ [30 hình bốn phương án](../outputs/alignment_step1/figures/) có 5–7 frame/clip, gồm cột Current Heuristic; `evidence_*.jpg` là hình phóng to 3 cột. GT luôn là nhãn Penn nguyên bản, không swap GT. Hình heuristic chỉ thể hiện alignment, không thêm các hiệu chỉnh tọa độ khác.

Bảng từng video: I=Identity, S=Fixed Swap, M=thay đổi theo thời gian/cục bộ, U=chưa kết luận. “Đạt” chỉ nghĩa nhất quán theo phạm vi quan sát nêu trên; M không được tính đạt toàn clip. Cột I/S/? là **số frame chẩn đoán tự động**, chưa loại nhầm target, không phải số frame chấm tay.

| ID | Hành động | Pose | I/S/? | Heuristic | Review | Ghi chú |
|---|---|---|---|---|---|---|
'''
for r in rows:
 sid=r['id'];h='S' if r['heuristic_swap'] else 'I';status='S; H đạt' if sid in S and r['heuristic_swap'] else 'S; H sai' if sid in S else 'M; không đạt clip' if sid in M else 'U'
 text+=f"| [{sid}](../outputs/alignment_step1/figures/{sid}.jpg) | {r['action']} | {r['pose']} | {r['I_frames']}/{r['S_frames']}/{r['uncertain_frames']} | {h} | {status} | {notes[sid]} |\n"
text+='''
Artifact tái lập: [sample.json](../outputs/alignment_step1/sample.json), [video_review.csv](../outputs/alignment_step1/video_review.csv), [frame_diagnostics.csv](../outputs/alignment_step1/frame_diagnostics.csv), [provenance.json](../outputs/alignment_step1/provenance.json) (hash model/code, phiên bản thư viện), dự đoán raw từng clip trong `outputs/alignment_step1/*.npz`. Script [audit_left_right_step1.py](../src/audit_left_right_step1.py) tái lập lấy mẫu, inference, chẩn đoán và các contact sheet; `video_review.csv` là quyết định review thủ công tách riêng, không được script tự suy ra từ vote.

```bash
/home/dekii2275/miniconda3/envs/penn_pose/bin/python src/audit_left_right_step1.py
```

**Kết luận giới hạn cho Bước 1:** có mismatch correspondence rõ và heuristic bỏ sót/tạo lỗi tại các đoạn có bằng chứng I, nhưng chưa định lượng phần đóng góp của nó vào sai số toàn dataset hoặc downstream accuracy. Mẫu này ủng hộ Fixed Swap làm baseline kiểm soát cần xem xét, không chứng minh fixed swap đúng mọi clip hay Penn dùng một quy ước viewer-centric toàn cục. Cần review các ngoại lệ và lỗi chọn người trước khi coi alignment đã hợp lệ cho downstream. Chưa thay đổi baseline hay thuật toán; chờ quyết định Bước 2. 30 clip là pilot tìm lỗi, chưa đủ khái quát toàn dataset.
'''
Path('reports/LEFT_RIGHT_ALIGNMENT_STEP1_2026-10-29.md').write_text(text)
print(summary)
