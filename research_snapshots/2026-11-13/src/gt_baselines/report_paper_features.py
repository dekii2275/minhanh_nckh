"""Write an evidence-backed report of the independent paper feature adaptation."""
import csv
from pathlib import Path

import gt_crop_paper_features_experiment as study

NAMES = {'coordinates': 'Tọa độ — baseline', 'angles': 'Tọa độ + góc',
         'velocity': 'Tọa độ + vận tốc', 'full': 'Tọa độ + góc + vận tốc + COM',
         'capacity_control': 'Tọa độ + 35 kênh zero (đối chứng tham số)',
         'mean5_coordinates': 'Trung bình 5 frame + tọa độ',
         'mean5_full': 'Trung bình 5 frame + bộ đặc trưng đầy đủ'}


def main(out=study.OUT):
    summary = study.previous.read_json(out/'summary.json')
    records = study.previous.read_json(out/'results.json')
    verified = study.previous.read_json(out/'verification.json')
    methods = summary['methods_vs_coordinates']
    lines = ['# Thử đặc trưng động học từ bài Ma và cộng sự trên YOLO crop GT → BiLSTM', '',
             '## Bài gốc và phần được áp dụng', '',
             f'[Interpretable skeleton-based movement profiles for formative feedback in school physical activity education]({study.PAPER}), Frontiers in Public Health, DOI 10.3389/fpubh.2026.1925220; đọc ngày 03/11/2026.', '',
             'Mục 3.8/Bảng 7 bổ sung góc, vận tốc và vị trí/tốc độ COM vào tọa độ. Đây là đặc trưng cho classifier; bốn điểm profile tổng hợp là đầu ra mô tả khác. Bài báo không cung cấp bộ sửa tọa độ riêng để lắp trực tiếp. Con số Penn 0,893 → 0,905 là macro-F1, với p Holm=0,079.', '',
             '## Thiết lập thích ứng của dự án', '',
             '- Cùng cache YOLO11n-Pose crop bbox GT và chọn detection bằng GT IoU ≥0,3; swap cố định như baseline. Chỉ lấy 1.005 train và 253 validation; 15 lớp, 12 khớp.',
             '- Giữ nguyên chuẩn hóa hông/shoulder và resampling 64 bước của pipeline đạt 93,54%. Không đổi sang chuẩn hóa torso của bài để tách tác động của đặc trưng bổ sung.',
             '- 8 góc: vai, khuỷu, hông, gối trái/phải; arccos tích vô hướng hai vector xương, chia π. Mapping cụ thể trong protocol và mã. Góc không hợp lệ/chiều dài xương bằng 0 đặt thành 0.',
             '- Vận tốc: sai phân bậc một của 24 tọa độ chuẩn hóa giữa hai bước trong chuỗi 64 bước; bước đầu bằng 0. Chỉ tính khi cả hai bước có khớp hợp lệ; không coi đây là vận tốc theo giây.',
             '- COM 2D proxy: trung bình hai vai và hai hông trong hệ tọa độ đã chuẩn hóa; thêm x, y và chuẩn độ dời giữa hai bước. Không phải tâm khối lượng được đo vật lý.',
             '- Mask hợp lệ từ pose YOLO gốc: finite, x/y>0 và cả hai điểm nguồn nội suy đều hợp lệ. Không dùng visibility hoặc tọa độ GT để sinh đặc trưng. Kênh tọa độ gốc được giữ nguyên tuyệt đối.',
             '- Input đầy đủ gồm 24 tọa độ + 8 góc + 24 vận tốc + 2 COM + 1 tốc độ COM = 59 chiều. Nhánh đối chứng thêm 35 kênh zero có cùng kích thước/khởi tạo mô hình với nhánh đầy đủ, để kiểm tra ảnh hưởng tăng số tham số.',
             '- BiLSTM + attention, ba seed 0/1/2, 50 epoch, batch 32; Adam lr 0,001, weight decay 0,0001, cosine scheduler, clip norm 1. Chọn checkpoint bằng accuracy validation cao nhất, ưu tiên epoch sớm khi hòa.',
             '- Tái sử dụng sáu checkpoint đối chứng raw/mean5 đã được kiểm chứng ở thí nghiệm trước; huấn luyện mới 15 checkpoint cho năm nhánh bổ sung. Nạp lại cả 21 checkpoint trên CPU để so toàn bộ dự đoán.',
             '- Đây là triển khai độc lập thích ứng sang BiLSTM và pose YOLO, với khác biệt về dữ liệu/split/normalization/model so với bài gốc; không phải tái lập chính xác kết quả bài.', '',
             '## Kết quả trên validation', '',
             '| Phương án | Chiều input | Tham số | Accuracy TB ± SD (%) | Macro-F1 TB (%) | Δ so baseline (điểm %) | Bootstrap 95% Δ |',
             '|---|---:|---:|---:|---:|---:|---|']
    for name, r in methods.items():
        lo, hi = r['paired_video_bootstrap_95_pp']
        dim = study.kinematics.DIMENSIONS[study.METHODS[name][1]]
        lines.append(f'| {NAMES[name]} | {dim} | {r["parameters"]:,} | {r["accuracy_mean_pct"]:.2f} ± {r["accuracy_sd_pct"]:.2f} | {r["macro_f1_mean_pct"]:.2f} | {r["gain_pp"]:+.2f} | [{lo:+.2f}; {hi:+.2f}] |')
    lines += ['', '| Phương án | Seed 0 | Seed 1 | Seed 2 | Epoch chọn |', '|---|---:|---:|---:|---|']
    for name, r in methods.items():
        accuracy = ' | '.join(f'{s["accuracy_pct"]:.2f}%' for s in r['seeds'])
        epochs = ', '.join(str(s['epoch']) for s in r['seeds'])
        lines.append(f'| {NAMES[name]} | {accuracy} | {epochs} |')
    lines += ['', '## Các đối chứng chính', '']
    for key, label in [('full_vs_capacity_control', 'Bộ đầy đủ so với tọa độ + 35 kênh zero'),
                       ('mean5_full_vs_mean5_coordinates', 'Thêm bộ đặc trưng đầy đủ trên pose đã lọc trung bình'),
                       ('mean5_full_vs_raw_full', 'Thêm lọc trung bình khi đã dùng bộ đặc trưng đầy đủ')]:
        r = summary[key]; lo, hi = r['paired_video_bootstrap_95_pp']
        lines.append(f'- {label}: **{r["gain_pp"]:+.2f} điểm phần trăm**, bootstrap 95% [{lo:+.2f}; {hi:+.2f}].')
    best = max(methods, key=lambda m: methods[m]['accuracy_mean_pct'])
    r = methods[best]
    lines += ['', '## Diễn giải', '',
              f'- Accuracy trung bình cao nhất trong các cấu hình đã thử: **{NAMES[best]}**, **{r["accuracy_mean_pct"]:.2f}%**. Việc chọn cấu hình này đã nhìn validation; không coi đó là kiểm định độc lập.',
              '- Trong lần thử này, thêm riêng góc hoặc vận tốc tăng accuracy 0,13 điểm phần trăm nhưng macro-F1 đều thấp hơn baseline; các khoảng bootstrap chênh lệch accuracy chứa 0. Bộ đầy đủ và bản đầy đủ kết hợp làm mượt đều thấp hơn baseline tương ứng. Chưa có bằng chứng để thay baseline bằng bộ đặc trưng thích ứng này.',
              '- Các khoảng bootstrap lấy mẫu lại video, trung bình chênh lệch ba seed trên từng video. Khoảng chứa 0 chưa xác nhận cải thiện rõ, cũng không chứng minh tương đương. Khoảng chưa hiệu chỉnh thử nhiều cấu hình, thiên lệch chọn checkpoint hoặc bất định seed.',
              '- Các nhánh góc/vận tốc có kích thước input và khởi tạo tham số khác baseline; dùng thêm đối chứng 59 chiều khi diễn giải nhánh đầy đủ.',
              '- Các đặc trưng này không sửa tọa độ nên không thể suy ra sai số pose giảm. Không dùng percentile cùng lớp hoặc nhãn hành động thật để sinh đầu vào.',
              '- Bbox GT vẫn được dùng mỗi frame. Validation đã tham gia phát triển; chưa chạy official test. Nhánh lọc trung bình dùng frame tương lai.', '',
              '## Artifact và kiểm chứng', '',
              f'- Đã kiểm chứng **{verified["runs"]} checkpoint**, **{verified["cpu_prediction_matches"]} dự đoán CPU khớp**.',
              f'- Thư mục kết quả: `{out.relative_to(study.ROOT).as_posix()}`; gồm protocol/hash, feature cache, lịch sử train, checkpoint, summary và dự đoán từng video.',
              '- Mã đặc trưng: `src/gt_baselines/paper_kinematic_features.py`.',
              '- Mã thí nghiệm: `src/gt_baselines/gt_crop_paper_features_experiment.py`.',
              '- Chạy: `.venv/Scripts/python.exe -u src/gt_baselines/gt_crop_paper_features_experiment.py --stage all` (yêu cầu thư mục output mới).', '']
    destination = study.ROOT/'report/03112026_nckh/BO_SUNG_DAC_TRUNG_BAI_MOVEMENT_PROFILE_2026-11-03.md'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    classes = []
    for name in study.METHODS:
        group = [r for r in records if r['method'] == name]
        for label, action in enumerate(study.base.ACTION_CLASSES):
            predictions = [[p for p in r['predictions'] if p['target'] == label] for r in group]
            scores = [100*sum(p['predicted'] == label for p in pp)/len(pp) for pp in predictions]
            classes.append({'method': name, 'action': action, 'validation_videos': len(predictions[0]),
                            'accuracy_mean_pct': sum(scores)/len(scores)})
    with (out/'per_class_accuracy.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(classes[0])); w.writeheader(); w.writerows(classes)
    print(destination)


if __name__ == '__main__':
    main()
