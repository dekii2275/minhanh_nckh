"""Pose diagnostics and Vietnamese report for the crop-YOLO correction study."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

import gt_crop_correction_experiment as study

NAMES = {'fixed_swap': 'Không hiệu chỉnh (swap như baseline)',
         'interpolate3': 'Nội suy khoảng thiếu ≤3 frame',
         'mean5': 'Trung bình 5 frame', 'median5': 'Trung vị 5 frame',
         'interpolate3_mean5': 'Nội suy + trung bình 5 frame'}


def pose_metrics(out):
    totals = {m: dict(n=0, error=0., pck=0, gt=0, predicted=0, all_pck=0,
                      motion_n=0, motion_error=0.) for m in study.METHODS}
    for row in study.source.rows():
        if row['seed0_split'] != 'validation':
            continue
        sid = row['id']
        with np.load(study.source.OUT/'poses'/f'{sid}.npz') as d:
            variants, _ = study.corrections.make_variants(d['xy'], d['conf'])
        if study.base.sha(study.base.LABELS/f'{sid}.mat') != row['label_sha256']:
            raise ValueError('GT changed')
        mat = sio.loadmat(study.base.LABELS/f'{sid}.mat')
        gt = np.stack((mat['x'][:, 1:13], mat['y'][:, 1:13]), axis=-1).astype(np.float64)
        bbox = np.full((len(gt), 4), np.nan)
        bbox[:min(len(bbox), len(mat['bbox']))] = mat['bbox'][:len(bbox)]
        wh = bbox[:, 2:] - bbox[:, :2]
        diag = np.linalg.norm(wh, axis=1)
        valid = ((mat['visibility'][:, 1:13] > 0) & np.isfinite(gt).all(2) & (gt > 0).all(2)
                 & (np.isfinite(wh).all(1) & (wh > 0).all(1))[:, None])
        pred_valid = {m: np.isfinite(variants[m][0]).all(2) & (variants[m][0] > 0).all(2)
                      & np.isfinite(variants[m][1]) & (variants[m][1] > 0) for m in study.METHODS}
        common = valid & np.logical_and.reduce(list(pred_valid.values()))
        motion_common = common[1:] & common[:-1]
        for name in study.METHODS:
            pose = variants[name][0]
            dist = np.linalg.norm(pose - gt, axis=2)
            norm = dist / np.maximum(diag[:, None], 1e-8)
            motion = study.corrections.displacement_error(pose, gt, diag)
            a = totals[name]
            a['n'] += int(common.sum()); a['error'] += float(dist[common].sum())
            a['pck'] += int((common & (norm <= .2)).sum())
            a['gt'] += int(valid.sum()); a['predicted'] += int((valid & pred_valid[name]).sum())
            a['all_pck'] += int((valid & pred_valid[name] & (norm <= .2)).sum())
            a['motion_n'] += int(motion_common.sum()); a['motion_error'] += float(motion[motion_common].sum())
    result = {m: {'common_joints': a['n'], 'mean_error_px': a['error']/a['n'],
                  'pck20_common_pct': 100*a['pck']/a['n'],
                  'coverage_pct': 100*a['predicted']/a['gt'],
                  'pck20_missing_incorrect_pct': 100*a['all_pck']/a['gt'],
                  'velocity_error_norm_per_frame': a['motion_error']/a['motion_n']}
              for m, a in totals.items()}
    study.write_json(out/'pose_quality.json', {'methods': result,
        'definition': 'Validation GT-visible 12 joints, valid GT bbox. Common joints valid in every method, conf>0. PCK threshold .2 GT bbox diagonal; no 1px coordinate shift. Motion: normalized displacement error, not acceleration or FPS-normalized velocity. GT used only for scoring, never correction.'})
    return result


def report(out, destination):
    s = study.read_json(out/'summary.json')
    q = study.read_json(out/'pose_quality.json')['methods']
    verification = study.read_json(out/'verification.json')
    class_rows = []
    for scenario in ('frozen', 'retrained'):
        records = study.read_json(out/f'{scenario}_results.json')
        for name in study.METHODS:
            group = [r for r in records if r['method'] == name]
            for label, action in enumerate(study.base.ACTION_CLASSES):
                accuracy = [100*np.mean([p['predicted'] == label for p in r['predictions'] if p['target'] == label])
                            for r in group]
                count = sum(p['target'] == label for p in group[0]['predictions'])
                class_rows.append({'scenario': scenario, 'method': name, 'action': action,
                                   'validation_videos': count, 'accuracy_mean_pct': float(np.mean(accuracy))})
    with (out/'per_class_accuracy.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(class_rows[0])); w.writeheader(); w.writerows(class_rows)
    lines = ['# Hiệu chỉnh skeleton trên YOLO crop GT → BiLSTM + attention', '',
             '## Thiết lập', '',
             '- Cùng cache pose YOLO11n-Pose crop bằng bbox GT, mở rộng 25% mỗi phía; chọn detection bằng IoU GT ≥0,3.',
             '- 1.005 video train, 253 validation; cùng split seed 0. Không truy cập official test.',
             '- Giữ swap cố định của baseline và chuẩn hóa/resampling 64 bước. Hiệu chỉnh chỉ dùng tọa độ và confidence dự đoán.',
             '- Nội suy khoảng thiếu/khớp confidence <0,30 có hai đầu tin cậy, tối đa 3 frame; không ngoại suy đầu/cuối. Giá trị chưa xử lý giữ nguyên như baseline.',
             '- Trung bình/trung vị dùng cửa sổ ±2 frame, ít nhất 3 quan sát confidence ≥0,30; chỉ sửa target tin cậy. Bản kết hợp nội suy trước rồi trung bình.',
             '- Nhánh giữ classifier dùng ba checkpoint YOLO-trained cũ. Nhánh huấn luyện lại: từ đầu cho mỗi phương án, 3 seed 0/1/2, 50 epoch, batch 32, Adam lr 0,001 và cùng scheduler.',
             '- Chọn checkpoint accuracy validation cao nhất, ưu tiên epoch sớm khi bằng nhau. Nhánh không hiệu chỉnh được huấn luyện lại với cùng ngân sách.', '',
             '## Kết quả nhận diện', '']
    for scenario, title in [('frozen', 'Giữ classifier đạt 93,54%'), ('retrained', 'Huấn luyện lại trên từng loại pose')]:
        lines += ['### '+title, '',
                  '| Phương án | Accuracy TB ± SD (%) | Macro-F1 TB (%) | Δ accuracy (điểm %) | Bootstrap 95% Δ |',
                  '|---|---:|---:|---:|---:|']
        for m in study.METHODS:
            r = s[scenario][m]; lo, hi = r['paired_video_bootstrap_95_pp']
            lines.append(f'| {NAMES[m]} | {r["accuracy_mean_pct"]:.2f} ± {r["accuracy_sd_pct"]:.2f} | {r["macro_f1_mean_pct"]:.2f} | {r["gain_vs_baseline_pp"]:+.2f} | [{lo:+.2f}; {hi:+.2f}] |')
        lines += ['', '| Phương án | Seed 0 | Seed 1 | Seed 2 | Epoch được chọn |', '|---|---:|---:|---:|---|']
        for m in study.METHODS:
            seeds = s[scenario][m]['seeds']
            acc = ' | '.join(f'{r["accuracy_pct"]:.2f}%' for r in seeds)
            epochs = ', '.join(str(r['epoch']) for r in seeds) if scenario == 'retrained' else 'Checkpoint gốc'
            lines.append(f'| {NAMES[m]} | {acc} | {epochs} |')
        lines += ['']
    lines += ['## Chất lượng pose trên cùng khớp hợp lệ', '',
              '| Phương án | Sai số (px) | PCK chung (%) | Coverage (%) | PCK tính thiếu là sai (%) | Sai số chuyển động chuẩn hóa/frame |',
              '|---|---:|---:|---:|---:|---:|']
    for m in study.METHODS:
        r = q[m]
        lines.append(f'| {NAMES[m]} | {r["mean_error_px"]:.2f} | {r["pck20_common_pct"]:.2f} | {r["coverage_pct"]:.2f} | {r["pck20_missing_incorrect_pct"]:.2f} | {r["velocity_error_norm_per_frame"]:.5f} |')
    lines += ['', 'PCK dùng ngưỡng 0,2 đường chéo bbox GT; không phải accuracy hành động. Sai số tính trên cùng tập khớp hợp lệ ở cả năm phương án.', '',
              '## Diễn giải và giới hạn', '']
    for scenario, title in [('frozen', 'Giữ classifier'), ('retrained', 'Huấn luyện lại classifier')]:
        best = max(study.METHODS[1:], key=lambda m: s[scenario][m]['accuracy_mean_pct'])
        r = s[scenario][best]; lo, hi = r['paired_video_bootstrap_95_pp']
        lines.append(f'- {title}: phương án hiệu chỉnh có accuracy TB cao nhất là **{NAMES[best]}**, **{r["accuracy_mean_pct"]:.2f}%**, chênh **{r["gain_vs_baseline_pp"]:+.2f} điểm %** so với baseline tương ứng; bootstrap 95% [{lo:+.2f}; {hi:+.2f}].')
    lines += ['- Khoảng bootstrap lấy mẫu lại video, trung bình chênh lệch của ba seed trên mỗi video; không coi 759 dự đoán là các mẫu độc lập. Chưa tính thiên lệch chọn checkpoint hoặc thử nhiều phương án trên cùng validation.',
              '- Khoảng chứa 0 chưa cho bằng chứng cải thiện rõ và không chứng minh hai phương án tương đương. Kết quả chỉ áp dụng cho các cấu hình đã thử.',
              '- Cả crop và chọn detection vẫn dùng bbox GT từng frame. Các bộ lọc dùng frame tương lai; đây là xử lý cả video.',
              '- Ba seed dùng cùng split; validation đã tham gia phát triển. Chưa đánh giá độc lập trên test; official test từng được dùng trong lịch sử dự án.', '',
              '## Kiểm chứng và artifact', '',
              f'- Baseline giữ classifier được đối chiếu với toàn bộ 759 dự đoán gốc. Đã nạp lại {len(verification)} checkpoint huấn luyện lại trên CPU; {sum(r["predictions_matched"] for r in verification)} dự đoán khớp kết quả đã lưu.',
              f'- Kết quả: `{out.relative_to(study.ROOT).as_posix()}/summary.json`.',
              '- Có protocol với hash mã/cấu hình/checkpoint, hash từng cache pose, feature cache, lịch sử 50 epoch, kết quả từng seed và dự đoán từng video.',
              '- Lệnh chạy: `.venv/Scripts/python.exe -u src/gt_baselines/gt_crop_correction_experiment.py --stage all`.', '']
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=study.DEFAULT_OUT)
    p.add_argument('--pose-only', action='store_true')
    args = p.parse_args()
    pose_metrics(args.output)
    if not args.pose_only:
        report(args.output, study.ROOT/'report/03112026_nckh/HIEU_CHINH_YOLO_CROP_BILSTM_2026-11-03.md')
