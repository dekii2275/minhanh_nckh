import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pose_view_audit"
c = pd.read_csv(OUT / "per_clip.csv", dtype={"id": str})
f = pd.read_csv(OUT / "per_frame.csv", dtype={"id": str})
s = pd.read_csv(OUT / "summary_by_pose.csv")
manifest = json.loads(
    (ROOT / "outputs/alignment_step2/validation_manifest.json").read_text(encoding="utf8")
)
assert set(c.id) == {v["id"] for v in manifest} and len(c) == 253 and c.id.is_unique
assert len(f) == c.total_frames.sum() == 17747
assert int((f.valid_joints > 0).sum()) == int(c.valid_frames.sum())
for row in c.itertuples():
    subset = f[(f.id == row.id) & (f.valid_joints > 0)]
    assert len(subset) == row.valid_frames
    if len(subset):
        assert np.isclose(subset.d_raw.mean(), row.d_raw) and np.isclose(
            subset.d_swap.mean(), row.d_swap
        )
for row in s.itertuples():
    sub = c[(c.pose == row.pose) & (c.decision != "no_valid")]
    assert len(sub) == row.clips_valid and (sub.decision == "swap").sum() == row.swap_better_n
    assert np.isclose(sub.d_raw.mean(), row.mean_d_raw_px) and np.isclose(
        sub.d_swap.mean(), row.mean_d_swap_px
    )
retained = c[(c.valid_frames > 0) & (c.valid_frames / c.total_frames >= 0.5)]
sensitivity = retained.groupby("pose").agg(
    valid_clips=("id", "size"), swap_n=("decision", lambda x: (x == "swap").sum())
)
sensitivity["swap_pct"] = 100 * sensitivity.swap_n / sensitivity.valid_clips
sensitivity.to_csv(OUT / "sensitivity_50pct_coverage.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
s = s.set_index("pose").loc[["front", "back", "left", "right"]].reset_index()
x = np.arange(len(s))
ax[0].bar(x - 0.18, s.mean_d_raw_px, width=0.36, label="Raw")
ax[0].bar(x + 0.18, s.mean_d_swap_px, width=0.36, label="Swap")
ax[0].set_xticks(x, s.pose)
ax[0].set_ylabel("Sai số trung bình / clip (px)")
ax[0].set_title("Khoảng cách đến GT")
ax[0].legend()
ax[1].bar(x, s.swap_better_pct, color="#348a83")
ax[1].set_xticks(x, s.pose)
ax[1].set_ylim(0, 110)
ax[1].set_ylabel("% clip hợp lệ")
ax[1].set_title("Clip có swap tốt hơn raw")
for i, row in enumerate(s.itertuples()):
    ax[1].text(i, row.swap_better_pct + 2, f"{row.swap_better_n}/{row.clips_valid}", ha="center")
fig.tight_layout()
fig.savefig(OUT / "comparison.png", dpi=150)
plt.close(fig)
rows = []
for r in s.itertuples():
    ci = binomtest(int(r.swap_better_n), int(r.clips_valid)).proportion_ci(confidence_level=0.95)
    rows.append(
        f"| {r.pose} | {r.clips_valid}/{r.clips_total} | {r.frames_valid}/{r.frames_total} | {r.mean_d_raw_px:.2f} | {r.mean_d_swap_px:.2f} | {r.swap_better_n}/{r.clips_valid} ({r.swap_better_pct:.2f}%) | {100 * ci.low:.1f}–{100 * ci.high:.1f}% |"
    )
lines = [
    "# Audit correspondence trái/phải theo góc nhìn — Penn Action validation",
    "",
    "Chạy lại YOLO11n-Pose trên 253 clip validation của Protocol v2. Chỉ dùng annotation official train, không train, không đọc official test. Trọng số YOLO trùng SHA-256 thí nghiệm trước; dự đoán được tính lại bằng GPU và phiên bản thư viện hiện tại, nên không tuyên bố giống bit với lần chạy CPU cũ. Mọi hash annotation được kiểm tra với manifest.",
    "",
    "| Pose | Clip hợp lệ / tổng | Frame hợp lệ / tổng | Raw (px) | Swap (px) | Swap tốt hơn | CI 95% chính xác theo nhị thức |",
    "|---|---:|---:|---:|---:|---:|---:|",
    *rows,
    "",
    f"Tổng: {len(c)} clip / {len(f):,} frame; {int((c.valid_frames > 0).sum())} clip và {int(c.valid_frames.sum()):,} frame có thể tính. Các clip không có frame hợp lệ bị loại khỏi mẫu số; không coi chúng là thất bại hay thành công.",
    "",
    "Cách tính: pose là nhãn cấp clip, không khẳng định góc nhìn không thay đổi theo thời gian. Với mỗi frame, chọn detection confidence cao nhất; lấy trung vị tọa độ của 12 khớp YOLO hữu hạn, dương và có confidence > 0 làm tâm; giữ frame nếu tâm nằm trong bbox GT xyxy. Với cả raw và swap, chỉ dùng cùng tập khớp có GT visibility=1, tọa độ GT hữu hạn, và cả hai dự đoán ở cặp trái/phải đều có tọa độ dương hữu hạn, confidence > 0. Đây không phải confidence ≥0.5. Tính trung bình khớp trong frame, trung bình frame trong clip, rồi trung bình clip trong nhóm. Tỷ lệ swap tốt hơn tính theo clip; hòa tính trong mẫu số nhưng không là swap tốt hơn. CI nhị thức giả định clip độc lập, không phản ánh biến thiên model/nguồn quay hay sai lệch do lọc.",
    "",
    "## Độ nhạy theo tỷ lệ frame được giữ",
    "",
    "Giữ clip có ít nhất 50% frame hợp lệ (phân tích bổ sung, không thay thế bảng chính):",
    "",
    "| Pose | Swap tốt hơn / clip | Tỷ lệ |",
    "|---|---:|---:|",
]
for pose, r in sensitivity.iterrows():
    lines.append(f"| {pose} | {int(r.swap_n)}/{int(r.valid_clips)} | {r.swap_pct:.2f}% |")
lines += [
    "",
    "## Diễn giải và quyết định",
    "",
    "- Đạt điều kiện số học A (>95% ở cả front và back); không đạt B. Bằng chứng cho thấy fixed swap thường làm giảm khoảng cách trên các frame được chọn; không chứng minh nhãn nào là giải phẫu, thứ tự gốc bị viết sai, hay nguyên nhân duy nhất của sự khác biệt.",
    "- Kiểm tra mã cho thấy danh sách tên COCO và các chỉ số Penn hiện đều được đặt theo thứ tự left/right tương ứng. Điều này không xác lập convention thực tế của annotation, nhưng chưa có bằng chứng để khẳng định lỗi lệch chẵn/lẻ. Pose là metadata cấp clip, và có thể xoay người giữa clip.",
    "- Bộ lọc tâm trong bbox chỉ là proxy, không chứng minh chọn đúng người khi có người khác trong bbox; cũng có thể loại nhầm target đúng khi bbox hẹp/crop. Clip chỉ cần một frame hợp lệ để vào bảng chính, nên cần xem độ nhạy phía trên. Sai số pixel phụ thuộc kích thước ảnh, pose, che khuất và độ tin cậy dự đoán; không dùng các khoảng sai số giả định làm kết luận.",
    "- Báo cáo cũ ghi 11/30 clip chưa thể kết luận (36,7%); không phải 36,7% clip đã xác nhận bắt nhầm người.",
    "- Đã tạo adapter tùy chọn `src/penn_fixed_swap.py`: swap đúng một lần và đường calibration tắt stage alignment. Không thay mặc định production hay xóa class dùng chung: thống kê này không chứng minh nguyên nhân, và checkpoint GT-trained dùng trong benchmark không có trong workspace nên chưa thể xác nhận tương thích end-to-end. Cần đánh giá pipeline dự kiến trên validation và test độc lập trước khi thay mặc định.",
    "- Benchmark đã công bố trước đó trên cùng validation: raw Top-1 44,66%, fixed swap 67,59%, heuristic 66,80% và full 64,43%. Đó là kết quả cũ trên dự đoán CPU/checkpoint cũ, không phải metric được chạy lại từ adapter mới và không phải test độc lập.",
    "",
    "## Tái lập",
    "",
    "Chạy `.\\.venv\\Scripts\\python.exe src/audit_pose_correspondence.py`, sau đó `.\\.venv\\Scripts\\python.exe src/report_pose_correspondence.py`. Cache lưu theo từng clip kèm fingerprint; thay model/phiên bản/thiết bị cần dùng thư mục cache mới hoặc xóa cache sau khi kiểm tra. Dự đoán không dùng GT; GT chỉ dùng trong bước lọc và đo chẩn đoán. Kiểm tra adapter: `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_penn_fixed_swap.py -v`.",
    "",
    "Chi tiết: `outputs/pose_view_audit/per_clip.csv`, `per_frame.csv`, `summary_by_pose.csv`, `sensitivity_50pct_coverage.csv`, `provenance.json` và `comparison.png`.",
]
(ROOT / "reports/POSE_VIEW_CORRESPONDENCE_AUDIT.md").write_text(
    "\n".join(lines) + "\n", encoding="utf8"
)
print("Independent aggregation checks passed. Sensitivity:")
print(sensitivity.to_string())
print("valid clips", int((c.valid_frames > 0).sum()), "valid frames", int(c.valid_frames.sum()))
