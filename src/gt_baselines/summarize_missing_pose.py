"""Verify cached controls and compare repaired pose on the same GT joints."""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

root = Path(__file__).resolve().parents[2]
p = root / "outputs/missing_pose_validation_20261102"
q = root / "outputs/person_selection_validation_20261102"
with (p / "predictions.csv").open(newline="") as f:
    predictions = list(csv.DictReader(f))
with (q / "predictions_iou_0_3.csv").open(newline="") as f:
    old = {(r["seed"], r["id"]): int(r["gt_assisted_pred"]) for r in csv.DictReader(f)}
missing = [r for r in predictions if r["method"] == "missing"]
assert len(missing) == 759 and all(int(r["predicted"]) == old[r["seed"], r["id"]] for r in missing)
paired = []
for seed in ("0", "1", "2"):
    groups = {
        k: {r["id"]: r for r in predictions if r["seed"] == seed and r["method"] == k}
        for k in ("missing", "interpolation", "flow_crop")
    }
    for kind in ("interpolation", "flow_crop"):
        diff = [
            int(groups[kind][sid]["predicted"] == r["target"]) - int(r["predicted"] == r["target"])
            for sid, r in groups["missing"].items()
        ]
        paired.append(
            {
                "seed": int(seed),
                "method": kind,
                "improved": sum(x > 0 for x in diff),
                "worsened": sum(x < 0 for x in diff),
            }
        )
files = list((p / "per_video").glob("*.npz"))
assert len(files) == 253
common_count = 0
correct = {k: 0 for k in ("interpolation", "flow_crop")}
errors = {k: 0.0 for k in correct}
for f in files:
    d = np.load(f)
    g = d["eligible_gap"]
    outside = ~g
    for k in correct:
        assert np.array_equal(d["baseline"][outside], d[k][outside], equal_nan=True)
    mat = sio.loadmat(root / "data/raw/Penn_Action/labels" / f"{f.stem}.mat")
    gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], axis=-1)
    vis = mat["visibility"][:, 1:13] > 0
    wh = mat["bbox"][:, 2:] - mat["bbox"][:, :2]
    scale = np.linalg.norm(wh, axis=1)
    gtvalid = np.isfinite(wh).all(1) & (wh[:, 0] > 0) & (wh[:, 1] > 0)
    a = d["interpolation"]
    b = d["flow_crop"]

    def valid(x):
        return np.isfinite(x).all(-1) & (x[:, :, 0] > 0) & (x[:, :, 1] > 0)

    ok = (
        g[:, None]
        & vis
        & valid(gt)
        & gtvalid[:, None]
        & valid(a)
        & valid(b)
        & (d["flow_crop_conf"] > 0)
        & (d["interpolation_conf"] > 0)
    )
    common_count += int(ok.sum())
    for name, xy in [("interpolation", a), ("flow_crop", b)]:
        distance = np.linalg.norm(xy - gt, axis=-1) / np.maximum(scale[:, None], 1e-9)
        correct[name] += int((ok & (distance <= 0.2)).sum())
        errors[name] += float(distance[ok].sum())
with (p / "flow_attempts.csv").open(newline="") as f:
    attempts = list(csv.DictReader(f))
recovered = [r for r in attempts if r["status"] == "recovered"]
result = {
    "verified_baseline_predictions": len(missing),
    "verified_videos": len(files),
    "paired_video_changes": paired,
    "flow_recovered_frames": len(recovered),
    "flow_bbox_iou_ge_0_3": sum(
        float(r["gt_bbox_iou"]) >= 0.3 for r in recovered if r["gt_bbox_iou"]
    ),
    "common_visible_joints_in_eligible_frames": common_count,
    "common_pck20_pct": {k: 100 * v / common_count for k, v in correct.items()},
    "common_mean_normalized_error": {k: v / common_count for k, v in errors.items()},
    "note": "Same valid joints; GT IoU and pose accuracy are diagnostic proxies, not proof of target identity.",
}
(p / "verification_and_common_joints.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
