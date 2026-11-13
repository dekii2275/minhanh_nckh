"""EDA of cached GT-crop YOLO keypoint confidence on Penn validation."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "report/02112026_nckh/penn_audit_artifacts/video_manifest.csv"
SOURCE = ROOT / "outputs/gt_crop_yolo_bilstm_20261103"
LABELS = ROOT / "data/raw/Penn_Action/labels"
OUT = ROOT / "outputs/eda_joint_confidence_validation_20261104"
NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
SWAP = np.arange(12) ^ 1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig", newline="")))
    samples = [r for r in rows if r["seed0_split"] == "validation"]
    assert len(samples) == 253
    records = []
    statuses = np.zeros(4, dtype=np.int64)
    for row in samples:
        sid = row["id"]
        label = LABELS / f"{sid}.mat"
        assert sha(label) == row["label_sha256"], sid
        with np.load(SOURCE / "poses" / f"{sid}.npz") as d:
            assert str(d["annotation_sha256"]) == row["label_sha256"], sid
            xy, conf, status = d["xy"][:, SWAP], d["conf"][:, SWAP], d["status"]
        mat = loadmat(label)
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], axis=-1)
        vis = mat["visibility"][:, 1:13] > 0
        T = len(status)
        assert len(gt) == T == int(row["nframes"]) and conf.shape == (T, 12), sid
        bbox = np.full((T, 4), np.nan)
        n = min(T, len(mat["bbox"]))
        bbox[:n] = mat["bbox"][:n]
        wh = bbox[:, 2:] - bbox[:, :2]
        boxok = np.isfinite(bbox).all(axis=1) & (wh > 0).all(axis=1)
        diag = np.linalg.norm(wh, axis=1)
        matched = status == 3
        statuses += np.bincount(status, minlength=4)
        observed = np.isfinite(xy).all(axis=2) & (xy > 0).all(axis=2) & np.isfinite(conf)
        eligible = vis & np.isfinite(gt).all(axis=2) & (gt > 0).all(axis=2) & boxok[:, None]
        dist = np.linalg.norm(xy - gt, axis=2) / diag[:, None]
        records.append(
            dict(
                id=sid,
                action=row["action"],
                conf=conf,
                observed=observed,
                matched=matched,
                eligible=eligible,
                dist=dist,
            )
        )
    OUT.mkdir(parents=True)
    by_joint, by_video, bins = [], [], []
    for j, name in enumerate(NAMES):
        all_scores = np.concatenate([r["conf"][r["matched"], j] for r in records])
        observed = sum(int((r["matched"] & r["observed"][:, j]).sum()) for r in records)
        total = sum(len(r["matched"]) for r in records)
        gt_total = sum(int(r["eligible"][:, j].sum()) for r in records)
        gt_valid = sum(
            int((r["eligible"][:, j] & r["matched"] & r["observed"][:, j]).sum()) for r in records
        )
        good = sum(
            int(
                (
                    r["eligible"][:, j]
                    & r["matched"]
                    & r["observed"][:, j]
                    & (r["dist"][:, j] <= 0.2)
                ).sum()
            )
            for r in records
        )
        by_joint.append(
            dict(
                joint=name,
                matched_frames=len(all_scores),
                mean_conf=float(np.mean(all_scores)),
                median_conf=float(np.median(all_scores)),
                p10_conf=float(np.quantile(all_scores, 0.1)),
                below_03_pct=float(np.mean(all_scores < 0.3) * 100),
                below_05_pct=float(np.mean(all_scores < 0.5) * 100),
                observed_pct_matched=observed / len(all_scores) * 100,
                observed_pct_all_frames=observed / total * 100,
                gt_eligible=gt_total,
                gt_observed=gt_valid,
                gt_coverage_pct=gt_valid / gt_total * 100,
                pck20_observed_pct=good / gt_valid * 100 if gt_valid else None,
                pck20_missing_incorrect_pct=good / gt_total * 100,
            )
        )
        for r in records:
            m = r["matched"]
            c = r["conf"][m, j]
            by_video.append(
                dict(
                    id=r["id"],
                    action=r["action"],
                    joint=name,
                    frames=len(m),
                    matched=int(m.sum()),
                    mean_conf=float(c.mean()) if len(c) else "",
                    below_03_pct=float((c < 0.3).mean() * 100) if len(c) else "",
                    observed=int((m & r["observed"][:, j]).sum()),
                )
            )
        for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
            n = goodbin = 0
            for r in records:
                c = r["conf"][:, j]
                mask = (
                    r["eligible"][:, j] & r["matched"] & r["observed"][:, j] & (c >= lo) & (c < hi)
                )
                n += int(mask.sum())
                goodbin += int((mask & (r["dist"][:, j] <= 0.2)).sum())
            bins.append(
                dict(
                    joint=name, conf_min=lo, conf_max=hi, n=n, pck20=goodbin / n * 100 if n else ""
                )
            )
    write_csv(OUT / "per_joint.csv", by_joint)
    write_csv(OUT / "per_video_joint.csv", by_video)
    write_csv(OUT / "confidence_bins.csv", bins)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    pos = np.arange(12)
    ax[0].barh(pos, [x["below_03_pct"] for x in by_joint], color="#e39839")
    ax[0].set_xlabel("Confidence < 0.30 (% of matched frames)")
    ax[1].barh(pos, [x["pck20_missing_incorrect_pct"] for x in by_joint], color="#4985b6")
    ax[1].set_xlabel("PCK@0.2, missing counted wrong (% GT-eligible)")
    for a in ax:
        a.set_yticks(pos, NAMES)
        a.invert_yaxis()
        a.grid(axis="x", alpha=0.25)
    ax[1].set_xlim(0, 100)
    fig.suptitle("Penn Action validation: YOLO GT-crop confidence and pose coverage")
    fig.tight_layout()
    fig.savefig(OUT / "joint_confidence_and_pck.png", dpi=170)
    plt.close(fig)
    metadata = {
        "videos": len(samples),
        "frames": int(statuses.sum()),
        "frame_status": dict(
            zip(["invalid_bbox", "no_detection", "no_match", "matched"], map(int, statuses))
        ),
        "scope": "seed0 validation; cached YOLO11n-Pose; GT bbox crop + IoU selection; fixed left/right swap",
        "confidence_denominator": "all matched frames, including missing coordinates; missing unmatched frames excluded",
        "pck_denominator": "GT visible, positive finite GT coordinates and valid bbox; missing pose counted incorrect in all-eligible PCK",
        "pck_threshold": "0.2 * GT bbox diagonal; direct pixel comparison",
        "caution": "exploratory; frame-weighted; GT-assisted selection; fixed swap not valid for all videos; confidence not calibrated probability; not independent test",
        "manifest_sha256": sha(MANIFEST),
        "source_provenance_sha256": sha(SOURCE / "provenance.json"),
        "script_sha256": sha(Path(__file__)),
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
