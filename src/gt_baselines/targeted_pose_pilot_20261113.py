"""Exploratory validation-only targeted repair; GT is used solely for evaluation."""

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
MAN = ROOT / "report/02112026_nckh/penn_audit_artifacts/video_manifest.csv"
SRC = ROOT / "outputs/gt_crop_yolo_bilstm_20261103/poses"
LABEL = ROOT / "data/raw/Penn_Action/labels"
FRAMES = ROOT / "data/raw/Penn_Action/frames"
OUT = ROOT / "outputs/targeted_pose_pilot_20261113"
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


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def valid(x):
    return np.isfinite(x).all(axis=-1) & (x > 0).all(axis=-1)


def correct(x, c):
    # A robust clip scale from predicted shoulder-to-hip distances only.
    torso = np.concatenate(
        [
            np.linalg.norm(
                x[valid(x[:, 0]) & valid(x[:, 6]), 0] - x[valid(x[:, 0]) & valid(x[:, 6]), 6],
                axis=1,
            ),
            np.linalg.norm(
                x[valid(x[:, 1]) & valid(x[:, 7]), 1] - x[valid(x[:, 1]) & valid(x[:, 7]), 7],
                axis=1,
            ),
        ]
    )
    torso = torso[np.isfinite(torso) & (torso > 1)]
    out = x.copy()
    kind = np.zeros(c.shape, np.uint8)
    if len(torso) < 5:
        return out, kind
    scale = np.median(torso)
    good = valid(x) & np.isfinite(c) & (c >= 0.5)
    suspect = (~valid(x)) | (~np.isfinite(c)) | (c < 0.3)
    for j in range(12):
        t = 0
        while t < len(x):
            if not suspect[t, j]:
                t += 1
                continue
            a = t
            while t < len(x) and suspect[t, j]:
                t += 1
            b = t - 1
            if (
                b - a + 1 <= 3
                and a > 0
                and t < len(x)
                and good[a - 1, j]
                and good[t, j]
                and np.linalg.norm(x[t, j] - x[a - 1, j]) <= 2 * scale
            ):
                for k in range(a, t):
                    out[k, j] = x[a - 1, j] + (x[t, j] - x[a - 1, j]) * ((k - a + 1) / (t - a + 1))
                    kind[k, j] = 1
        # Isolated spike; neighbors and target high-confidence, original sequence only.
        for t in range(1, len(x) - 1):
            if (
                good[t - 1 : t + 2, j].all()
                and np.linalg.norm(x[t + 1, j] - x[t - 1, j]) <= 2 * scale
            ):
                mid = (x[t - 1, j] + x[t + 1, j]) / 2
                if np.linalg.norm(x[t, j] - mid) > 0.75 * scale:
                    out[t, j] = mid
                    kind[t, j] = 2
    return out, kind


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [
        r
        for r in csv.DictReader(MAN.open(encoding="utf-8-sig", newline=""))
        if r["seed0_split"] == "validation"
    ]
    assert len(rows) == 253
    events = []
    videos = []
    examples = []
    totals = np.zeros(4, dtype=int)
    all_good = all_eligible = all_obs = all_good_corr = all_corr_obs = 0
    for r in rows:
        sid = r["id"]
        p = SRC / f"{sid}.npz"
        label = LABEL / f"{sid}.mat"
        assert sha(label) == r["label_sha256"]
        with np.load(p) as d:
            assert str(d["annotation_sha256"]) == r["label_sha256"]
            x = d["xy"][:, SWAP].astype(float)
            c = d["conf"][:, SWAP].astype(float)
            status = d["status"]
        mat = loadmat(label)
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], -1).astype(float)
        vis = mat["visibility"][:, 1:13] > 0
        bbox = np.full((len(x), 4), np.nan)
        n = min(len(x), len(mat["bbox"]))
        bbox[:n] = mat["bbox"][:n]
        wh = bbox[:, 2:] - bbox[:, :2]
        boxok = np.isfinite(bbox).all(axis=1) & (wh > 0).all(axis=1)
        diag = np.linalg.norm(wh, axis=1)
        eligible = vis & valid(gt) & boxok[:, None]
        obs = valid(x) & (status[:, None] == 3)
        y, kind = correct(x, c)
        obs2 = valid(y) & ((status[:, None] == 3) | (kind > 0))
        d0 = np.linalg.norm(x - gt, axis=2) / diag[:, None]
        d1 = np.linalg.norm(y - gt, axis=2) / diag[:, None]
        g0 = eligible & obs & (d0 <= 0.2)
        g1 = eligible & obs2 & (d1 <= 0.2)
        all_eligible += eligible.sum()
        all_obs += (eligible & obs).sum()
        all_corr_obs += (eligible & obs2).sum()
        all_good += g0.sum()
        all_good_corr += g1.sum()
        totals += np.bincount(status, minlength=4)
        videos.append(
            {
                "id": sid,
                "action": r["action"],
                "eligible": int(eligible.sum()),
                "good_raw": int(g0.sum()),
                "good_corrected": int(g1.sum()),
                "changed": int((kind > 0).sum()),
                "input_sha256": sha(p),
            }
        )
        for t, j in np.argwhere((kind > 0) & eligible):
            if obs[t, j]:
                before = float(d0[t, j])
            else:
                before = None
            after = float(d1[t, j])
            change = int(g1[t, j]) - int(g0[t, j])
            events.append(
                {
                    "id": sid,
                    "action": r["action"],
                    "frame": int(t + 1),
                    "joint": NAMES[j],
                    "kind": "low_conf_or_missing" if kind[t, j] == 1 else "spike",
                    "confidence": float(c[t, j]) if np.isfinite(c[t, j]) else None,
                    "before_norm": before,
                    "after_norm": after,
                    "pck_change": change,
                }
            )
            if before is not None:
                examples.append((sid, t, j, kind[t, j], before, after, float(c[t, j])))
    OUT.mkdir(parents=True)
    for name, data in [("per_video.csv", videos), ("changes_gt_eligible.csv", events)]:
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    rng = np.random.default_rng(20261113)
    delta = np.array(
        [
            (v["good_corrected"] - v["good_raw"]) / v["eligible"] * 100 if v["eligible"] else 0
            for v in videos
        ]
    )
    boot = rng.integers(0, len(videos), (10000, len(videos)))
    ev0 = [e for e in events if e["before_norm"] is not None]
    evmiss = [e for e in events if e["before_norm"] is None]
    summary = {
        "videos": len(rows),
        "status_counts": totals.tolist(),
        "eligible_joint_frames": int(all_eligible),
        "raw_observed": int(all_obs),
        "corrected_observed": int(all_corr_obs),
        "raw_pck20_missing_wrong_pct": float(all_good / all_eligible * 100),
        "corrected_pck20_missing_wrong_pct": float(all_good_corr / all_eligible * 100),
        "changed_joint_frames_all": sum(v["changed"] for v in videos),
        "changed_eligible": len(events),
        "changed_eligible_observed": len(ev0),
        "changed_eligible_missing": len(evmiss),
        "pck_improved": sum(e["pck_change"] == 1 for e in events),
        "pck_worsened": sum(e["pck_change"] == -1 for e in events),
        "mean_error_before_observed": float(np.mean([e["before_norm"] for e in ev0])),
        "mean_error_after_same": float(np.mean([e["after_norm"] for e in ev0])),
        "error_better_observed": sum(e["after_norm"] < e["before_norm"] for e in ev0),
        "error_worse_observed": sum(e["after_norm"] > e["before_norm"] for e in ev0),
        "per_video_pck_delta_mean_pp": float(delta.mean()),
        "per_video_delta_bootstrap95_pp": np.quantile(
            delta[boot].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "kind_counts": {
            k: sum(e["kind"] == k for e in events) for k in ("spike", "low_conf_or_missing")
        },
        "manifest_sha256": sha(MAN),
        "script_sha256": sha(Path(__file__)),
        "scope": "validation only; exploratory thresholds; GT crop and selection; fixed swap; uses future frames; no classifier evaluation; status 0 invalid bbox, 1 no detection, 2 no match, 3 matched",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Two extreme observed examples for visual inspection, selected using GT after analysis.
    examples.sort(key=lambda a: a[4] - a[5])
    chosen = [("worsened", examples[0]), ("improved", examples[-1])]
    for title, (sid, t, j, k, before, after, conf) in chosen:
        with np.load(SRC / f"{sid}.npz") as d:
            x = d["xy"][:, SWAP].astype(float)
            c = d["conf"][:, SWAP].astype(float)
        y, _ = correct(x, c)
        mat = loadmat(LABEL / f"{sid}.mat")
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], -1)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for ax, idx in zip(axs, [max(0, t - 1), t, min(len(x) - 1, t + 1)]):
            ax.imshow(plt.imread(FRAMES / sid / f"{idx + 1:06d}.jpg"))
            ax.set_title(f"Frame {idx + 1}")
            ax.axis("off")
            if valid(x[idx, j]):
                ax.scatter(*x[idx, j], c="red", s=70, marker="x", label="YOLO")
            if mat["visibility"][idx, j + 1] and valid(gt[idx, j]):
                ax.scatter(*gt[idx, j], c="lime", s=65, marker="+", label="GT")
            if idx == t and valid(y[idx, j]):
                ax.scatter(
                    *y[idx, j], c="cyan", s=50, marker="o", facecolors="none", label="repaired"
                )
        axs[1].legend(loc="lower right")
        fig.suptitle(
            f"{title}: video {sid}, {NAMES[j]}, confidence {conf:.2f}, normalized error {before:.3f} -> {after:.3f}"
        )
        fig.tight_layout()
        fig.savefig(OUT / f"case_{title}_{sid}_f{t + 1}.png", dpi=140)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
