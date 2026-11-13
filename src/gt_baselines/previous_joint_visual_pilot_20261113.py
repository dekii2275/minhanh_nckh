"""Replace flagged YOLO joints by the previous original frame; evaluate and visualize."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
from targeted_pose_pilot_20261113 import FRAMES, LABEL, MAN, NAMES, ROOT, SRC, SWAP, sha, valid

OUT = ROOT / "outputs/previous_joint_visual_pilot_20261113"
INPUT = ROOT / "outputs/unusual_joint_motion_validation_20261113"
EDGES = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
    (6, 8),
    (8, 10),
    (7, 9),
    (9, 11),
]
CASES = [("1625", 89, 97), ("0425", 13, 21), ("1534", 31, 39)]


def render(im, xy, conf, gt, vis, bbox, flags, title):
    img = im.copy()
    h, w = img.shape[:2]
    if np.isfinite(bbox).all() and (bbox[2:] > bbox[:2]).all():
        b = bbox.astype(int)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 200, 0), 1)
    for a, b in EDGES:
        if vis[a] and vis[b] and valid(gt[a]) and valid(gt[b]):
            cv2.line(
                img, tuple(gt[a].astype(int)), tuple(gt[b].astype(int)), (0, 210, 0), 2, cv2.LINE_AA
            )
        if conf[a] >= 0.3 and conf[b] >= 0.3 and valid(xy[a]) and valid(xy[b]):
            cv2.line(
                img, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 0, 255), 2, cv2.LINE_AA
            )
    for j in range(12):
        if valid(xy[j]) and conf[j] >= 0.3:
            cv2.circle(img, tuple(xy[j].astype(int)), 3, (0, 0, 255), -1, cv2.LINE_AA)
        if flags[j] and valid(xy[j]):
            cv2.circle(img, tuple(xy[j].astype(int)), 7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(
        img, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [
        r
        for r in csv.DictReader(MAN.open(encoding="utf-8-sig", newline=""))
        if r["seed0_split"] == "validation"
    ]
    assert len(rows) == 253
    events = list(
        csv.DictReader((INPUT / "joint_events.csv").open(encoding="utf-8-sig", newline=""))
    )
    source = json.loads((INPUT / "summary.json").read_text(encoding="utf-8"))
    assert len(events) == 606
    indexed = defaultdict(list)
    for e in events:
        indexed[e["id"]].append(e)
    records = []
    videos = []
    all_good0 = all_good1 = eligible_n = 0
    visuals = {}
    for r in rows:
        sid = r["id"]
        p = SRC / f"{sid}.npz"
        assert sha(p) == source["pose_hashes"][sid]
        assert sha(LABEL / f"{sid}.mat") == r["label_sha256"]
        with np.load(p) as d:
            assert str(d["annotation_sha256"]) == r["label_sha256"]
            x = d["xy"][:, SWAP].astype(float)
            conf = d["conf"][:, SWAP]
            status = d["status"]
        flags = np.zeros(conf.shape, bool)
        y = x.copy()
        for e in indexed[sid]:
            t = int(e["frame_current"]) - 1
            j = NAMES.index(e["joint"])
            assert t >= 1 and valid(x[t, j]) and valid(x[t - 1, j])
            assert status[t] == 3 and status[t - 1] == 3
            flags[t, j] = True
            y[t, j] = x[t - 1, j]  # previous ORIGINAL point; not recursive
        assert np.array_equal(y[~flags], x[~flags], equal_nan=True)
        mat = loadmat(LABEL / f"{sid}.mat")
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], -1).astype(float)
        vis = mat["visibility"][:, 1:13] > 0
        bbox = np.full((len(x), 4), np.nan)
        n = min(len(x), len(mat["bbox"]))
        bbox[:n] = mat["bbox"][:n]
        wh = bbox[:, 2:] - bbox[:, :2]
        boxok = np.isfinite(bbox).all(axis=1) & (wh > 0).all(axis=1)
        diag = np.linalg.norm(wh, axis=1)
        eligible = vis & valid(gt) & boxok[:, None]
        observed = valid(x) & (status[:, None] == 3)
        d0 = np.linalg.norm(x - gt, axis=2) / diag[:, None]
        d1 = np.linalg.norm(y - gt, axis=2) / diag[:, None]
        good0 = eligible & observed & (d0 <= 0.2)
        good1 = eligible & observed & (d1 <= 0.2)
        eligible_n += int(eligible.sum())
        all_good0 += int(good0.sum())
        all_good1 += int(good1.sum())
        for t, j in np.argwhere(flags):
            has = bool(eligible[t, j])
            before = float(d0[t, j]) if has else None
            after = float(d1[t, j]) if has else None
            records.append(
                {
                    "id": sid,
                    "action": r["action"],
                    "frame": int(t + 1),
                    "joint": NAMES[j],
                    "gt_eligible": int(has),
                    "error_before_normalized": before,
                    "error_after_normalized": after,
                    "pck_change": int(good1[t, j]) - int(good0[t, j]) if has else None,
                    "confidence_current": float(conf[t, j]),
                    "confidence_previous": float(conf[t - 1, j]),
                }
            )
        videos.append(
            {
                "id": sid,
                "action": r["action"],
                "changed_joint_frames": int(flags.sum()),
                "changed_frames": int(flags.any(axis=1).sum()),
                "gt_eligible": int(eligible.sum()),
                "good_before": int(good0.sum()),
                "good_after": int(good1.sum()),
                "pose_sha256": sha(p),
            }
        )
        if sid in [c[0] for c in CASES]:
            visuals[sid] = (x, y, conf, flags, gt, vis, bbox)
    assert len(records) == 606 and sum(v["changed_frames"] for v in videos) == 295
    OUT.mkdir(parents=True)
    for name, data in [("events.csv", records), ("per_video.csv", videos)]:
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    known = [r for r in records if r["gt_eligible"]]
    ds = np.array([r["error_after_normalized"] - r["error_before_normalized"] for r in known])
    delta = np.array(
        [
            (v["good_after"] - v["good_before"]) / v["gt_eligible"] * 100 if v["gt_eligible"] else 0
            for v in videos
        ]
    )
    rng = np.random.default_rng(20261113)
    boot = rng.integers(0, len(videos), (10000, len(videos)))
    summary = {
        "videos": len(rows),
        "flagged_videos": sum(v["changed_frames"] > 0 for v in videos),
        "changed_frames": sum(v["changed_frames"] for v in videos),
        "changed_joint_frames": len(records),
        "changed_gt_eligible": len(known),
        "error_closer_to_gt": int((ds < 0).sum()),
        "error_farther_from_gt": int((ds > 0).sum()),
        "same_error": int((ds == 0).sum()),
        "mean_error_before": float(np.mean([r["error_before_normalized"] for r in known])),
        "mean_error_after": float(np.mean([r["error_after_normalized"] for r in known])),
        "pck_flip_improved": sum(r["pck_change"] == 1 for r in known),
        "pck_flip_worsened": sum(r["pck_change"] == -1 for r in known),
        "all_eligible": eligible_n,
        "pck20_before_pct": all_good0 / eligible_n * 100,
        "pck20_after_pct": all_good1 / eligible_n * 100,
        "video_mean_pck_delta_pp": float(delta.mean()),
        "video_bootstrap95_pp": np.quantile(delta[boot].mean(1), [0.025, 0.975]).tolist(),
        "rule": "for flagged joint copy previous ORIGINAL frame joint; flags compare original adjacent frames; confidence unchanged; missing not filled",
        "caution": "exploratory validation only; GT assisted crop/matching; fixed swap; clip-wide scale offline; confidence >=.5 in two frames; GT evaluation is not perfect; PCK misses dynamics and recognition",
        "events_sha256": sha(INPUT / "joint_events.csv"),
        "manifest_sha256": sha(MAN),
        "script_sha256": sha(Path(__file__)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for sid, start, end in CASES:
        x, y, c, flags, gt, vis, bbox = visuals[sid]
        frames = []
        for frame in range(start, end + 1):
            t = frame - 1
            im = cv2.imread(str(FRAMES / sid / f"{frame:06d}.jpg"))
            assert im is not None
            left = render(
                im,
                x[t],
                c[t],
                gt[t],
                vis[t],
                bbox[t],
                flags[t],
                f"{sid} frame {frame} | BEFORE: YOLO",
            )
            right = render(
                im,
                y[t],
                c[t],
                gt[t],
                vis[t],
                bbox[t],
                flags[t],
                f"{sid} frame {frame} | AFTER: previous joint",
            )
            canvas = np.hstack((left, right))
            h, w = canvas.shape[:2]
            canvas = cv2.copyMakeBorder(canvas, 0, 28, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(
                canvas,
                f"GT green | YOLO red | cyan ring = changed joint: {int(flags[t].sum())}",
                (5, h + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            frames.append(
                Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert(
                    "P", palette=Image.Palette.ADAPTIVE, colors=128
                )
            )
        frames[0].save(
            OUT / f"case_{sid}_{start}_{end}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
            optimize=False,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
