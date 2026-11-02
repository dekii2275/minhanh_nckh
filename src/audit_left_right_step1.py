"""Penn train-only correspondence pilot. No calibration changes or tuning.
Run with the penn_pose environment; outputs are isolated from prior experiments.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from ultralytics import YOLO

from skeleton_calibrator import FullCalibrationPipeline, LeftRightContinuityCalibrator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/alignment_step1"
DATA = ROOT / "data/raw/Penn_Action"
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
SWAP = np.arange(12).reshape(6, 2)[:, ::-1].ravel()
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


def select():
    groups = defaultdict(list)
    for p in sorted((DATA / "labels").glob("*.mat")):
        m = sio.loadmat(p)
        if int(m["train"].item()) != 1:
            continue
        groups[str(m["action"][0])].append(
            {
                "id": p.stem,
                "action": str(m["action"][0]),
                "pose": str(m["pose"][0]),
                "frames": int(m["nframes"].item()),
                "official_train": 1,
            }
        )
    rng = random.Random(20261029)
    chosen = []
    for act, items in sorted(groups.items()):
        first = rng.choice(items)
        second = rng.choice(
            [x for x in items if x["pose"] != first["pose"]]
            or [x for x in items if x["id"] != first["id"]]
        )
        chosen += [first, second]
    return chosen


def infer(sample):
    torch.set_num_threads(4)
    model = YOLO(str(ROOT / "models/yolo11n-pose.pt"))
    for item in sample:
        target = OUT / f"{item['id']}.npz"
        if target.exists():
            continue
        paths = sorted((DATA / "frames" / item["id"]).glob("*.jpg"))
        assert len(paths) == item["frames"]
        xy = []
        cf = []
        n = []
        box = []
        for start in range(0, len(paths), 16):
            results = model.predict(
                [str(p) for p in paths[start : start + 16]],
                imgsz=640,
                conf=0.25,
                device="cpu",
                verbose=False,
            )
            for r in results:
                n.append(len(r.boxes))
                if len(r.boxes):
                    k = int(r.boxes.conf.argmax())
                    xy.append(r.keypoints.xy[k].cpu().numpy()[5:])
                    cf.append(r.keypoints.conf[k].cpu().numpy()[5:])
                    box.append(float(r.boxes.conf[k]))
                else:
                    xy.append(np.full((12, 2), np.nan))
                    cf.append(np.full(12, np.nan))
                    box.append(np.nan)
        np.savez_compressed(target, xy=xy, conf=cf, n=n, box=box)
        print("INFER", item["id"], item["action"], len(paths), flush=True)


def draw(im, pts, valid, gt=False):
    # Same side colors for GT and predictions; GT thin line + circles, YOLO thick + crosses.
    for a, b in EDGES:
        if valid[a] and valid[b]:
            color = (255, 180, 40) if a % 2 == 0 else (255, 60, 230)
            cv2.line(
                im,
                tuple(pts[a].astype(int)),
                tuple(pts[b].astype(int)),
                color,
                1 if gt else 2,
                cv2.LINE_AA,
            )
    for j, p in enumerate(pts):
        if valid[j]:
            q = tuple(p.astype(int))
            color = (255, 180, 40) if j % 2 == 0 else (255, 60, 230)
            if gt:
                cv2.circle(im, q, 5, color, 1, cv2.LINE_AA)
            else:
                cv2.drawMarker(im, q, color, cv2.MARKER_TILTED_CROSS, 8, 2)
            cv2.putText(
                im,
                ("L" if j % 2 == 0 else "R") + str(j // 2),
                (q[0] + 3, q[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
    return im


def analyze(sample):
    rows = []
    frame_rows = []
    for item in sample:
        sid = item["id"]
        m = sio.loadmat(DATA / "labels" / f"{sid}.mat")
        z = np.load(OUT / f"{sid}.npz")
        gt = np.stack([m["x"][:, 1:], m["y"][:, 1:]], -1).astype(float)
        vis = m["visibility"][:, 1:].astype(bool)
        xy = z["xy"]
        conf = z["conf"]
        T = len(xy)
        d = {
            f"pred_{name}_{s}": (xy[:, j, k] if s != "conf" else conf[:, j])
            for j, name in enumerate(NAMES)
            for k, s in enumerate(["x", "y", "conf"])
        }
        df = pd.DataFrame(d)
        raw_delta = float(np.nanmedian(xy[:, 0, 0] - xy[:, 1, 0]))
        rawswap = raw_delta > 0
        # Exact current production input to stage 4, retaining default stages 1-3.
        pipe = FullCalibrationPipeline()
        pre = pipe.stage3_smooth.calibrate_sequence(
            pipe.stage2_bone.calibrate_sequence(pipe.stage1_interp.calibrate_sequence(df))
        )
        delta = float(np.nanmedian(pre.pred_left_shoulder_x - pre.pred_right_shoulder_x))
        swap = delta > 0
        actual = LeftRightContinuityCalibrator().calibrate_sequence(pre)
        for j, name in enumerate(NAMES):
            src = NAMES[SWAP[j]] if swap else name
            assert np.allclose(actual[f"pred_{name}_x"], pre[f"pred_{src}_x"], equal_nan=True)
        stats = []
        for t in range(T):
            votes = []
            ei = []
            es = []
            for a in range(0, 12, 2):
                b = a + 1
                if not (
                    vis[t, a]
                    and vis[t, b]
                    and (conf[t, [a, b]] >= 0.5).all()
                    and np.isfinite(xy[t, [a, b]]).all()
                    and (xy[t, [a, b]] > 0).all()
                ):
                    continue
                sep = np.linalg.norm(gt[t, a] - gt[t, b])
                psep = np.linalg.norm(xy[t, a] - xy[t, b])
                scale = max(1.0, np.linalg.norm((gt[t, 0] + gt[t, 1] - gt[t, 6] - gt[t, 7]) / 2))
                if min(sep, psep) < max(5.0, 0.15 * scale):
                    continue
                i = np.linalg.norm(xy[t, [a, b]] - gt[t, [a, b]], axis=1).mean()
                s = np.linalg.norm(xy[t, [b, a]] - gt[t, [a, b]], axis=1).mean()
                ei.append(i)
                es.append(s)
                if abs(i - s) > max(3.0, 0.15 * sep):
                    votes.append("I" if i < s else "S")
            ni = votes.count("I")
            ns = votes.count("S")
            vote = "I" if ni >= 2 and ns == 0 else "S" if ns >= 2 and ni == 0 else "?"
            r = {
                "id": sid,
                "frame": t + 1,
                "vote": vote,
                "I_pairs": ni,
                "S_pairs": ns,
                "identity_error": float(np.mean(ei)) if ei else None,
                "swap_error": float(np.mean(es)) if es else None,
            }
            frame_rows.append(r)
            stats.append(r)
        iv = sum(x["vote"] == "I" for x in stats)
        sv = sum(x["vote"] == "S" for x in stats)
        rows.append(
            dict(
                item,
                heuristic_swap=bool(swap),
                raw_heuristic_swap=bool(rawswap),
                median_dx=delta,
                raw_median_dx=raw_delta,
                I_frames=iv,
                S_frames=sv,
                uncertain_frames=T - iv - sv,
                multi_person_frames=int((z["n"] > 1).sum()),
                no_detection_frames=int((z["n"] == 0).sum()),
            )
        )
        # Uniform samples plus strongest opposing evidence if present.
        frames = list(np.linspace(0, T - 1, 5).round().astype(int))
        for v in ["I", "S"]:
            candidates = [k for k, r in enumerate(stats) if r["vote"] == v]
            if candidates:
                best = max(
                    candidates,
                    key=lambda k: abs(stats[k]["identity_error"] - stats[k]["swap_error"]),
                )
                if best not in frames:
                    frames.append(best)
        frames = sorted(set(frames))
        tiles = []
        for t in frames:
            orig = cv2.imread(str(DATA / "frames" / sid / f"{t + 1:06d}.jpg"))
            panels = []
            for title, p, pc in [
                ("GT", gt[t], None),
                ("Identity", xy[t], conf[t]),
                ("Fixed Swap", xy[t, SWAP], conf[t, SWAP]),
                (
                    "Heuristic " + ("S" if swap else "I"),
                    xy[t, SWAP] if swap else xy[t],
                    conf[t, SWAP] if swap else conf[t],
                ),
            ]:
                im = orig.copy()
                valid = (
                    vis[t] & np.isfinite(p).all(1)
                    if pc is None
                    else (pc >= 0.5) & np.isfinite(p).all(1) & (p > 0).all(1)
                )
                if pc is not None:
                    draw(im, gt[t], vis[t], True)
                draw(im, p, valid, pc is None)
                im = cv2.resize(im, (360, 270))
                banner = np.zeros((35, 360, 3), np.uint8)
                cv2.putText(
                    banner,
                    f"{sid} f{t + 1} {title}",
                    (5, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                panels.append(np.vstack([banner, im]))
            tiles.append(np.hstack(panels))
        cv2.imwrite(str(OUT / "figures" / f"{sid}.jpg"), np.vstack(tiles))
    pd.DataFrame(rows).to_csv(OUT / "diagnostics.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(OUT / "frame_diagnostics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    # Overview: each row is a clip, five evenly spaced GT/identity overlays.
    for g in range(5):
        lines = []
        for item in sample[g * 6 : g * 6 + 6]:
            sid = item["id"]
            m = sio.loadmat(DATA / "labels" / f"{sid}.mat")
            z = np.load(OUT / f"{sid}.npz")
            panels = []
            for t in np.linspace(0, item["frames"] - 1, 5).round().astype(int):
                im = cv2.imread(str(DATA / "frames" / sid / f"{t + 1:06d}.jpg"))
                gt = np.stack([m["x"][t, 1:], m["y"][t, 1:]], -1)
                draw(im, gt, m["visibility"][t, 1:].astype(bool), True)
                draw(im, z["xy"][t], (z["conf"][t] >= 0.5) & np.isfinite(z["xy"][t]).all(1), False)
                im = cv2.resize(im, (320, 240))
                ban = np.zeros((30, 320, 3), np.uint8)
                cv2.putText(
                    ban,
                    f"{sid} {item['pose']} f{t + 1}",
                    (3, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                panels.append(np.vstack([ban, im]))
            lines.append(np.hstack(panels))
        if lines:
            cv2.imwrite(str(OUT / "figures" / f"overview_{g + 1}.jpg"), np.vstack(lines))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "figures").mkdir(exist_ok=True)
    sample = select()
    assert len(sample) == 30 and len({x["id"] for x in sample}) == 30
    assert all(x["official_train"] == 1 for x in sample)
    (OUT / "sample.json").write_text(json.dumps(sample, indent=2))
    # Freeze sampled IDs before predictions, independent of GT errors and outcomes.
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "seed": 20261029,
                "sampling": "2 per action; second from different pose; official train==1 only",
                "model_sha256": hashlib.sha256(
                    (ROOT / "models/yolo11n-pose.pt").read_bytes()
                ).hexdigest(),
                "ultralytics": __import__("ultralytics").__version__,
                "calibrator_sha256": hashlib.sha256(
                    (ROOT / "src/skeleton_calibrator.py").read_bytes()
                ).hexdigest(),
                "torch": torch.__version__,
                "device": "cpu",
                "imgsz": 640,
                "conf": 0.25,
                "selection": "highest person box confidence, no GT",
                "diagnostic_conf": 0.5,
            },
            indent=2,
        )
    )
    infer(sample)
    analyze(sample)
