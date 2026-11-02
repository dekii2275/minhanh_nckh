"""Audit raw YOLO/GT left-right correspondence on the frozen Penn validation IDs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics_config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "pose_view_audit" / "mpl"))
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = ROOT / "data/raw/Penn_Action"
OUT = ROOT / "outputs/pose_view_audit"
CACHE = OUT / "raw_predictions"
MODEL = ROOT / "models/yolo11n-pose.pt"
MANIFEST = ROOT / "outputs/alignment_step2/validation_manifest.json"
SWAP = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    assert len(manifest) == len({v["id"] for v in manifest}) == 253
    device = "0" if torch.cuda.is_available() else "cpu"
    info = {
        "model_sha256": digest(MODEL),
        "manifest_sha256": digest(MANIFEST),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "device": device,
        "imgsz": 640,
        "box_confidence": 0.25,
        "selection": "highest box confidence",
        "batch_size": 16,
        "joint_mask": "GT visible; both raw and swapped YOLO coordinates finite, positive and confidence > 0",
        "target_filter": "median of all finite positive detected body keypoints inside GT bbox (xyxy), inclusive",
        "note": "GPU/library versions may differ from historical CPU inference; independently recomputed.",
    }
    fp = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    (OUT / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf8")
    model = None
    rows, frames = [], []
    for index, item in enumerate(manifest):
        sid = item["id"]
        path = DATA / "labels" / (sid + ".mat")
        assert digest(path) == item["annotation_sha256"], f"Annotation mismatch {sid}"
        mat = sio.loadmat(path)
        assert int(mat["train"].item()) == 1
        gt = np.stack([mat["x"][:, 1:], mat["y"][:, 1:]], axis=-1)
        vis = mat["visibility"][:, 1:] == 1
        bbox = mat["bbox"]
        T = len(gt)
        assert gt.shape == (item["frames"], 12, 2) and bbox.shape == (T, 4)
        cache = CACHE / (sid + ".npz")
        if cache.exists():
            z = np.load(cache)
            if str(z["fingerprint"].item()) != fp:
                raise ValueError(f"Cache fingerprint mismatch: {cache}")
            xy, conf = z["xy"], z["conf"]
        else:
            if model is None:
                model = YOLO(str(MODEL))
            image_paths = [DATA / "frames" / sid / f"{t + 1:06d}.jpg" for t in range(T)]
            assert all(p.is_file() for p in image_paths), sid
            xy, conf = [], []
            for start in range(0, T, 16):
                result = model.predict(
                    [str(p) for p in image_paths[start : start + 16]],
                    imgsz=640,
                    conf=0.25,
                    device=device,
                    verbose=False,
                )
                for r in result:
                    if len(r.boxes):
                        i = int(r.boxes.conf.argmax())
                        xy.append(r.keypoints.xy[i].cpu().numpy()[5:17])
                        conf.append(r.keypoints.conf[i].cpu().numpy()[5:17])
                    else:
                        xy.append(np.full((12, 2), np.nan))
                        conf.append(np.full(12, np.nan))
            xy, conf = np.asarray(xy), np.asarray(conf)
            np.savez_compressed(cache, xy=xy, conf=conf, fingerprint=np.array(fp))
        assert xy.shape == (T, 12, 2) and conf.shape == (T, 12)
        valid_pred = (
            np.isfinite(xy).all(axis=2) & (xy > 0).all(axis=2) & np.isfinite(conf) & (conf > 0)
        )
        ctr = np.full((T, 2), np.nan)
        for t in range(T):
            if valid_pred[t].any():
                ctr[t] = np.median(xy[t, valid_pred[t]], axis=0)
        bbox_ok = (
            np.isfinite(bbox).all(axis=1) & (bbox[:, 0] <= bbox[:, 2]) & (bbox[:, 1] <= bbox[:, 3])
        )
        target = (
            bbox_ok
            & (ctr[:, 0] >= bbox[:, 0])
            & (ctr[:, 0] <= bbox[:, 2])
            & (ctr[:, 1] >= bbox[:, 1])
            & (ctr[:, 1] <= bbox[:, 3])
        )
        valid_gt = vis & np.isfinite(gt).all(axis=2)
        common = valid_gt & valid_pred & valid_pred[:, SWAP] & target[:, None]
        dr = np.linalg.norm(xy - gt, axis=2)
        ds = np.linalg.norm(xy[:, SWAP] - gt, axis=2)
        count = common.sum(axis=1)
        good = count > 0
        raw = np.divide(
            np.where(common, dr, 0).sum(axis=1), count, out=np.full(T, np.nan), where=good
        )
        swap = np.divide(
            np.where(common, ds, 0).sum(axis=1), count, out=np.full(T, np.nan), where=good
        )
        pose = str(mat["pose"][0]).strip().lower()
        rows.append(
            {
                "id": sid,
                "action": item["action"],
                "pose": pose,
                "total_frames": T,
                "target_frames": int(target.sum()),
                "valid_frames": int(good.sum()),
                "valid_joint_observations": int(count.sum()),
                "d_raw": float(np.nanmean(raw)) if good.any() else np.nan,
                "d_swap": float(np.nanmean(swap)) if good.any() else np.nan,
            }
        )
        frames.extend(
            {
                "id": sid,
                "pose": pose,
                "frame": t + 1,
                "target": bool(target[t]),
                "valid_joints": int(count[t]),
                "d_raw": raw[t],
                "d_swap": swap[t],
            }
            for t in range(T)
        )
        print(f"{index + 1:3d}/253 {sid} {pose} valid={good.sum()}/{T}", flush=True)
    clips = pd.DataFrame(rows)
    clips["decision"] = np.select(
        [clips.d_swap < clips.d_raw, clips.d_raw < clips.d_swap, clips.d_raw.isna()],
        ["swap", "raw", "no_valid"],
        default="tie",
    )
    clips.to_csv(OUT / "per_clip.csv", index=False)
    pd.DataFrame(frames).to_csv(OUT / "per_frame.csv", index=False)
    groups = []
    for pose, sub in clips.groupby("pose"):
        usable = sub[sub.decision != "no_valid"]
        groups.append(
            {
                "pose": pose,
                "clips_total": len(sub),
                "clips_valid": len(usable),
                "frames_total": int(sub.total_frames.sum()),
                "frames_valid": int(sub.valid_frames.sum()),
                "mean_d_raw_px": usable.d_raw.mean(),
                "mean_d_swap_px": usable.d_swap.mean(),
                "swap_better_n": int((usable.decision == "swap").sum()),
                "raw_better_n": int((usable.decision == "raw").sum()),
                "tie_n": int((usable.decision == "tie").sum()),
                "swap_better_pct": 100 * (usable.decision == "swap").mean(),
                "raw_better_pct": 100 * (usable.decision == "raw").mean(),
            }
        )
    summary = pd.DataFrame(groups)
    summary.to_csv(OUT / "summary_by_pose.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
