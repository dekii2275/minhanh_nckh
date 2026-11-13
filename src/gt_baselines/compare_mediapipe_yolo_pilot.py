"""Small paired GT-crop pose pilot, MediaPipe image mode versus cached YOLO."""

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/mediapipe_runtime"))
import compare_person_selection as base  # noqa: E402
import cv2  # noqa: E402
import gt_crop_pose_experiment as crop  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

OUT = ROOT / "outputs/mediapipe_yolo_pilot_20261103"
MODEL = ROOT / "models/pose_landmarker_full.task"
JOINTS = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
HARD = ["0018", "0137", "0399", "1005", "1516"]
SWAP = np.arange(12) ^ 1


def choose():
    rows = [r for r in crop.rows() if r["seed0_split"] == "validation"]
    rng = np.random.default_rng(20261103)
    selected = []
    for action in base.ACTION_CLASSES:
        candidates = [r for r in rows if r["action"] == action]
        selected.append(candidates[int(rng.integers(len(candidates)))])
    byid = {r["id"]: r for r in rows}
    for sid in HARD:
        if sid not in {r["id"] for r in selected}:
            selected.append(byid[sid])
    return selected


def main():
    if OUT.exists():
        raise FileExistsError("Choose a new output directory")
    OUT.mkdir(parents=True)
    (OUT / "poses").mkdir()
    selections = choose()
    provenance = {
        "python": sys.executable,
        "mediapipe": mp.__version__,
        "mediapipe_model_sha256": base.sha(MODEL),
        "manifest_sha256": base.sha(base.MANIFEST),
        "script_sha256": base.sha(Path(__file__)),
        "yolo_source_sha256": base.sha(crop.OUT / "provenance.json"),
        "yolo_cache": "GT crop experiment, YOLO11n-pose GPU, confidence .25, gt box IoU >=.3",
        "mediapipe_delegate": "CPU",
        "mode": "IMAGE",
        "num_poses": 1,
        "detection_confidence": 0.5,
        "pose_presence_confidence": 0.5,
        "crop_margin_each_side": 0.25,
        "joint_mapping": JOINTS.tolist(),
        "joint_validity": "finite coordinates and confidence or visibility >= .5; same numerical threshold, scores not calibrated equally",
        "sample": "one pseudorandom validation video per class, plus explicitly selected hard cases; not representative",
        "seed": 20261103,
        "ids": [r["id"] for r in selections],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    start = time.time()
    video = []
    aggregate = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as detector:
        for k, row in enumerate(selections, 1):
            sid = row["id"]
            mat = sio.loadmat(base.LABELS / f"{sid}.mat")
            T = int(row["nframes"])
            bbox = np.asarray(mat["bbox"], float)
            y = np.load(crop.OUT / "poses" / f"{sid}.npz")
            yp = y["xy"]
            yc = y["conf"]
            mpxy = np.full((T, 12, 2), np.nan, np.float32)
            mc = np.zeros((T, 12), np.float32)
            for t in range(T):
                im = cv2.imread(str(base.FRAMES / sid / f"{t + 1:06d}.jpg"))
                if im is None:
                    raise ValueError("Cannot read " + sid)
                rect = (
                    crop.crop_rectangle(bbox[t], im.shape[1], im.shape[0])
                    if t < len(bbox)
                    else None
                )
                if rect is None:
                    continue
                x1, y1, x2, y2 = rect
                rgb = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if result.pose_landmarks:
                    points = result.pose_landmarks[0]
                    for j, idx in enumerate(JOINTS):
                        p = points[int(idx)]
                        mpxy[t, j] = [p.x * (x2 - x1) + x1, p.y * (y2 - y1) + y1]
                        mc[t, j] = p.visibility if p.visibility is not None else 0
            np.savez_compressed(
                OUT / "poses" / f"{sid}.npz",
                xy=mpxy,
                confidence=mc,
                annotation_sha256=row["label_sha256"],
            )
            gx = np.asarray(mat["x"][:, 1:13], float)
            gy = np.asarray(mat["y"][:, 1:13], float)
            gt = np.stack((gx, gy), axis=-1)
            vis = np.asarray(mat["visibility"][:, 1:13]) > 0
            diag = np.full(T, np.nan)
            n = min(T, len(bbox))
            wh = bbox[:n, 2:] - bbox[:n, :2]
            diag[:n] = np.where(
                np.isfinite(bbox[:n]).all(1) & (wh > 0).all(1), np.linalg.norm(wh, axis=1), np.nan
            )
            g = (
                vis
                & np.isfinite(gt).all(2)
                & (gt > 0).all(2)
                & np.isfinite(diag[:, None])
                & (diag[:, None] > 0)
            )
            scores = {}
            for name, coords, confidence in [
                ("mp_raw", mpxy, mc),
                ("mp_swap", mpxy[:, SWAP], mc[:, SWAP]),
                ("yolo_raw", yp, yc),
                ("yolo_swap", yp[:, SWAP], yc[:, SWAP]),
            ]:
                good = (
                    np.isfinite(coords).all(2)
                    & (coords > 0).all(2)
                    & np.isfinite(confidence)
                    & (confidence >= 0.5)
                )
                err = np.linalg.norm(coords.astype(float) - gt, axis=2)
                ok = g & good
                scores[name] = {"valid": ok, "error": err, "pck": err <= 0.2 * diag[:, None]}
            common = np.logical_and.reduce([g] + [s["valid"] for s in scores.values()])
            v = {
                "id": sid,
                "action": row["action"],
                "hard_selected": sid in HARD,
                "frames": T,
                "gt_joints": int(g.sum()),
                "common": int(common.sum()),
                "mp_detected_frames": int(np.isfinite(mpxy).all(2).any(1).sum()),
                "yolo_matched_frames": int((y["status"] == 3).sum()),
            }
            for name, s in scores.items():
                v[name + "_valid"] = int(s["valid"].sum())
                v[name + "_pck_all_numerator"] = int((s["pck"] & s["valid"]).sum())
                v[name + "_common_error_sum"] = float(s["error"][common].sum())
                v[name + "_common_pck_numerator"] = int((s["pck"] & common).sum())
            video.append(v)
            aggregate.append(scores)
            print(
                f"{k}/{len(selections)} {sid} {row['action']} mp={v['mp_detected_frames']}/{T} yolo={v['yolo_matched_frames']}/{T} elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    totals = {
        "videos": len(video),
        "frames": sum(v["frames"] for v in video),
        "gt_visible_joints": sum(v["gt_joints"] for v in video),
        "common_joints": sum(v["common"] for v in video),
        "mp_detected_frames": sum(v["mp_detected_frames"] for v in video),
        "yolo_matched_frames": sum(v["yolo_matched_frames"] for v in video),
        "seconds": time.time() - start,
        "methods": {},
    }
    for name in ("mp_raw", "mp_swap", "yolo_raw", "yolo_swap"):
        valid = sum(v[name + "_valid"] for v in video)
        total = totals["gt_visible_joints"]
        common = totals["common_joints"]
        totals["methods"][name] = {
            "joint_coverage_pct": 100 * valid / total,
            "pck_missing_wrong_pct": 100
            * sum(v[name + "_pck_all_numerator"] for v in video)
            / total,
            "mean_error_px_on_common": sum(v[name + "_common_error_sum"] for v in video) / common,
            "pck_on_common_pct": 100
            * sum(v[name + "_common_pck_numerator"] for v in video)
            / common,
        }
    (OUT / "summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    with (OUT / "per_video.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(video[0]))
        w.writeheader()
        w.writerows(video)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
