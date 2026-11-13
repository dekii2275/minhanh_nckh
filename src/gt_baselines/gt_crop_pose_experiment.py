"""GT bbox crop -> YOLO pose -> BiLSTM attention. Exploratory train/validation only."""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics_config"))
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

sys.path.insert(0, str(ROOT / "src/gt_baselines"))
import compare_person_selection as base  # noqa: E402
import train_gt_baselines as trainer  # noqa: E402

OUT = ROOT / "outputs/gt_crop_yolo_bilstm_20261103"
MARGIN = 0.25
SWAP = np.arange(12) ^ 1


def crop_rectangle(box, width, height, margin=MARGIN):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    dx, dy = (box[2:] - box[:2]) * margin
    x1, y1 = max(0, int(np.floor(box[0] - dx))), max(0, int(np.floor(box[1] - dy)))
    x2, y2 = min(width, int(np.ceil(box[2] + dx))), min(height, int(np.ceil(box[3] + dy)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def rows():
    with base.MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        all_rows = [r for r in csv.DictReader(f) if r["seed0_split"] in ("train", "validation")]
    assert sum(r["seed0_split"] == "train" for r in all_rows) == 1005
    assert sum(r["seed0_split"] == "validation" for r in all_rows) == 253
    return sorted(all_rows, key=lambda r: (r["seed0_split"] != "validation", r["id"]))


def infer(samples, device, batch):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "poses"
    cache.mkdir(exist_ok=True)
    meta = {
        "script_sha256": base.sha(Path(__file__)),
        "manifest_sha256": base.sha(base.MANIFEST),
        "model_sha256": base.sha(base.MODEL),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "device": device,
        "batch": batch,
        "margin_each_side": MARGIN,
        "confidence": 0.25,
        "imgsz": 640,
        "select": "max IoU against GT bbox >= .3 within GT crop; missing otherwise",
        "input": "RGB frame crop using GT bbox, no GT keypoints to YOLO",
        "missing_bbox": "mark frame missing; never substitute GT pose",
    }
    meta_file = OUT / "provenance.json"
    if meta_file.exists():
        if json.loads(meta_file.read_text(encoding="utf-8")) != meta:
            raise ValueError("Provenance mismatch")
    else:
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    model = YOLO(str(base.MODEL))
    start = time.time()
    for k, row in enumerate(samples, 1):
        sid = row["id"]
        dest = cache / f"{sid}.npz"
        if dest.exists():
            continue
        label = base.LABELS / f"{sid}.mat"
        if base.sha(label) != row["label_sha256"]:
            raise ValueError("label hash mismatch " + sid)
        mat = sio.loadmat(label)
        T = int(row["nframes"])
        if int(mat["train"][0, 0]) != 1 or str(mat["action"][0]) != row["action"]:
            raise ValueError("split mismatch")
        bbox = np.asarray(mat["bbox"], dtype=float)
        xy = np.full((T, 12, 2), np.nan, np.float32)
        conf = np.zeros((T, 12), np.float32)
        status = np.zeros(
            T, np.uint8
        )  # 0 missing/invalid bbox; 1 no detection; 2 no match; 3 matched
        frame_hash = hashlib.sha256()
        for start_idx in range(0, T, batch):
            images = []
            data = []
            for t in range(start_idx, min(T, start_idx + batch)):
                raw = (base.FRAMES / sid / f"{t + 1:06d}.jpg").read_bytes()
                frame_hash.update(len(raw).to_bytes(8, "big"))
                frame_hash.update(raw)
                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("decode " + sid)
                rect = (
                    crop_rectangle(bbox[t], image.shape[1], image.shape[0])
                    if t < len(bbox)
                    else None
                )
                if rect is None:
                    continue
                x1, y1, x2, y2 = rect
                images.append(image[y1:y2, x1:x2])
                data.append((t, x1, y1))
            if not images:
                continue
            results = model.predict(
                source=images, imgsz=640, conf=0.25, device=device, verbose=False
            )
            if len(results) != len(images):
                raise ValueError("inference count mismatch")
            for result, (t, x1, y1) in zip(results, data):
                status[t] = 1
                if not len(result.boxes):
                    continue
                boxes = result.boxes.xyxy.cpu().numpy().copy()
                boxes += np.array([x1, y1, x1, y1])
                scores = base.ious(boxes, bbox[t])
                status[t] = 2
                if not np.isfinite(scores).all() or scores.max() < 0.3:
                    continue
                idx = int(np.argmax(scores))
                p = result.keypoints.xy[idx, 5:17].cpu().numpy().copy()
                c = result.keypoints.conf[idx, 5:17].cpu().numpy().copy()
                # Preserve absent keypoints as missing rather than translating (0,0) to crop origin.
                valid = np.isfinite(p).all(axis=1) & (p > 0).all(axis=1)
                p[valid] += np.array([x1, y1])
                p[~valid] = np.nan
                xy[t] = p
                conf[t] = c
                status[t] = 3
        tmp = cache / f"{sid}.tmp.npz"
        np.savez_compressed(
            tmp,
            xy=xy,
            conf=conf,
            status=status,
            annotation_sha256=row["label_sha256"],
            image_bytes_sha256=frame_hash.hexdigest(),
        )
        os.replace(tmp, dest)
        print(
            f"{k}/{len(samples)} {row['seed0_split']} {sid} frames={T} matched={(status == 3).sum()} elapsed={time.time() - start:.0f}s",
            flush=True,
        )


def evaluate(samples):

    torch.set_num_threads(4)
    data = {"train": [], "validation": []}
    data_identity = []
    counts = {s: np.zeros(4, int) for s in data}
    for row in samples:
        d = np.load(OUT / "poses" / f"{row['id']}.npz")
        T = int(row["nframes"])
        if len(d["xy"]) != T or str(d["annotation_sha256"]) != row["label_sha256"]:
            raise ValueError("cache mismatch")
        counts[row["seed0_split"]] += np.bincount(d["status"], minlength=4)
        x = d["xy"]
        features = base.resample(x[:, SWAP])
        y = base.ACTION_CLASSES.index(row["action"])
        data[row["seed0_split"]].append((torch.from_numpy(features), y, row["id"]))
        if row["seed0_split"] == "validation":
            data_identity.append((torch.from_numpy(base.resample(x)), y, row["id"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = []
    for variant, dataset in [("identity", data_identity), ("fixed_swap", data["validation"])]:
        x = torch.stack([v[0] for v in dataset]).to(device)
        target = [v[1] for v in dataset]
        for seed in (0, 1, 2):
            net = trainer.model_for("bilstm_attention").to(device)
            path = base.CKPTS / f"seed{seed}" / "best.pt"
            net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            net.eval()
            with torch.inference_mode():
                pred = torch.cat([net(a).argmax(-1) for a in x.split(32)]).cpu().tolist()
            outputs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint_sha256": base.sha(path),
                    **base.metric(target, pred),
                    "predictions": [
                        {"id": v[2], "target": v[1], "predicted": p} for v, p in zip(dataset, pred)
                    ],
                }
            )
            print("GT trained", variant, seed, outputs[-1]["accuracy_pct"], flush=True)
    (OUT / "gt_trained_validation.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    (OUT / "coverage.json").write_text(
        json.dumps(
            {
                s: dict(zip(["invalid_bbox", "no_detection", "no_match", "matched"], map(int, c)))
                for s, c in counts.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(batch=32, epochs=50)
    for seed in (0, 1, 2):
        trainer.run("bilstm_attention", seed, data, args, OUT / "yolo_trained_fixed_swap")
    summary = []
    for seed in (0, 1, 2):
        path = (
            OUT
            / "yolo_trained_fixed_swap"
            / "bilstm_attention"
            / f"seed{seed}"
            / "best_validation.json"
        )
        r = json.loads(path.read_text(encoding="utf-8"))
        summary.append(
            {
                "seed": seed,
                "epoch": r["epoch"],
                "accuracy_pct": r["accuracy"],
                "macro_f1_pct": r["macro_f1"],
            }
        )
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "gt_trained": [{k: v for k, v in r.items() if k != "predictions"} for r in outputs],
                "yolo_trained": summary,
                "training": "fixed swap, best validation epoch",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("infer", "evaluate"), required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    samples = rows()
    samples = samples[: args.limit] if args.limit else samples
    if args.stage == "infer":
        infer(samples, args.device, args.batch)
    else:
        if args.limit:
            raise ValueError("Evaluation requires full dataset")
        evaluate(samples)
