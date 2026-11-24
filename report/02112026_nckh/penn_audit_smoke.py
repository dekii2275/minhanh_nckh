"""Small inference and untrained classifier plumbing check on official-train frames."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "penn_audit_artifacts"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from data.action_dataset import normalize_skeleton_sequence  # noqa: E402
from models.action_bilstm import ActionBiLSTMAttentionClassifier  # noqa: E402

model = YOLO(str(ROOT / "models/yolo11n-pose.pt"))
samples = [("2062", 1), ("0864", 41), ("0137", 1), ("0757", 176)]
records = []
start = time.perf_counter()
for sid, frame in samples:
    p = ROOT / f"data/raw/Penn_Action/frames/{sid}/{frame:06}.jpg"
    image = cv2.imread(str(p))
    mat = sio.loadmat(ROOT / f"data/raw/Penn_Action/labels/{sid}.mat")
    res = model.predict(source=image, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
    count = len(res.boxes)
    kpts = None
    confidence = None
    if count:
        best = int(res.boxes.conf.cpu().numpy().argmax())
        kpts = res.keypoints.xy[best].cpu().numpy()
        confidence = res.keypoints.conf[best].cpu().numpy()
        coords = kpts[5:17].reshape(1, 12, 2).copy()
    else:
        coords = np.full((1, 12, 2), np.nan, dtype=np.float32)
    feats = normalize_skeleton_sequence(coords)
    torch.manual_seed(0)
    clf = ActionBiLSTMAttentionClassifier().eval()
    with torch.no_grad():
        logits = clf(torch.tensor(feats[None, :, :]), torch.tensor([1]))
    panels = []
    for mode in ("GT", "YOLO identity", "YOLO swapped"):
        pane = image.copy()
        for j in range(12):
            color = (255, 180, 0) if j % 2 == 0 else (255, 0, 255)
            if mode == "GT":
                if not mat["visibility"][frame - 1, j + 1]:
                    continue
                x, y = mat["x"][frame - 1, j + 1], mat["y"][frame - 1, j + 1]
            elif kpts is not None:
                idx = 5 + (j if mode == "YOLO identity" else j ^ 1)
                if confidence[idx] < 0.5:
                    continue
                x, y = kpts[idx]
            else:
                continue
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(pane, (int(round(x)), int(round(y))), 5, color, 2)
                cv2.putText(
                    pane,
                    str(j // 2),
                    (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                )
        cv2.putText(
            pane,
            f"{sid} frame {frame} {mode}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            pane,
            f"{sid} frame {frame} {mode}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
        panels.append(pane)
    cv2.imwrite(str(OUT / f"visual_{sid}_{frame:06}.jpg"), np.concatenate(panels, axis=1))
    records.append(
        {
            "id": sid,
            "frame_one_based": frame,
            "detections": count,
            "chosen_box_conf": float(res.boxes.conf[best]) if count else None,
            "gt_shape": list(mat["x"].shape),
            "image_hw": list(image.shape[:2]),
            "yolo_xy_shape": list(kpts.shape) if count else None,
            "normalized_shape": list(feats.shape),
            "untrained_classifier_logits_shape": list(logits.shape),
            "finite_logits": bool(torch.isfinite(logits).all()),
        }
    )
(OUT / "smoke.json").write_text(
    json.dumps(
        {
            "samples": records,
            "elapsed_seconds_including_model_load_and_render": time.perf_counter() - start,
            "device": "cpu",
            "imgsz": 640,
            "box_conf": 0.25,
            "classifier": "random weights, plumbing check only",
        },
        indent=2,
    ),
    encoding="utf8",
)
print(json.dumps(records, indent=2))
