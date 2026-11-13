"""Illustrate cached original vs combined interpolation; flags are highlighted."""

import cv2
import numpy as np
from evaluate_abrupt_joint_interpolation_20261113 import correct
from PIL import Image
from previous_joint_visual_pilot_20261113 import render
from scipy.io import loadmat
from targeted_pose_pilot_20261113 import FRAMES, LABEL, ROOT, SRC, SWAP

OUT = ROOT / "outputs/abrupt_joint_interpolation_20261113"

for sid, start, end in [("1625", 89, 97), ("0425", 13, 21), ("1534", 31, 39)]:
    path = OUT / f"example_{sid}_{start}_{end}.gif"
    if path.exists():
        raise FileExistsError(path)
    with np.load(SRC / f"{sid}.npz") as d:
        x = d["xy"][:, SWAP]
        conf = d["conf"][:, SWAP]
    y, k = correct(x, conf)
    m = loadmat(LABEL / f"{sid}.mat")
    gt = np.stack([m["x"][:, 1:13], m["y"][:, 1:13]], -1)
    vis = m["visibility"][:, 1:13] > 0
    bb = np.full((len(x), 4), np.nan)
    n = min(len(x), len(m["bbox"]))
    bb[:n] = m["bbox"][:n]
    frames = []
    for f in range(start, end + 1):
        t = f - 1
        im = cv2.imread(str(FRAMES / sid / f"{f:06d}.jpg"))
        assert im is not None
        modified = k[t] > 0
        cvis = conf[t].copy()
        cvis[modified] = np.maximum(cvis[modified], 0.5)
        left = render(
            im, x[t], conf[t], gt[t], vis[t], bb[t], modified, f"{sid} frame {f} | BEFORE"
        )
        right = render(im, y[t], cvis, gt[t], vis[t], bb[t], modified, f"{sid} frame {f} | AFTER")
        canvas = np.hstack((left, right))
        h = canvas.shape[0]
        canvas = cv2.copyMakeBorder(canvas, 0, 26, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(
            canvas,
            "GT green | YOLO red | cyan = interpolated (shown even if confidence low)",
            (5, h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
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
        path, save_all=True, append_images=frames[1:], duration=300, loop=0, optimize=False
    )
print("Rendered two illustrative clips; confidence visualization only, no data changed.")
