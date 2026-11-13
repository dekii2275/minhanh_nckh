"""Gate 3-frame repairs near multi-joint jumps; experimental, not identity tracking."""

import json
import sys
from pathlib import Path

import cv2
import evaluate_three_frame_spike_20261113 as core
import numpy as np
import train_drop_jump_frames_20261113 as experiment
from PIL import Image
from previous_joint_visual_pilot_20261113 import render
from scipy.io import loadmat
from targeted_pose_pilot_20261113 import (
    FRAMES,
    LABEL,
    NAMES,
    SRC,
    SWAP,
    sha,
    valid,
)
from targeted_pose_pilot_20261113 import (
    correct as old_correct,
)

ROOT = core.ROOT
OUT = ROOT / "outputs/uncertain_skeleton_gating_20261113"
core.OUT = OUT
experiment.OUT = OUT
audit = []


def guarded_correct(x, c):
    y, kind = old_correct(x, c)
    raw = kind == 2
    lengths = []
    for a, b in ((0, 6), (1, 7)):
        m = valid(x[:, a]) & valid(x[:, b])
        lengths.extend(np.linalg.norm(x[m, a] - x[m, b], axis=1))
    lengths = np.asarray(lengths)
    lengths = lengths[np.isfinite(lengths) & (lengths > 1)]
    trigger = np.zeros(len(x), bool)
    if len(lengths) >= 5:
        scale = np.median(lengths)
        good = valid(x) & np.isfinite(c) & (c >= 0.5)
        jump = good[1:] & good[:-1] & (np.linalg.norm(x[1:] - x[:-1], axis=2) > 1.5 * scale)
        trigger[1:] = jump.sum(axis=1) >= 3
    uncertain = trigger.copy()
    uncertain[1:] |= trigger[:-1]
    uncertain[:-1] |= trigger[1:]
    blocked = raw & uncertain[:, None]
    outkind = kind.copy()
    outkind[blocked] = 0
    audit.append(
        {
            "trigger_frames": (np.flatnonzero(trigger) + 1).tolist(),
            "uncertain_frames": (np.flatnonzero(uncertain) + 1).tolist(),
            "blocked_joint_frames": [(int(t + 1), NAMES[j]) for t, j in np.argwhere(blocked)],
            "raw_spike_joint_frames": int(raw.sum()),
        }
    )
    return y, outkind


def prepare():
    core.correct = guarded_correct
    core.prepare()
    rows = core.source.rows()
    assert len(rows) == len(audit)
    per = []
    for row, entry in zip(rows, audit):
        per.append({"id": row["id"], "split": row["seed0_split"], **entry})
    (OUT / "uncertainty_by_video.json").write_text(json.dumps(per, indent=2), encoding="utf-8")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    protocol.update(
        {
            "definition": "3+ YOLO joints with confidence >=0.5 at both frames and displacement >1.5 * median clip shoulder-hip scale: later frame is multi-joint jump trigger. Mark trigger +/-1 frames uncertain. Suppress ALL 3-frame spike corrections on uncertain frames; retain original coordinates. Other frames: same 3-frame rule as prior experiment.",
            "interpretation": "heuristic measures simultaneous joint jumps; no explicit anatomy or person ID model, flag is not confirmed identity switch; label does not enter classifier",
            "warning": "thresholds heuristic; validation reused and motivated by inspecting validation video 1625; uses future frame and clip scale; offline; no test",
            "script_sha256": sha(Path(__file__)),
            "base_experiment_script_sha256": sha(Path(core.__file__)),
        }
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(
        "gating:",
        json.dumps(
            {
                s: {
                    "trigger_frames": sum(len(a["trigger_frames"]) for a in per if a["split"] == s),
                    "uncertain_frames": sum(
                        len(a["uncertain_frames"]) for a in per if a["split"] == s
                    ),
                    "videos_uncertain": sum(
                        bool(a["uncertain_frames"]) for a in per if a["split"] == s
                    ),
                    "blocked_joint_frames": sum(
                        len(a["blocked_joint_frames"]) for a in per if a["split"] == s
                    ),
                }
                for s in ("train", "validation")
            },
            indent=2,
        ),
        flush=True,
    )


def compare():
    json.loads((ROOT / "outputs/three_frame_spike_isolated_20261113/summary.json").read_text())
    json.loads((OUT / "summary.json").read_text())
    allsets = {}
    for label, folder in [
        ("baseline", ROOT / "outputs/gt_crop_corrections_20261103/retrained/fixed_swap"),
        ("ungated", ROOT / "outputs/three_frame_spike_isolated_20261113/trained"),
        ("guarded", OUT / "trained"),
    ]:
        allsets[label] = [
            json.loads(
                (folder / "bilstm_attention" / f"seed{s}" / "best_validation.json").read_text()
            )
            for s in (0, 1, 2)
        ]
    rng = np.random.default_rng(20261113)
    boot = rng.integers(0, 253, (10000, 253))
    out = {}
    for a, b in [("guarded", "baseline"), ("guarded", "ungated")]:
        ds = []
        seedrows = []
        for x, y in zip(allsets[a], allsets[b]):
            assert [(p["id"], p["target"]) for p in x["predictions"]] == [
                (p["id"], p["target"]) for p in y["predictions"]
            ]
            z = np.array(
                [
                    int(p["predicted"] == p["target"]) - int(q["predicted"] == q["target"])
                    for p, q in zip(x["predictions"], y["predictions"])
                ]
            )
            ds.append(z)
            seedrows.append(
                {
                    "seed": x["seed"],
                    "difference_pp": float(z.mean() * 100),
                    "improved_videos": int((z > 0).sum()),
                    "worsened_videos": int((z < 0).sum()),
                }
            )
        d = np.mean(ds, axis=0)
        out[f"{a}_vs_{b}"] = {
            "mean_delta_pp": float(d.mean() * 100),
            "paired_video_bootstrap95_pp": np.quantile(
                d[boot].mean(1) * 100, [0.025, 0.975]
            ).tolist(),
            "seeds": seedrows,
        }
    (OUT / "pairwise_classification_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2), flush=True)


def visuals():
    entries = {r["id"]: r for r in json.loads((OUT / "uncertainty_by_video.json").read_text())}
    for sid, start, end in [("1625", 89, 97), ("0425", 13, 21), ("1534", 31, 39)]:
        with np.load(SRC / f"{sid}.npz") as d:
            x = d["xy"][:, SWAP]
            c = d["conf"][:, SWAP]
        y, k = old_correct(x, c)
        oldflags = k == 2
        z = x.copy()
        z[oldflags] = y[oldflags]
        u = np.zeros(len(x), bool)
        u[np.array(entries[sid]["uncertain_frames"], dtype=int) - 1] = True
        newflags = oldflags & ~u[:, None]
        guard = x.copy()
        guard[newflags] = y[newflags]
        mat = loadmat(LABEL / f"{sid}.mat")
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], -1)
        vis = mat["visibility"][:, 1:13] > 0
        bb = np.full((len(x), 4), np.nan)
        n = min(len(x), len(mat["bbox"]))
        bb[:n] = mat["bbox"][:n]
        frames = []
        for f in range(start, end + 1):
            t = f - 1
            im = cv2.imread(str(FRAMES / sid / f"{f:06d}.jpg"))
            title = f"{sid} frame {f} | uncertain: {'YES' if u[t] else 'no'}"
            left = render(im, z[t], c[t], gt[t], vis[t], bb[t], oldflags[t], title + " | 3-frame")
            right = render(
                im, guard[t], c[t], gt[t], vis[t], bb[t], newflags[t], title + " | guarded"
            )
            image = np.hstack((left, right))
            h = image.shape[0]
            image = cv2.copyMakeBorder(image, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(
                image,
                "Green = GT | red = pose | cyan = corrected; uncertain = repair skipped",
                (5, h + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            frames.append(
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert(
                    "P", palette=Image.Palette.ADAPTIVE, colors=128
                )
            )
        frames[0].save(
            OUT / f"guarded_{sid}_{start}_{end}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
            optimize=False,
        )


if __name__ == "__main__":
    {
        "prepare": prepare,
        "train": experiment.train,
        "summary": experiment.summarize,
        "compare": compare,
        "frozen": core.frozen,
        "visuals": visuals,
    }[sys.argv[1]]()
