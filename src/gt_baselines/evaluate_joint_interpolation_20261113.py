"""Combined short-gap and 3-frame interpolation; GPU train/validation pilot."""

import csv
import json
import sys
import time

import compare_person_selection as base
import evaluate_three_frame_spike_20261113 as three
import gt_crop_pose_experiment as source
import numpy as np
import torch
import train_drop_jump_frames_20261113 as experiment
from scipy.io import loadmat
from targeted_pose_pilot_20261113 import NAMES, correct, sha, valid

ROOT = source.ROOT
OUT = ROOT / "outputs/joint_interpolation_combined_20261113"
OLD = ROOT / "outputs/gt_crop_corrections_20261103"
experiment.OUT = OUT
three.OUT = OUT


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows = source.rows()
    old = np.load(OLD / "features_fixed_swap.npz")
    assert (
        sha(OLD / "features_fixed_swap.npz")
        == json.loads((OLD / "prepared.json").read_text())["feature_sha256"]["fixed_swap"]
    )
    assert np.array_equal(old["ids"], [r["id"] for r in rows])
    OUT.mkdir(parents=True)
    protocol = {
        "scope": "seed0 train 1005 validation 253, no official test; exploratory same validation previously used for method development",
        "source": "GT bbox supported crop and IoU detection selection; fixed left/right swap; 12 keypoints",
        "method": "simultaneously from original pose: (1) linearly interpolate missing or confidence <.3 runs <=3 frames with both anchors confidence >=.5 and endpoint distance <=2 predicted torso median scale; no extrapolation. (2) replace center of 3 reliable frames by midpoint if center deviation >.75 scale and endpoints <=2 scale. No multi-joint uncertainty gate. Pose only; confidence unchanged.",
        "caution": "same exact method explored for pose previously; GT keypoints not used in correction; clip-wide torso and future anchors make offline. Missing and low-confidence handled; 3-frame rule can alter correct points. Unresolved absent points use standard preprocessing. Not claiming identity repair.",
        "classifier": "coordinate-only BiLSTM+attention, 3 seeds 0/1/2, 50 epochs batch 32, Adam lr .001 wd .0001 cosine, best validation accuracy earliest tie; train all models GPU",
        "baseline": "existing fixed swap YOLO trained baseline same data and seeds; all baseline features reverified",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "manifest_sha256": sha(base.MANIFEST),
        "script_sha256": sha(__import__("pathlib").Path(__file__)),
        "correction_sha256": sha(ROOT / "src/gt_baselines/targeted_pose_pilot_20261113.py"),
        "training_sha256": sha(__import__("pathlib").Path(experiment.trainer.__file__)),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    xs = []
    video = []
    records = []
    gc0 = gc1 = gt_n = 0
    motion_n = 0
    mv0 = mv1 = 0.0
    start = time.time()
    for i, r in enumerate(rows):
        sid = r["id"]
        p = source.OUT / "poses" / f"{sid}.npz"
        assert sha(base.LABELS / f"{sid}.mat") == r["label_sha256"]
        with np.load(p) as d:
            assert str(d["annotation_sha256"]) == r["label_sha256"]
            x = d["xy"][:, source.SWAP]
            c = d["conf"][:, source.SWAP]
            status = d["status"]
        assert len(x) == int(r["nframes"])
        assert np.array_equal(base.resample(x), old["x"][i])
        y, k = correct(x, c)
        assert np.array_equal(y[k == 0], x[k == 0], equal_nan=True)
        xs.append(base.resample(y))
        video.append(
            {
                "id": sid,
                "action": r["action"],
                "split": r["seed0_split"],
                "frames": len(x),
                "frames_corrected": int((k > 0).any(1).sum()),
                "joints_gap": int((k == 1).sum()),
                "joints_spike": int((k == 2).sum()),
                "pose_sha256": sha(p),
            }
        )
        if r["seed0_split"] == "validation":
            m = loadmat(base.LABELS / f"{sid}.mat")
            gt = np.stack([m["x"][:, 1:13], m["y"][:, 1:13]], -1).astype(float)
            vis = m["visibility"][:, 1:13] > 0
            bb = np.full((len(x), 4), np.nan)
            n = min(len(x), len(m["bbox"]))
            bb[:n] = m["bbox"][:n]
            wh = bb[:, 2:] - bb[:, :2]
            bok = np.isfinite(bb).all(1) & (wh > 0).all(1)
            diag = np.linalg.norm(wh, axis=1)
            eligible = vis & valid(gt) & bok[:, None]
            obs0 = valid(x) & (status[:, None] == 3)
            obs1 = valid(y) & ((status[:, None] == 3) | (k > 0))
            d0 = np.linalg.norm(x - gt, axis=2) / diag[:, None]
            d1 = np.linalg.norm(y - gt, axis=2) / diag[:, None]
            g0 = eligible & obs0 & (d0 <= 0.2)
            g1 = eligible & obs1 & (d1 <= 0.2)
            gt_n += eligible.sum()
            gc0 += g0.sum()
            gc1 += g1.sum()
            common = eligible & obs0
            pair = common[1:] & common[:-1]
            gm = gt[1:] - gt[:-1]
            delt0 = np.linalg.norm((x[1:] - x[:-1]) - gm, axis=2) / diag[1:, None]
            delt1 = np.linalg.norm((y[1:] - y[:-1]) - gm, axis=2) / diag[1:, None]
            motion_n += pair.sum()
            mv0 += float(delt0[pair].sum())
            mv1 += float(delt1[pair].sum())
            for t, j in np.argwhere((k > 0) & eligible):
                records.append(
                    {
                        "id": sid,
                        "action": r["action"],
                        "frame": int(t + 1),
                        "joint": NAMES[j],
                        "kind": "gap" if k[t, j] == 1 else "spike",
                        "was_observed": int(obs0[t, j]),
                        "error_before": float(d0[t, j]) if obs0[t, j] else None,
                        "error_after": float(d1[t, j]),
                        "pck_delta": int(g1[t, j]) - int(g0[t, j]),
                    }
                )
        if (i + 1) % 250 == 0:
            print(f"prepared {i + 1}/{len(rows)} seconds {time.time() - start:.0f}", flush=True)
    np.savez_compressed(
        OUT / "features_dropped.npz", x=np.stack(xs), ids=old["ids"], y=old["y"], split=old["split"]
    )
    for name, data in [("per_video.csv", video), ("validation_changed_joints.csv", records)]:
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    splits = {
        s: {
            "videos": sum(v["split"] == s for v in video),
            "videos_changed": sum(v["split"] == s and v["frames_corrected"] > 0 for v in video),
            "frames_changed": sum(v["frames_corrected"] for v in video if v["split"] == s),
            "joints_gap": sum(v["joints_gap"] for v in video if v["split"] == s),
            "joints_spike": sum(v["joints_spike"] for v in video if v["split"] == s),
        }
        for s in ("train", "validation")
    }
    observed = [v for v in records if v["was_observed"]]
    pose = {
        "gt_eligible": int(gt_n),
        "pck_before": float(gc0 / gt_n * 100),
        "pck_after": float(gc1 / gt_n * 100),
        "changed_gt_eligible": len(records),
        "changed_observed": len(observed),
        "changed_missing": len(records) - len(observed),
        "pck_improved": sum(v["pck_delta"] == 1 for v in records),
        "pck_worsened": sum(v["pck_delta"] == -1 for v in records),
        "observed_closer": sum(v["error_after"] < v["error_before"] for v in observed),
        "observed_farther": sum(v["error_after"] > v["error_before"] for v in observed),
        "motion_pairs_common": int(motion_n),
        "motion_error_before": float(mv0 / motion_n),
        "motion_error_after": float(mv1 / motion_n),
        "motion_caveat": "only pairs observed both before correction; filling missing excluded; normalized 2D displacement error not physical velocity",
    }
    prior = json.loads((ROOT / "outputs/targeted_pose_pilot_20261113/summary.json").read_text())
    assert (
        prior["changed_joint_frames_all"]
        == splits["validation"]["joints_gap"] + splits["validation"]["joints_spike"]
    )
    assert abs(prior["corrected_pck20_missing_wrong_pct"] - pose["pck_after"]) < 1e-8
    (OUT / "pose_summary.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")
    (OUT / "prepared.json").write_text(
        json.dumps(
            {
                "features_sha256": sha(OUT / "features_dropped.npz"),
                "baseline_all_features_identical": True,
                "prior_validation_pilot_reproduced": True,
                "splits": splits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"splits": splits, "pose": pose}, indent=2), flush=True)


def compare():
    vals = {}
    paths = {
        "baseline": ROOT / "outputs/gt_crop_corrections_20261103/retrained/fixed_swap",
        "spike_only": ROOT / "outputs/three_frame_spike_isolated_20261113/trained",
        "guarded_spike": ROOT / "outputs/uncertain_skeleton_gating_20261113/trained",
        "combined_interpolation": OUT / "trained",
    }
    for name, path in paths.items():
        vals[name] = [
            json.loads(
                (path / "bilstm_attention" / f"seed{s}" / "best_validation.json").read_text()
            )
            for s in (0, 1, 2)
        ]
    rng = np.random.default_rng(20261113)
    boot = rng.integers(0, 253, (10000, 253))
    out = {}
    for name in ("baseline", "spike_only", "guarded_spike"):
        ds = []
        seeds = []
        for a, b in zip(vals["combined_interpolation"], vals[name]):
            assert [(z["id"], z["target"]) for z in a["predictions"]] == [
                (z["id"], z["target"]) for z in b["predictions"]
            ]
            d = np.array(
                [
                    int(x["predicted"] == x["target"]) - int(y["predicted"] == y["target"])
                    for x, y in zip(a["predictions"], b["predictions"])
                ]
            )
            ds.append(d)
            seeds.append(
                {
                    "seed": a["seed"],
                    "delta_pp": float(d.mean() * 100),
                    "improved": int((d > 0).sum()),
                    "worsened": int((d < 0).sum()),
                }
            )
        d = np.mean(ds, axis=0)
        out["combined_vs_" + name] = {
            "delta_pp": float(d.mean() * 100),
            "paired_video_bootstrap95_pp": np.quantile(
                d[boot].mean(1) * 100, [0.025, 0.975]
            ).tolist(),
            "seeds": seeds,
        }
    (OUT / "pairwise_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    {
        "prepare": prepare,
        "train": experiment.train,
        "summary": experiment.summarize,
        "frozen": three.frozen,
        "compare": compare,
    }[sys.argv[1]]()
