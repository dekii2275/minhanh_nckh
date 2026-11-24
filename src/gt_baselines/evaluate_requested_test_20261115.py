"""Re-evaluate existing selected checkpoints on Penn official test using CUDA."""

import csv
import json
from pathlib import Path

import compare_person_selection as base
import confidence_torso_features as feat
import gt_crop_confidence_experiment as confexp
import gt_crop_pose_experiment as pose
import numpy as np
import scipy.io as sio
import targeted_pose_pilot_20261113 as targeted
import torch
import train_gt_baselines as trainer
from train_gt_cascade_scratch import CascadeScratch

ROOT = pose.ROOT
OUT = ROOT / "outputs/requested_test_gpu_20261115"
OLD = ROOT / "outputs/official_test_abrupt_interpolation_20261113"
CONF = ROOT / "outputs/confidence_confirmation_penn_20261105"
GT = ROOT / "outputs/gt_baselines_20261102_v1"
COMB = ROOT / "outputs/joint_interpolation_combined_20261113"
FIXED = ROOT / "outputs/gt_crop_corrections_20261103/retrained/fixed_swap/bilstm_attention"


def gt_feature(row):
    path = base.LABELS / (row["id"] + ".mat")
    assert trainer.sha(path) == row["label_sha256"]
    m = sio.loadmat(path)
    assert int(m["train"][0, 0]) == -1 and str(m["action"][0]) == row["action"]
    x, y, v = (np.asarray(m[k])[:, 1:13] for k in ("x", "y", "visibility"))
    xy = np.stack((x, y), axis=-1).astype(np.float32)
    xy[(v == 0) | (x <= 0)] = np.nan
    f = trainer.normalize_skeleton_sequence(xy)
    t = np.linspace(0, len(f) - 1, 64)
    f = np.stack([np.interp(t, np.arange(len(f)), f[:, j]) for j in range(24)], axis=-1).astype(
        np.float32
    )
    assert np.isfinite(f).all()
    return f, m


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    assert not OUT.exists(), f"Choose a new output: {OUT}"
    torch.set_num_threads(4)
    OUT.mkdir(parents=True)
    rows = sorted(
        (
            r
            for r in csv.DictReader(base.MANIFEST.open(encoding="utf-8-sig", newline=""))
            if r["official_split"] == "test"
        ),
        key=lambda r: r["id"],
    )
    assert len(rows) == 1068 and all(r["seed0_split"] == "test" for r in rows)
    fs = {
        k: [] for k in ("raw", "combined", "torso_capacity_control", "torso_confidence_mask", "gt")
    }
    pck = np.zeros(3, dtype=np.int64)
    changes = []
    for i, r in enumerate(rows):
        g, mat = gt_feature(r)
        path = CONF / "test_pose/poses" / (r["id"] + ".npz")
        with np.load(path) as d:
            assert str(d["annotation_sha256"]) == r["label_sha256"] and len(d["xy"]) == int(
                r["nframes"]
            )
            xy, conf, status = d["xy"][:, pose.SWAP], d["conf"][:, pose.SWAP], d["status"]
        corrected, k = targeted.correct(xy, conf)
        raw = base.resample(xy)
        variants, _ = feat.make_variants(raw, xy, conf)
        fs["raw"].append(raw)
        fs["combined"].append(base.resample(corrected))
        fs["gt"].append(g)
        for name in ("torso_capacity_control", "torso_confidence_mask"):
            fs[name].append(variants[name])
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], -1).astype(float)
        vis = mat["visibility"][:, 1:13] > 0
        bb = np.full((len(xy), 4), np.nan)
        n = min(len(xy), len(mat["bbox"]))
        bb[:n] = mat["bbox"][:n]
        wh = bb[:, 2:] - bb[:, :2]
        bok = np.isfinite(bb).all(1) & (wh > 0).all(1)
        diag = np.linalg.norm(wh, axis=1)
        eligible = vis & targeted.valid(gt) & bok[:, None]
        obs0 = targeted.valid(xy) & (status[:, None] == 3)
        obs1 = targeted.valid(corrected) & ((status[:, None] == 3) | (k > 0))
        d0 = np.linalg.norm(xy - gt, axis=2) / diag[:, None]
        d1 = np.linalg.norm(corrected - gt, axis=2) / diag[:, None]
        pck += [
            eligible.sum(),
            (eligible & obs0 & (d0 <= 0.2)).sum(),
            (eligible & obs1 & (d1 <= 0.2)).sum(),
        ]
        changes.append(
            {
                "id": r["id"],
                "pose_sha256": trainer.sha(path),
                "annotation_sha256": r["label_sha256"],
                "modified_joints": int((k > 0).sum()),
            }
        )
        if (i + 1) % 250 == 0:
            print("features", i + 1, flush=True)
    fs = {k: np.stack(v) for k, v in fs.items()}
    with np.load(OLD / "test_raw.npz") as z:
        assert np.array_equal(z["ids"], [r["id"] for r in rows]) and np.array_equal(
            z["x"], fs["raw"]
        )
        fs["abrupt"] = np.load(OLD / "test_interpolation.npz")["x"].copy()
    ids = np.array([r["id"] for r in rows])
    targets = np.array([base.ACTION_CLASSES.index(r["action"]) for r in rows])
    np.savez_compressed(OUT / "features.npz", ids=ids, y=targets, **fs)
    paths = {
        "raw_fulltrain": (OLD / "trained/raw", "epoch50.pt", "raw", "bilstm"),
        "abrupt_fulltrain": (OLD / "trained/interpolation", "epoch50.pt", "abrupt", "bilstm"),
        "raw_validation": (FIXED, "best.pt", "raw", "bilstm"),
        "combined_validation": (COMB / "trained/bilstm_attention", "best.pt", "combined", "bilstm"),
        "confidence_baseline": (CONF / "trained/baseline", "best.pt", "raw", "bilstm"),
        "torso_control": (
            CONF / "trained/torso_capacity_control",
            "best.pt",
            "torso_capacity_control",
            "control",
        ),
        "torso_confidence": (
            CONF / "trained/torso_confidence_mask",
            "best.pt",
            "torso_confidence_mask",
            "confidence",
        ),
        "gt_bilstm": (GT / "bilstm_attention", "best.pt", "gt", "bilstm"),
        "gt_cascade": (GT / "cascade_scratch", "best.pt", "gt", "cascade"),
    }
    records = []
    predictions = {}
    for name, (directory, filename, feature, kind) in paths.items():
        for seed in range(3):
            ck = directory / f"seed{seed}" / filename
            meta = json.loads((ck.parent / "provenance.json").read_text(encoding="utf8"))
            assert trainer.sha(ck) == meta["checkpoint_sha256"]
            if kind == "cascade":
                model = CascadeScratch()
            elif kind == "control":
                model = confexp.model_for("torso_capacity_control")
            elif kind == "confidence":
                model = confexp.model_for("torso_confidence_mask")
            else:
                model = trainer.model_for("bilstm_attention")
            model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
            model = model.cuda().eval()
            x = torch.from_numpy(fs[feature])
            with torch.inference_mode():
                pred = torch.cat([model(b.cuda()).argmax(-1).cpu() for b in x.split(32)]).numpy()
            predictions[(name, seed)] = pred
            result = {
                "method": name,
                "seed": seed,
                **base.metric(targets, pred),
                "checkpoint": str(ck),
                "checkpoint_sha256": trainer.sha(ck),
            }
            records.append(result)
            print(name, seed, result["accuracy_pct"], result["macro_f1_pct"], flush=True)
            del model
    summary = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "n": len(rows),
        "pck": {
            "eligible": int(pck[0]),
            "raw_pct": float(100 * pck[1] / pck[0]),
            "combined_pct": float(100 * pck[2] / pck[0]),
        },
        "methods": {},
        "comparisons": {},
    }
    for name in paths:
        rs = [x for x in records if x["method"] == name]
        summary["methods"][name] = {
            "accuracy_mean_pct": float(np.mean([x["accuracy_pct"] for x in rs])),
            "accuracy_sd_pct": float(np.std([x["accuracy_pct"] for x in rs], ddof=1)),
            "macro_f1_mean_pct": float(np.mean([x["macro_f1_pct"] for x in rs])),
            "seeds": rs,
        }
    rng = np.random.default_rng(20261115)
    boot = rng.integers(0, len(rows), (10000, len(rows)))
    for a, b in [
        ("abrupt_fulltrain", "raw_fulltrain"),
        ("combined_validation", "raw_validation"),
        ("torso_confidence", "confidence_baseline"),
        ("torso_confidence", "torso_control"),
    ]:
        d = np.mean(
            [
                (predictions[a, s] == targets).astype(int)
                - (predictions[b, s] == targets).astype(int)
                for s in range(3)
            ],
            axis=0,
        )
        summary["comparisons"][a + "_vs_" + b] = {
            "delta_pp": float(d.mean() * 100),
            "bootstrap95_pp": np.quantile(d[boot].mean(1) * 100, [0.025, 0.975]).tolist(),
        }
    with (OUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "id", "target", "prediction"])
        for (name, seed), pred in predictions.items():
            w.writerows(zip([name] * len(ids), [seed] * len(ids), ids, targets, pred))
    (OUT / "input_hashes.json").write_text(json.dumps(changes, indent=2), encoding="utf8")
    summary["manifest_sha256"] = trainer.sha(base.MANIFEST)
    summary["script_sha256"] = trainer.sha(Path(__file__))
    summary["features_sha256"] = trainer.sha(OUT / "features.npz")
    summary["notes"] = (
        "Historical test, exploratory. All GPU inference; existing checkpoints. Full-train final epoch versus 1005 train / 253 validation-selected models must be reported separately. GT boxes assist YOLO. Offline."
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
