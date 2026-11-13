"""Locked Penn follow-up: train A/B/C on CUDA, evaluate historical official test."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import confidence_torso_features as feat
import gt_crop_confidence_experiment as exp
import gt_crop_pose_experiment as pose
import numpy as np
import torch

ROOT = pose.ROOT
OUT = ROOT / "outputs/confidence_confirmation_penn_20261105"
METHODS = ("baseline", "torso_capacity_control", "torso_confidence_mask")
SEEDS = (0, 1, 2)
SRC = ROOT / "outputs/gt_crop_confidence_torso_20261103"


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def initialize():
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "question": "C versus A and C versus B on held-out Penn official test",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "epochs": 50,
        "batch": 32,
        "training": "Adam lr .001 wd .0001, cosine T=50 eta_min=.00001, clip norm 1",
        "selection": "maximum validation accuracy, earliest tie; never use test for selection",
        "split": "seed0 manifest: 1005 train, 253 validation, 1068 official test",
        "history": "official test influenced prior research; not a fully independent test; validation used for development",
        "inference": "GT bbox crop 25% margin, max IoU >=.3, YOLO11n pose, fixed swap; offline",
        "primary": "C-A and C-B accuracy percentage points, paired video bootstrap 95%, seed-averaged per video; descriptive macro-F1 and per seed",
        "interpretation": "prespecified practical reference +1 percentage point C-A; C-B positive; exploratory given historical exposure; CIs are not multiplicity adjusted",
        "bootstrap": "10000 paired video resamples, seed 20261105; conditional on trained models, excludes seed and selection uncertainty",
        "feature_hashes": {m: pose.base.sha(SRC / f"features_{m}.npz") for m in METHODS},
        "script_sha256": pose.base.sha(Path(__file__)),
        "feature_script_sha256": pose.base.sha(Path(feat.__file__)),
        "training_script_sha256": pose.base.sha(Path(exp.__file__)),
        "pose_script_sha256": pose.base.sha(Path(pose.__file__)),
        "manifest_sha256": pose.base.sha(pose.base.MANIFEST),
        "yolo_sha256": pose.base.sha(pose.base.MODEL),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    write(OUT / "protocol.json", protocol)
    for method in METHODS:
        shutil.copyfile(SRC / f"features_{method}.npz", OUT / f"features_{method}.npz")


def train():
    assert torch.cuda.is_available(), "CUDA required"
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    for method in METHODS:
        path = SRC / f"features_{method}.npz"
        assert pose.base.sha(path) == protocol["feature_hashes"][method]
        with np.load(path) as a:
            data = {
                s: [
                    (torch.from_numpy(x.copy()), int(y), str(i))
                    for x, y, i, sp in zip(a["x"], a["y"], a["ids"], a["split"])
                    if sp == s
                ]
                for s in ("train", "validation")
            }
        assert (len(data["train"]), len(data["validation"])) == (1005, 253)
        for seed in SEEDS:
            directory = OUT / "trained" / method / f"seed{seed}"
            if directory.exists():
                assert (directory / "provenance.json").exists(), f"incomplete {directory}"
                continue
            print(f"START {method} {seed}", flush=True)
            exp.train_run(OUT, method, seed, data)


def test_rows():
    with pose.base.MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(
            (r for r in csv.DictReader(f) if r["seed0_split"] == "test"), key=lambda r: r["id"]
        )
    assert len(rows) == 1068
    return rows


def infer():
    # Use the exact historical implementation with only the expected MAT split assertion changed.
    import inspect

    code = inspect.getsource(pose.infer).replace(
        "int(mat['train'][0,0]) != 1", "int(mat['train'][0,0]) != -1"
    )
    assert code != inspect.getsource(pose.infer)
    pose.OUT = OUT / "test_pose"
    exec(code, pose.__dict__)
    pose.infer(test_rows(), "0", 16)


def evaluate():
    records = []
    xs = {m: [] for m in METHODS}
    ys = []
    ids = []
    counts = np.zeros(4, dtype=int)
    for row in test_rows():
        with np.load(OUT / "test_pose/poses" / f"{row['id']}.npz") as a:
            assert str(a["annotation_sha256"]) == row["label_sha256"] and len(a["xy"]) == int(
                row["nframes"]
            )
            xy, conf = a["xy"][:, pose.SWAP], a["conf"][:, pose.SWAP]
            variants, _ = feat.make_variants(pose.base.resample(xy), xy, conf)
            for m in METHODS:
                xs[m].append(variants[m])
            counts += np.bincount(a["status"], minlength=4)
        ys.append(pose.base.ACTION_CLASSES.index(row["action"]))
        ids.append(row["id"])
    y = np.array(ys)
    for method in METHODS:
        x = torch.from_numpy(np.stack(xs[method]))
        for seed in SEEDS:
            d = OUT / "trained" / method / f"seed{seed}"
            meta = json.loads((d / "provenance.json").read_text())
            assert pose.base.sha(d / "best.pt") == meta["checkpoint_sha256"]
            model = exp.model_for(method).cuda()
            model.load_state_dict(torch.load(d / "best.pt", map_location="cuda", weights_only=True))
            model.eval()
            with torch.inference_mode():
                p = torch.cat([model(b.cuda()).argmax(-1).cpu() for b in x.split(32)]).numpy()
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    **pose.base.metric(y, p),
                    "checkpoint_sha256": meta["checkpoint_sha256"],
                    "predictions": [
                        {"id": i, "target": int(t), "predicted": int(v)}
                        for i, t, v in zip(ids, y, p)
                    ],
                }
            )
    write(OUT / "test_predictions.json", records)
    rng = np.random.default_rng(20261105)
    indices = rng.integers(0, len(y), size=(10000, len(y)))
    summary = {
        "n": len(y),
        "coverage": dict(
            zip(("invalid_bbox", "no_detection", "no_match", "matched"), map(int, counts))
        ),
        "methods": {},
        "comparisons": {},
    }
    for m in METHODS:
        rs = [r for r in records if r["method"] == m]
        summary["methods"][m] = {
            "accuracy_mean_pct": float(np.mean([r["accuracy_pct"] for r in rs])),
            "accuracy_sd_pct": float(np.std([r["accuracy_pct"] for r in rs], ddof=1)),
            "macro_f1_mean_pct": float(np.mean([r["macro_f1_pct"] for r in rs])),
            "seeds": [{k: r[k] for k in ("seed", "accuracy_pct", "macro_f1_pct")} for r in rs],
        }
    for ref in METHODS[:2]:
        ds = []
        for seed in SEEDS:
            c = next(r for r in records if r["method"] == METHODS[2] and r["seed"] == seed)
            b = next(r for r in records if r["method"] == ref and r["seed"] == seed)
            ds.append(
                np.array(
                    [
                        int(a["predicted"] == a["target"]) - int(z["predicted"] == z["target"])
                        for a, z in zip(c["predictions"], b["predictions"])
                    ]
                )
            )
        d = np.mean(ds, axis=0)
        boot = d[indices].mean(axis=1) * 100
        summary["comparisons"]["C_vs_" + ref] = {
            "delta_accuracy_pp": float(d.mean() * 100),
            "paired_video_bootstrap_95_pp": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    write(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("init", "train", "infer", "evaluate"))
    args = p.parse_args()
    torch.set_num_threads(4)
    {"init": initialize, "train": train, "infer": infer, "evaluate": evaluate}[args.stage]()
