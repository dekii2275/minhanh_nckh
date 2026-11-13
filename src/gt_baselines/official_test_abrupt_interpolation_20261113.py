"""Fixed-epoch full official-train GPU experiment, then locked official test."""

import csv
import json
import random
import sys
import time
from pathlib import Path

import compare_person_selection as base
import gt_crop_pose_experiment as pose
import numpy as np
import torch
import train_gt_baselines as trainer
from evaluate_abrupt_joint_interpolation_20261113 import correct
from scipy.io import loadmat
from targeted_pose_pilot_20261113 import sha
from torch import nn
from torch.utils.data import DataLoader

ROOT = pose.ROOT
OUT = ROOT / "outputs/official_test_abrupt_interpolation_20261113"
TEST = ROOT / "outputs/confidence_confirmation_penn_20261105/test_pose"
METHODS = ("raw", "interpolation")
SEEDS = (0, 1, 2)


def dump(path, v):
    path.write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(split):
    r = list(csv.DictReader(base.MANIFEST.open(encoding="utf-8-sig", newline="")))
    return sorted((v for v in r if v["official_split"] == split), key=lambda v: v["id"])


def verify_code():
    p = read(OUT / "protocol.json")
    assert p["script_sha256"] == sha(Path(__file__)) and p["correction_sha256"] == sha(
        ROOT / "src/gt_baselines/evaluate_abrupt_joint_interpolation_20261113.py"
    )
    assert p["classifier_sha256"] == sha(ROOT / "src/models/action_bilstm.py") and p[
        "resampler_sha256"
    ] == sha(Path(base.__file__))


def prepare_train():
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    rr = rows("train")
    assert len(rr) == 1258 and all(v["seed0_split"] in ("train", "validation") for v in rr)
    trprov = read(pose.OUT / "provenance.json")
    testprov = read(TEST / "provenance.json")
    assert trprov == testprov, "train/test pose inference provenance differs"
    OUT.mkdir(parents=True)
    protocol = {
        "question": "does targeted abrupt-joint interpolation improve accuracy versus raw pose on Penn official test?",
        "split": "1258 official train (including previous validation) / 1068 official test; test previously influenced historical research, not fully independent",
        "train": "full official train from scratch; 50 fixed epochs, final epoch checkpoint, no validation, no test before checkpoint lock; batch32 Adam .001 wd .0001 cosine T=50 eta_min .00001 clip1; 3 seeds 0/1/2; same class order and training data for both methods",
        "methods": "raw versus per-joint abrupt-jump interpolation from prior validation experiment, no configuration changes and no uncertainty gate",
        "correction": "consecutive original joints confidence >=.5, jump >1.5 median clip shoulder-hip scale; interpolate contiguous flagged run <=3 using reliable unflagged anchors whose displacement <=2 scale; no extrapolation; confidence unchanged",
        "inference": "YOLO11n-Pose GT bbox crop 25% and GT IoU matching >=0.3; fixed left-right swap, clip-wide scale and future frame; offline",
        "metrics": "accuracy, macro-F1, paired accuracy differences; 10000 bootstrap video resamples of seed-averaged paired correctness, no seed/selection/ historical-test uncertainty",
        "caveats": "method and epoch budget based on historical validation; test exposed in research history; not fully independent. No GT keypoints used by correction, bbox GT required. No claim online, automatic person selection, or novel contribution.",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "manifest_sha256": sha(base.MANIFEST),
        "train_pose_provenance_sha256": sha(pose.OUT / "provenance.json"),
        "test_pose_provenance_sha256": sha(TEST / "provenance.json"),
        "script_sha256": sha(Path(__file__)),
        "correction_sha256": sha(
            ROOT / "src/gt_baselines/evaluate_abrupt_joint_interpolation_20261113.py"
        ),
        "classifier_sha256": sha(ROOT / "src/models/action_bilstm.py"),
        "normalizer_sha256": sha(ROOT / "src/data/action_dataset.py"),
        "resampler_sha256": sha(Path(base.__file__)),
    }
    dump(OUT / "protocol.json", protocol)
    data = {m: [] for m in METHODS}
    changes = []
    start = time.time()
    for i, r in enumerate(rr):
        sid = r["id"]
        label = base.LABELS / f"{sid}.mat"
        p = pose.OUT / "poses" / f"{sid}.npz"
        assert sha(label) == r["label_sha256"]
        with np.load(p) as d:
            assert str(d["annotation_sha256"]) == r["label_sha256"]
            x = d["xy"][:, pose.SWAP]
            c = d["conf"][:, pose.SWAP]
            assert len(x) == int(r["nframes"])
        y, k = correct(x, c)
        assert np.array_equal(x[k == 0], y[k == 0], equal_nan=True)
        data["raw"].append(base.resample(x))
        data["interpolation"].append(base.resample(y))
        changes.append(
            {
                "id": sid,
                "frames": len(x),
                "modified_frames": int((k > 0).any(1).sum()),
                "modified_joint_frames": int((k > 0).sum()),
                "pose_sha256": sha(p),
                "label_sha256": r["label_sha256"],
            }
        )
        if (i + 1) % 250 == 0:
            print(f"train features {i + 1}/{len(rr)} {time.time() - start:.0f}s", flush=True)
    for m in METHODS:
        np.savez_compressed(
            OUT / f"train_{m}.npz",
            x=np.stack(data[m]),
            y=np.array([base.ACTION_CLASSES.index(r["action"]) for r in rr]),
            ids=np.array([r["id"] for r in rr]),
        )
    with (OUT / "train_changes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(changes[0]))
        w.writeheader()
        w.writerows(changes)
    dump(
        OUT / "train_prepared.json",
        {
            "count": len(rr),
            "features": {m: sha(OUT / f"train_{m}.npz") for m in METHODS},
            "modified_videos": sum(v["modified_frames"] > 0 for v in changes),
            "modified_frames": sum(v["modified_frames"] for v in changes),
            "modified_joints": sum(v["modified_joint_frames"] for v in changes),
        },
    )
    print("train prepared", json.dumps(read(OUT / "train_prepared.json")), flush=True)


def train():
    verify_code()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    prep = read(OUT / "train_prepared.json")
    assert all(sha(OUT / f"train_{m}.npz") == prep["features"][m] for m in METHODS)
    for m in METHODS:
        with np.load(OUT / f"train_{m}.npz") as a:
            xx = torch.from_numpy(a["x"].copy())
            yy = torch.from_numpy(a["y"].copy())
            dataset = torch.utils.data.TensorDataset(xx, yy)
        for seed in SEEDS:
            dest = OUT / "trained" / m / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=False)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            net = trainer.model_for("bilstm_attention").cuda()
            gen = torch.Generator().manual_seed(seed)
            loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=gen)
            opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0.00001)
            criterion = nn.CrossEntropyLoss()
            history = []
            start = time.time()
            for epoch in range(1, 51):
                net.train()
                loss_sum = 0.0
                for x, y in loader:
                    x = x.cuda()
                    y = y.cuda()
                    opt.zero_grad()
                    logits = net(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    loss_sum += loss.item() * len(y)
                sched.step()
                history.append({"epoch": epoch, "train_loss": loss_sum / len(dataset)})
                if epoch % 10 == 0:
                    print(
                        f"{m} seed={seed} epoch={epoch}/50 loss={history[-1]['train_loss']:.5f} seconds={time.time() - start:.1f}",
                        flush=True,
                    )
            torch.save(net.state_dict(), dest / "epoch50.pt")
            dump(dest / "history.json", history)
            dump(
                dest / "provenance.json",
                {
                    "method": m,
                    "seed": seed,
                    "epochs": 50,
                    "train_size": len(dataset),
                    "device": "cuda",
                    "seconds": time.time() - start,
                    "checkpoint_sha256": sha(dest / "epoch50.pt"),
                    "train_feature_sha256": prep["features"][m],
                },
            )


def lock():
    verify_code()
    items = {}
    for m in METHODS:
        items[m] = {}
        for seed in SEEDS:
            p = OUT / "trained" / m / f"seed{seed}"
            meta = read(p / "provenance.json")
            assert (
                meta["epochs"] == 50
                and meta["device"] == "cuda"
                and len(read(p / "history.json")) == 50
                and sha(p / "epoch50.pt") == meta["checkpoint_sha256"]
            )
            items[m][str(seed)] = meta["checkpoint_sha256"]
    if (OUT / "model_lock.json").exists():
        raise FileExistsError("Already locked")
    dump(
        OUT / "model_lock.json",
        {
            "checkpoint_sha256": items,
            "train_prepared_sha256": sha(OUT / "train_prepared.json"),
            "protocol_sha256": sha(OUT / "protocol.json"),
            "fixed_epoch": 50,
            "test_evaluated_before_lock": False,
        },
    )
    print("6 checkpoints locked before test", flush=True)


def test_prepare():
    verify_code()
    locked = read(OUT / "model_lock.json")
    assert locked["protocol_sha256"] == sha(OUT / "protocol.json")
    for m in METHODS:
        for seed in SEEDS:
            assert (
                sha(OUT / "trained" / m / f"seed{seed}" / "epoch50.pt")
                == locked["checkpoint_sha256"][m][str(seed)]
            )
    if (OUT / "test_prepared.json").exists():
        raise FileExistsError("Test features already prepared")
    rr = rows("test")
    assert len(rr) == 1068 and all(v["seed0_split"] == "test" for v in rr)
    data = {m: [] for m in METHODS}
    changes = []
    status = np.zeros(4, int)
    start = time.time()
    for i, r in enumerate(rr):
        sid = r["id"]
        p = TEST / "poses" / f"{sid}.npz"
        label = base.LABELS / f"{sid}.mat"
        assert sha(label) == r["label_sha256"]
        mat = loadmat(label)
        assert int(mat["train"][0, 0]) == -1 and str(mat["action"][0]) == r["action"]
        with np.load(p) as a:
            assert str(a["annotation_sha256"]) == r["label_sha256"]
            x = a["xy"][:, pose.SWAP]
            c = a["conf"][:, pose.SWAP]
            assert len(x) == int(r["nframes"])
            status += np.bincount(a["status"], minlength=4)
        y, k = correct(x, c)
        data["raw"].append(base.resample(x))
        data["interpolation"].append(base.resample(y))
        changes.append(
            {
                "id": sid,
                "frames": len(x),
                "modified_frames": int((k > 0).any(1).sum()),
                "modified_joint_frames": int((k > 0).sum()),
                "pose_sha256": sha(p),
                "label_sha256": r["label_sha256"],
            }
        )
        if (i + 1) % 250 == 0:
            print(f"test features {i + 1}/{len(rr)} {time.time() - start:.0f}s", flush=True)
    for m in METHODS:
        np.savez_compressed(
            OUT / f"test_{m}.npz",
            x=np.stack(data[m]),
            y=np.array([base.ACTION_CLASSES.index(r["action"]) for r in rr]),
            ids=np.array([r["id"] for r in rr]),
        )
    with (OUT / "test_changes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(changes[0]))
        w.writeheader()
        w.writerows(changes)
    dump(
        OUT / "test_prepared.json",
        {
            "count": len(rr),
            "features": {m: sha(OUT / f"test_{m}.npz") for m in METHODS},
            "modified_videos": sum(v["modified_frames"] > 0 for v in changes),
            "modified_frames": sum(v["modified_frames"] for v in changes),
            "modified_joints": sum(v["modified_joint_frames"] for v in changes),
            "status_counts": dict(
                zip(["invalid_bbox", "no_detection", "no_match", "matched"], map(int, status))
            ),
        },
    )
    print("test features prepared", json.dumps(read(OUT / "test_prepared.json")), flush=True)


def evaluate():
    verify_code()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    if (OUT / "summary.json").exists():
        raise FileExistsError("Test already evaluated")
    lock = read(OUT / "model_lock.json")
    prep = read(OUT / "test_prepared.json")
    preds = {}
    summ = []
    cpu_mismatch = 0
    for m in METHODS:
        assert sha(OUT / f"test_{m}.npz") == prep["features"][m]
        with np.load(OUT / f"test_{m}.npz") as a:
            x = torch.from_numpy(a["x"].copy())
            y = a["y"].copy()
            ids = a["ids"].copy()
        for seed in SEEDS:
            path = OUT / "trained" / m / f"seed{seed}" / "epoch50.pt"
            assert sha(path) == lock["checkpoint_sha256"][m][str(seed)]
            net = trainer.model_for("bilstm_attention")
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            net.eval()
            with torch.inference_mode():
                cpu = torch.cat([net(batch).argmax(-1) for batch in x.split(32)]).numpy()
            net = net.cuda()
            with torch.inference_mode():
                gpu = torch.cat(
                    [net(batch.cuda()).argmax(-1).cpu() for batch in x.split(32)]
                ).numpy()
            cpu_mismatch += int((cpu != gpu).sum())
            metrics = base.metric(y, gpu)
            preds[(m, seed)] = gpu
            summ.append(
                {
                    "method": m,
                    "seed": seed,
                    **metrics,
                    "cpu_gpu_mismatches": int((cpu != gpu).sum()),
                    "checkpoint_sha256": sha(path),
                }
            )
            print(m, seed, metrics, flush=True)
    with (OUT / "paired_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["id", "target", "seed", "raw", "interpolation", "delta_correct"]
        )
        w.writeheader()
        for seed in SEEDS:
            for i in range(len(y)):
                a = int(preds["raw", seed][i])
                b = int(preds["interpolation", seed][i])
                w.writerow(
                    {
                        "id": str(ids[i]),
                        "target": int(y[i]),
                        "seed": seed,
                        "raw": a,
                        "interpolation": b,
                        "delta_correct": int(b == y[i]) - int(a == y[i]),
                    }
                )
    diffs = np.stack(
        [
            (preds["interpolation", s] == y).astype(int) - (preds["raw", s] == y).astype(int)
            for s in SEEDS
        ]
    )
    avg = diffs.mean(0)
    rng = np.random.default_rng(20261113)
    idx = rng.integers(0, len(y), (10000, len(y)))
    result = {
        "seeds": summ,
        "n_videos": len(y),
        "accuracy_mean": {
            m: float(np.mean([v["accuracy_pct"] for v in summ if v["method"] == m]))
            for m in METHODS
        },
        "macro_f1_mean": {
            m: float(np.mean([v["macro_f1_pct"] for v in summ if v["method"] == m]))
            for m in METHODS
        },
        "delta_accuracy_pp": float(avg.mean() * 100),
        "delta_accuracy_per_seed_pp": [float(d.mean() * 100) for d in diffs],
        "bootstrap95_pp": np.quantile(avg[idx].mean(1) * 100, [0.025, 0.975]).tolist(),
        "improved_worsened_per_seed": [
            {"seed": s, "improved": int((d > 0).sum()), "worsened": int((d < 0).sum())}
            for s, d in zip(SEEDS, diffs)
        ],
        "cpu_gpu_prediction_mismatches": cpu_mismatch,
        "bootstrap_scope": "paired resampling of test videos; conditional on trained models; excludes seed and model/protocol selection uncertainty",
        "caution": "official test influenced historical research, not fully independent; GT bbox-assisted; offline; method chosen on validation",
    }
    dump(OUT / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    {
        "prepare_train": prepare_train,
        "train": train,
        "lock": lock,
        "prepare_test": test_prepare,
        "evaluate": evaluate,
    }[sys.argv[1]]()
