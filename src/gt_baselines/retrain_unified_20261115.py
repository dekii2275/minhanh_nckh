"""Train listed Penn methods from scratch on CUDA and evaluate shared splits."""

import csv
import json
import random
import sys
import time
from pathlib import Path

import compare_person_selection as base
import numpy as np
import torch
import train_gt_baselines as trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from train_gt_cascade_scratch import CascadeScratch

ROOT = trainer.ROOT
OUT = ROOT / "outputs/unified_retrain_20261115"
TEST = ROOT / "outputs/requested_test_gpu_20261115/features.npz"
METHODS = {
    "raw": ("outputs/gt_crop_corrections_20261103/features_fixed_swap.npz", "raw", 24),
    "abrupt": ("outputs/abrupt_joint_interpolation_20261113/features_dropped.npz", "abrupt", 24),
    "combined": (
        "outputs/joint_interpolation_combined_20261113/features_dropped.npz",
        "combined",
        24,
    ),
    "torso_zero": (
        "outputs/gt_crop_confidence_torso_20261103/features_torso_capacity_control.npz",
        "torso_capacity_control",
        48,
    ),
    "torso_confidence": (
        "outputs/gt_crop_confidence_torso_20261103/features_torso_confidence_mask.npz",
        "torso_confidence_mask",
        48,
    ),
    "gt_bilstm": (None, "gt", 24),
    "gt_cascade": (None, "gt", 24),
}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")


def model(name, dim):
    return (
        CascadeScratch()
        if name == "gt_cascade"
        else trainer.ActionBiLSTMAttentionClassifier(input_size=dim)
    )


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True)
    rows = list(csv.DictReader(trainer.MANIFEST.open(encoding="utf-8-sig", newline="")))
    assert len(rows) == 2326 and len(set(r["id"] for r in rows)) == 2326
    assert {
        s: sum(r["seed0_split"] == s for r in rows) for s in ("train", "validation", "test")
    } == {"train": 1005, "validation": 253, "test": 1068}
    byid = {r["id"]: r for r in rows}
    gt = trainer.load_data()
    with np.load(TEST) as z:
        tests = {k: z[k].copy() for k in z.files}
    assert len(tests["ids"]) == 1068
    assert all(
        byid[str(i)]["seed0_split"] == "test"
        and base.ACTION_CLASSES.index(byid[str(i)]["action"]) == int(y)
        for i, y in zip(tests["ids"], tests["y"])
    )
    hashes = {}
    for name, (source, testname, dim) in METHODS.items():
        if source:
            path = ROOT / source
            hashes[source] = trainer.sha(path)
            with np.load(path) as z:
                x, ids, y, split = (z[k].copy() for k in ("x", "ids", "y", "split"))
            assert len(ids) == 1258
            assert all(
                byid[str(i)]["seed0_split"] == str(s)
                and base.ACTION_CLASSES.index(byid[str(i)]["action"]) == int(v)
                for i, s, v in zip(ids, split, y)
            )
        else:
            vals = [
                (x.numpy(), y, sid, split)
                for split in ("train", "validation")
                for x, y, sid in gt[split]
            ]
            x = np.stack([v[0] for v in vals])
            y = np.array([v[1] for v in vals])
            ids = np.array([v[2] for v in vals])
            split = np.array([v[3] for v in vals])
        assert x.shape == (1258, 64, dim) and tests[testname].shape == (1068, 64, dim)
        assert np.isfinite(x).all() and np.isfinite(tests[testname]).all()
        np.savez_compressed(
            OUT / f"features_{name}.npz",
            x=x,
            y=y,
            ids=ids,
            split=split,
            test_x=tests[testname],
            test_y=tests["y"],
            test_ids=tests["ids"],
        )
        print("prepared", name, flush=True)
    write(
        OUT / "protocol.json",
        {
            "split": {"train": 1005, "validation": 253, "test": 1068},
            "selection": "highest validation accuracy, earliest tie; test never used for selection",
            "epochs": 50,
            "batch": 32,
            "seeds": [0, 1, 2],
            "optimizer": "Adam lr=.001 weight_decay=.0001 cosine T=50 eta_min=.00001 gradient_clip=1",
            "methods": list(METHODS),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "source_hashes": hashes,
            "test_features_sha256": trainer.sha(TEST),
            "manifest_sha256": trainer.sha(trainer.MANIFEST),
            "script_sha256": trainer.sha(Path(__file__)),
            "caveats": "historically exposed test; GT-assisted YOLO selection; offline; GT methods use annotation; from scratch; test features precomputed not inspected for model selection",
        },
    )


def fit():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    for name, (_, _, dim) in METHODS.items():
        with np.load(OUT / f"features_{name}.npz") as z:
            tr = z["split"] == "train"
            va = z["split"] == "validation"
            train_x = torch.from_numpy(z["x"][tr].copy())
            train_y = torch.from_numpy(z["y"][tr].copy())
            val_x = torch.from_numpy(z["x"][va].copy())
            val_y = z["y"][va].copy()
        assert len(train_y) == 1005 and len(val_y) == 253
        for seed in range(3):
            dest = OUT / "trained" / name / f"seed{seed}"
            if (dest / "provenance.json").exists():
                meta = json.loads((dest / "provenance.json").read_text())
                assert trainer.sha(dest / "best.pt") == meta["checkpoint_sha256"]
                continue
            dest.mkdir(parents=True, exist_ok=False)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            net = model(name, dim).cuda()
            loader = DataLoader(
                TensorDataset(train_x, train_y),
                batch_size=32,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0.00001)
            criterion = nn.CrossEntropyLoss()
            best = -1
            history = []
            start = time.time()
            for epoch in range(1, 51):
                net.train()
                total = 0.0
                for x, y in loader:
                    x = x.cuda()
                    y = y.cuda()
                    opt.zero_grad()
                    loss = criterion(net(x), y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    total += loss.item() * len(y)
                sched.step()
                net.eval()
                with torch.inference_mode():
                    p = torch.cat([net(b.cuda()).argmax(-1).cpu() for b in val_x.split(32)]).numpy()
                m = base.metric(val_y, p)
                history.append({"epoch": epoch, "loss": total / len(train_y), **m})
                if m["accuracy_pct"] > best:
                    best = m["accuracy_pct"]
                    torch.save(net.state_dict(), dest / "best.pt")
                    best_epoch = epoch
                if epoch % 10 == 0:
                    print(
                        name,
                        seed,
                        "epoch",
                        epoch,
                        "best",
                        round(best, 2),
                        "seconds",
                        round(time.time() - start),
                        flush=True,
                    )
            write(dest / "history.json", history)
            write(
                dest / "provenance.json",
                {
                    "seed": seed,
                    "method": name,
                    "best_epoch": best_epoch,
                    "checkpoint_sha256": trainer.sha(dest / "best.pt"),
                    "feature_sha256": trainer.sha(OUT / f"features_{name}.npz"),
                    "device": "cuda",
                    "seconds": time.time() - start,
                },
            )
            del net


def evaluate():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    assert not (OUT / "summary.json").exists()
    records = []
    preds = {}
    identities = {}
    for name, (_, _, dim) in METHODS.items():
        with np.load(OUT / f"features_{name}.npz") as z:
            datasets = {
                "validation": (
                    z["x"][z["split"] == "validation"].copy(),
                    z["y"][z["split"] == "validation"].copy(),
                    z["ids"][z["split"] == "validation"].copy(),
                ),
                "test": (z["test_x"].copy(), z["test_y"].copy(), z["test_ids"].copy()),
            }
        for scenario in (
            ["retrained", "frozen"] if name in ("abrupt", "combined") else ["retrained"]
        ):
            for seed in range(3):
                source = "raw" if scenario == "frozen" else name
                ck = OUT / "trained" / source / f"seed{seed}" / "best.pt"
                meta = json.loads((ck.parent / "provenance.json").read_text())
                assert trainer.sha(ck) == meta["checkpoint_sha256"]
                net = model(source, 24 if source == "raw" else dim).cuda()
                net.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
                net.eval()
                for split, (x, y, ids) in datasets.items():
                    with torch.inference_mode():
                        out = torch.cat(
                            [net(b.cuda()).argmax(-1).cpu() for b in torch.from_numpy(x).split(32)]
                        ).numpy()
                    key = (name, scenario, split, seed)
                    preds[key] = out
                    identities[(name, split)] = (ids, y)
                    records.append(
                        {
                            "method": name,
                            "scenario": scenario,
                            "split": split,
                            "seed": seed,
                            "epoch": meta["best_epoch"],
                            "checkpoint_sha256": meta["checkpoint_sha256"],
                            **base.metric(y, out),
                        }
                    )
                    print("evaluated", key, records[-1]["accuracy_pct"], flush=True)
                del net
    summary = {
        "n": {"validation": 253, "test": 1068},
        "device": torch.cuda.get_device_name(0),
        "methods": {},
        "comparisons": {},
    }
    with (OUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "scenario", "split", "seed", "id", "target", "predicted"])
        for (name, scenario, split, seed), p in preds.items():
            ids, y = identities[(name, split)]
            w.writerows(
                zip(
                    [name] * len(y),
                    [scenario] * len(y),
                    [split] * len(y),
                    [seed] * len(y),
                    ids,
                    y,
                    p,
                )
            )
    rng = np.random.default_rng(20261115)
    for name, scenario, split in sorted(
        {(r["method"], r["scenario"], r["split"]) for r in records}
    ):
        group = [
            r
            for r in records
            if (r["method"], r["scenario"], r["split"]) == (name, scenario, split)
        ]
        k = f"{name}/{scenario}/{split}"
        summary["methods"][k] = {
            "accuracy_mean_pct": float(np.mean([r["accuracy_pct"] for r in group])),
            "accuracy_sd_pct": float(np.std([r["accuracy_pct"] for r in group], ddof=1)),
            "macro_f1_mean_pct": float(np.mean([r["macro_f1_pct"] for r in group])),
            "seeds": group,
        }
        if name in ("abrupt", "combined", "torso_zero", "torso_confidence"):
            ids, y = identities[(name, split)]
            baseids, basey = identities[("raw", split)]
            assert np.array_equal(ids, baseids) and np.array_equal(y, basey)
            diff = np.mean(
                [
                    (preds[name, scenario, split, s] == y).astype(int)
                    - (preds["raw", "retrained", split, s] == y).astype(int)
                    for s in range(3)
                ],
                axis=0,
            )
            idx = rng.integers(0, len(y), (10000, len(y)))
            summary["comparisons"][k] = {
                "delta_pp": float(100 * diff.mean()),
                "bootstrap95_pp": np.quantile(100 * diff[idx].mean(1), [0.025, 0.975]).tolist(),
            }
    write(OUT / "summary.json", summary)


if __name__ == "__main__":
    {"prepare": prepare, "train": fit, "evaluate": evaluate}[sys.argv[1]]()
