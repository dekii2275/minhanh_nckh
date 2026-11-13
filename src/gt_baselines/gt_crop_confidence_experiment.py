"""Confidence/mask x torso-normalization ablation; train/validation only."""

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import confidence_torso_features as features
import gt_crop_correction_experiment as source
import gt_crop_paper_features_experiment as comparisons
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

base = source.base
ROOT = source.ROOT
OUT = ROOT / "outputs/gt_crop_confidence_torso_20261103"
METHODS = {
    "baseline": 24,
    "mean5": 24,
    "confidence_mask": 48,
    "torso": 24,
    "torso_confidence_mask": 48,
    "capacity_control": 48,
    "torso_capacity_control": 48,
}
NEW_METHODS = tuple(m for m in METHODS if m not in ("baseline", "mean5"))
SEEDS = (0, 1, 2)


def old_dir(method, seed):
    name = "fixed_swap" if method == "baseline" else "mean5"
    return source.DEFAULT_OUT / "retrained" / name / "bilstm_attention" / f"seed{seed}"


def protocol():
    return {
        "methods": METHODS,
        "new_methods": NEW_METHODS,
        "seeds": SEEDS,
        "train": {
            "epochs": 50,
            "batch": 32,
            "optimizer": "Adam lr=.001 wd=.0001",
            "scheduler": "CosineAnnealingLR T_max=50 eta_min=.00001",
            "clip_norm": 1.0,
            "selection": "highest validation accuracy, earliest tie",
        },
        "scope": "1005 train / 253 validation, fixed seed0 split, 15 classes; no official test",
        "confidence": "12 clipped scores, zero for absent coordinates; linearly resampled to 64",
        "mask": "12 binary masks: both contributing native joints finite positive and confidence>=.30",
        "torso": "one clip-median torso length from reliable shoulders/hips, >5 pixels; fallback median shoulder width, reliable radius, then 50px",
        "root": "reliable paired hips, linearly interpolate root with endpoint hold; if no paired hips use reliable torso/all-joint centroids; zero if no reliable joints",
        "missing": "preserve all observed predicted joints including low-confidence; absent coordinates zero before resampling; never fill joint coordinates",
        "mapping": "fixed left/right swap once, same as source",
        "capacity_controls": "24 normalized coordinates plus 24 zeros; identical model dimensions/initialization shape to confidence+mask",
        "information": "GT bbox crop/IoU selection retained; transforms and confidence use predicted pose only, no GT joint/visibility/label input",
        "script_sha256": base.sha(Path(__file__)),
        "features_script_sha256": base.sha(Path(features.__file__)),
        "classifier_sha256": base.sha(ROOT / "src/models/action_bilstm.py"),
        "manifest_sha256": base.sha(base.MANIFEST),
        "source_protocol_sha256": base.sha(source.DEFAULT_OUT / "protocol.json"),
        "source_input_manifest_sha256": base.sha(source.DEFAULT_OUT / "input_manifest.json"),
        "source_feature_sha256": {
            m: base.sha(source.DEFAULT_OUT / f"features_{m}.npz") for m in ("fixed_swap", "mean5")
        },
        "source_classifier_hashes": {
            m: {str(s): base.sha(old_dir(m, s) / "best.pt") for s in SEEDS}
            for m in ("baseline", "mean5")
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "caution": "exploratory reused validation, GT-assisted target selection; clip scale/root interpolation use future frames; no independent-test or online claim",
    }


def check(out):
    if source.read_json(out / "protocol.json") != json.loads(json.dumps(protocol())):
        raise ValueError("Protocol changed; choose a new output directory")


def prepare(out):
    source.check_protocol(source.DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=False)
    source.write_json(out / "protocol.json", protocol())
    inputs = source.read_json(source.DEFAULT_OUT / "input_manifest.json")
    accumulated = {m: [] for m in METHODS}
    diagnostics = []
    with (
        np.load(source.DEFAULT_OUT / "features_fixed_swap.npz") as raw,
        np.load(source.DEFAULT_OUT / "features_mean5.npz") as smooth,
    ):
        ids, labels, splits = raw["ids"], raw["y"], raw["split"]
        if ids.tolist() != [r["id"] for r in inputs] or not np.array_equal(ids, smooth["ids"]):
            raise ValueError("Input order mismatch")
        for i, row in enumerate(inputs):
            path = source.source.OUT / "poses" / f"{row['id']}.npz"
            if base.sha(path) != row["pose_sha256"]:
                raise ValueError("Pose source changed")
            with np.load(path) as cache:
                if str(cache["annotation_sha256"]) != row["annotation_sha256"]:
                    raise ValueError("Annotation mismatch")
                xy, conf = cache["xy"][:, source.source.SWAP], cache["conf"][:, source.source.SWAP]
                variants, stats = features.make_variants(raw["x"][i], xy, conf)
            variants["mean5"] = smooth["x"][i]
            for method in METHODS:
                x = variants[method]
                if x.shape != (64, METHODS[method]) or not np.isfinite(x).all():
                    raise ValueError("Invalid features " + method)
                accumulated[method].append(x)
            diagnostics.append({"id": row["id"], "split": row["split"], **stats})
            if (i + 1) % 250 == 0:
                print(f"prepared {i + 1}/{len(inputs)}", flush=True)
        hashes = {}
        for method, items in accumulated.items():
            path = out / f"features_{method}.npz"
            np.savez_compressed(path, x=np.stack(items), ids=ids, y=labels, split=splits)
            hashes[method] = base.sha(path)
            print(f"features {method} {np.stack(items).shape}", flush=True)
    source.write_json(out / "normalization_diagnostics.json", diagnostics)
    source.write_json(out / "prepared.json", {"feature_sha256": hashes})


def data_for(out, method):
    path = out / f"features_{method}.npz"
    if base.sha(path) != source.read_json(out / "prepared.json")["feature_sha256"][method]:
        raise ValueError("Feature hash mismatch")
    with np.load(path) as a:
        result = {
            s: [
                (torch.from_numpy(x.copy()), int(y), str(sid))
                for x, y, sid, split in zip(a["x"], a["y"], a["ids"], a["split"])
                if split == s
            ]
            for s in ("train", "validation")
        }
    if len(result["train"]) != 1005 or len(result["validation"]) != 253:
        raise ValueError("Split mismatch")
    return result


def model_for(method):
    return source.trainer.ActionBiLSTMAttentionClassifier(input_size=METHODS[method])


def train_run(out, method, seed, data):
    directory = out / "trained" / method / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_for(method).to(device)
    tr = DataLoader(
        data["train"], batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    va = DataLoader(data["validation"], batch_size=32, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0.00001)
    criterion = nn.CrossEntropyLoss()
    best = -1.0
    history = []
    start = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        loss_sum = 0.0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(y)
        scheduler.step()
        model.eval()
        pred = []
        target = []
        ids = []
        probabilities = []
        with torch.inference_mode():
            for x, y, sid in va:
                z = model(x.to(device))
                pred.extend(z.argmax(-1).cpu().tolist())
                target.extend(y.tolist())
                ids.extend(sid)
                probabilities.extend(z.softmax(-1).cpu().tolist())
        metrics = base.metric(target, pred)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(data["train"]), **metrics})
        if metrics["accuracy_pct"] > best:
            best = metrics["accuracy_pct"]
            torch.save(model.state_dict(), directory / "best.pt")
            source.write_json(
                directory / "best_validation.json",
                {
                    "method": method,
                    "seed": seed,
                    "epoch": epoch,
                    **metrics,
                    "predictions": [
                        {"id": sid, "target": y, "predicted": p, "probabilities": prob}
                        for sid, y, p, prob in zip(ids, target, pred, probabilities)
                    ],
                },
            )
        source.write_json(directory / "history.json", history)
        print(
            f"{method} seed={seed} epoch={epoch}/50 val={metrics['accuracy_pct']:.2f} best={best:.2f} seconds={time.perf_counter() - start:.1f}",
            flush=True,
        )
    source.write_json(
        directory / "provenance.json",
        {
            "method": method,
            "seed": seed,
            "epochs": 50,
            "parameters": sum(p.numel() for p in model.parameters()),
            "input_size": METHODS[method],
            "checkpoint_sha256": base.sha(directory / "best.pt"),
            "protocol_sha256": base.sha(out / "protocol.json"),
            "feature_sha256": base.sha(out / f"features_{method}.npz"),
            "seconds": time.perf_counter() - start,
            "train_ids": [r[2] for r in data["train"]],
            "validation_ids": [r[2] for r in data["validation"]],
        },
    )


def predict(out, method, seed, dataset):
    directory = (
        old_dir(method, seed)
        if method in ("baseline", "mean5")
        else out / "trained" / method / f"seed{seed}"
    )
    meta = source.read_json(directory / "provenance.json")
    path = directory / "best.pt"
    if base.sha(path) != meta["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    model = model_for(method)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    x = torch.stack([r[0] for r in dataset])
    pred = []
    with torch.inference_mode():
        for batch in x.split(32):
            pred.extend(model(batch).argmax(-1).tolist())
    saved = source.read_json(directory / "best_validation.json")
    predictions = [{"id": r[2], "target": r[1], "predicted": p} for r, p in zip(dataset, pred)]
    if len(predictions) != len(saved["predictions"]) or any(
        p[k] != q[k]
        for p, q in zip(predictions, saved["predictions"])
        for k in ("id", "target", "predicted")
    ):
        raise ValueError("CPU checkpoint reload mismatch")
    return {
        "method": method,
        "seed": seed,
        "epoch": saved["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": base.sha(path),
        **base.metric([r[1] for r in dataset], pred),
        "predictions": predictions,
    }


def summarize(out):
    records = []
    for method in METHODS:
        data = data_for(out, method)["validation"]
        for seed in SEEDS:
            records.append(predict(out, method, seed, data))
            print(
                f"verified {method} seed={seed} accuracy={records[-1]['accuracy_pct']:.4f}",
                flush=True,
            )
    groups = {m: [r for r in records if r["method"] == m] for m in METHODS}
    comparison = comparisons.comparison
    summary = {
        "methods_vs_baseline": {m: comparison(r, groups["baseline"]) for m, r in groups.items()},
        "methods_vs_mean5": {m: comparison(r, groups["mean5"]) for m, r in groups.items()},
        "confidence_vs_capacity_control": comparison(
            groups["confidence_mask"], groups["capacity_control"]
        ),
        "combined_vs_torso_capacity_control": comparison(
            groups["torso_confidence_mask"], groups["torso_capacity_control"]
        ),
        "combined_vs_confidence": comparison(
            groups["torso_confidence_mask"], groups["confidence_mask"]
        ),
        "combined_vs_torso": comparison(groups["torso_confidence_mask"], groups["torso"]),
        "train_videos": 1005,
        "validation_videos": 253,
        "caution": "Exploratory reused validation, GT bbox assistance. Bootstrap video pairs with seeds averaged; excludes model selection/seed uncertainty, not multiplicity-corrected. No official test.",
    }
    source.write_json(out / "results.json", records)
    source.write_json(out / "summary.json", summary)
    source.write_json(
        out / "verification.json",
        {
            "runs": len(records),
            "new_checkpoints": len(NEW_METHODS) * len(SEEDS),
            "cpu_prediction_matches": sum(len(r["predictions"]) for r in records),
        },
    )
    events = []
    for r in records:
        b = groups["baseline"][r["seed"]]
        for p, q in zip(r["predictions"], b["predictions"]):
            events.append(
                {
                    "method": r["method"],
                    "seed": r["seed"],
                    **p,
                    "baseline_predicted": q["predicted"],
                    "correctness_delta": int(p["predicted"] == p["target"])
                    - int(q["predicted"] == q["target"]),
                }
            )
    with (out / "per_video_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0]))
        w.writeheader()
        w.writerows(events)
    print(
        json.dumps(
            {
                m: {k: v for k, v in r.items() if k != "seeds"}
                for m, r in summary["methods_vs_baseline"].items()
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--stage", choices=("prepare", "train", "summarize", "all"), required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage in ("prepare", "all"):
        prepare(args.output)
    check(args.output)
    if args.stage in ("train", "all"):
        for method in NEW_METHODS:
            data = data_for(args.output, method)
            for seed in SEEDS:
                directory = args.output / "trained" / method / f"seed{seed}"
                if directory.exists():
                    meta = source.read_json(directory / "provenance.json")
                    if (
                        meta["protocol_sha256"] != base.sha(args.output / "protocol.json")
                        or len(source.read_json(directory / "history.json")) != 50
                    ):
                        raise ValueError("Partial or changed old training run")
                    continue
                train_run(args.output, method, seed, data)
    if args.stage in ("summarize", "all"):
        summarize(args.output)


if __name__ == "__main__":
    main()
