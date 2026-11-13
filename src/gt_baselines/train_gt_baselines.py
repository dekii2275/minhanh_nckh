"""Exploratory Penn GT baselines; fixed seed-0 manifest, no test access."""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.action_dataset import ACTION_CLASSES, normalize_skeleton_sequence  # noqa: E402
from models.action_bilstm import ActionBiLSTMAttentionClassifier  # noqa: E402
from train_action_recognition import compute_metrics  # noqa: E402

MANIFEST = ROOT / "report/02112026_nckh/penn_audit_artifacts/video_manifest.csv"
LABELS = ROOT / "data/raw/Penn_Action/labels"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig", newline="")))
    datasets = {"train": [], "validation": []}
    for row in rows:
        split = row["seed0_split"]
        if split not in datasets:
            continue
        p = LABELS / (row["id"] + ".mat")
        if sha(p) != row["label_sha256"]:
            raise ValueError("Annotation hash mismatch: " + row["id"])
        m = sio.loadmat(p)
        if str(m["action"][0]) != row["action"] or int(m["train"][0, 0]) != 1:
            raise ValueError("Label/split mismatch: " + row["id"])
        x, y, v = (np.asarray(m[k])[:, 1:13] for k in ("x", "y", "visibility"))
        xy = np.stack((x, y), axis=-1).astype(np.float32)
        xy[(v == 0) | (x <= 0)] = np.nan
        f = normalize_skeleton_sequence(xy)
        # Uniform linear resampling, after the historical per-frame normalization.
        t = np.linspace(0, len(f) - 1, 64)
        f = np.stack([np.interp(t, np.arange(len(f)), f[:, j]) for j in range(24)], axis=-1).astype(
            np.float32
        )
        if not np.isfinite(f).all():
            raise ValueError("Non-finite: " + row["id"])
        datasets[split].append(
            (torch.from_numpy(f), ACTION_CLASSES.index(row["action"]), row["id"])
        )
    assert len(datasets["train"]) == 1005 and len(datasets["validation"]) == 253
    return datasets


class Recurrent(nn.Module):
    def __init__(self, kind):
        super().__init__()
        bidir = kind == "bilstm"
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(24, 128, num_layers=2, dropout=0.3, bidirectional=bidir, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * (2 if bidir else 1), 15))

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y.mean(dim=1))


def model_for(kind):
    if kind == "bilstm_attention":
        return ActionBiLSTMAttentionClassifier()
    return Recurrent(kind)


def run(kind, seed, data, args, out):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_for(kind).to(device)
    generator = torch.Generator().manual_seed(seed)
    tr = DataLoader(data["train"], batch_size=args.batch, shuffle=True, generator=generator)
    va = DataLoader(data["validation"], batch_size=args.batch, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.00001)
    criterion = nn.CrossEntropyLoss()
    best = -1
    history = []
    start = time.time()
    run_dir = out / kind / ("seed" + str(seed))
    run_dir.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(y)
        sched.step()
        model.eval()
        preds = []
        targets = []
        ids = []
        probs = []
        with torch.inference_mode():
            for x, y, sid in va:
                z = model(x.to(device))
                preds.extend(z.argmax(-1).cpu().tolist())
                targets.extend(y.tolist())
                ids.extend(sid)
                probs.extend(z.softmax(-1).cpu().tolist())
        acc, f1, _, cm = compute_metrics(targets, preds)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(data["train"]),
                "val_accuracy": acc,
                "val_macro_f1": f1,
            }
        )
        if acc > best:
            best = acc
            torch.save(model.state_dict(), run_dir / "best.pt")
            result = {
                "model": kind,
                "seed": seed,
                "epoch": epoch,
                "accuracy": acc,
                "macro_f1": f1,
                "confusion_matrix": cm.tolist(),
                "predictions": [
                    {"id": i, "target": a, "predicted": b, "probabilities": p}
                    for i, a, b, p in zip(ids, targets, preds, probs)
                ],
            }
            (run_dir / "best_validation.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"{kind} seed={seed} epoch={epoch}/{args.epochs} loss={loss_sum / len(data['train']):.4f} val={acc:.2f} f1={f1:.2f} best={best:.2f} seconds={time.time() - start:.1f}",
            flush=True,
        )
    meta = {
        "model": kind,
        "seed": seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "torch": torch.__version__,
        "seconds": time.time() - start,
        "checkpoint_sha256": sha(run_dir / "best.pt"),
        "manifest_sha256": sha(MANIFEST),
        "script_sha256": sha(Path(__file__)),
        "selection": "highest validation top1, earliest tie",
        "train_ids": [x[2] for x in data["train"]],
        "validation_ids": [x[2] for x in data["validation"]],
    }
    (run_dir / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "bilstm", "gru", "bilstm_attention"],
        choices=["lstm", "bilstm", "gru", "bilstm_attention"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_data()
    for kind in args.models:
        for seed in args.seeds:
            run(kind, seed, data, args, out)


if __name__ == "__main__":
    main()
