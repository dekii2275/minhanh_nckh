"""Run three Penn input representations using CascadeFormer on CUDA."""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import retrain_unified_20261115 as experiment
import torch
import train_gt_baselines as trainer
from torch import nn
from train_gt_cascade_scratch import SOURCE, CascadeScratch

ROOT = trainer.ROOT
OLD = ROOT / "outputs/unified_retrain_20261115"
OUT = ROOT / "outputs/cascade_three_inputs_20261123"
METHODS = {
    "raw": (None, None, 24),
    "torso_confidence": (None, None, 48),
    "gt_cascade": (None, None, 24),
}


def model(name, dimension):
    net = CascadeScratch()
    if dimension == 48:
        net.t1.embedding = nn.Linear(48, 256)
    return net


def prepare():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if OUT.exists():
        raise FileExistsError(OUT)
    feature_hashes = {}
    reference = None
    for name in METHODS:
        path = OLD / f"features_{name}.npz"
        feature_hashes[name] = trainer.sha(path)
        for seed in range(3):
            metadata = json.loads(
                (OLD / "trained" / name / f"seed{seed}" / "provenance.json").read_text()
            )
            if feature_hashes[name] != metadata["feature_sha256"]:
                raise ValueError(f"Feature hash mismatch for {name}, seed {seed}")
        with np.load(path) as features:
            values = {
                key: features[key].copy() for key in ("ids", "y", "split", "test_ids", "test_y")
            }
            for key in ("x", "test_x"):
                if (
                    features[key].shape[-1] != METHODS[name][2]
                    or not np.isfinite(features[key]).all()
                ):
                    raise ValueError(f"Invalid features {name}/{key}")
        if reference is None:
            reference = values
        else:
            for ids_key, label_keys in (("ids", ("y", "split")), ("test_ids", ("test_y",))):
                current = {
                    str(video): tuple(str(values[k][i]) for k in label_keys)
                    for i, video in enumerate(values[ids_key])
                }
                expected = {
                    str(video): tuple(str(reference[k][i]) for k in label_keys)
                    for i, video in enumerate(reference[ids_key])
                }
                if len(current) != len(values[ids_key]) or current != expected:
                    raise ValueError(f"Split mismatch for {name}/{ids_key}")
    OUT.mkdir(parents=True)
    for name in METHODS:
        shutil.copyfile(OLD / f"features_{name}.npz", OUT / f"features_{name}.npz")
        if trainer.sha(OUT / f"features_{name}.npz") != feature_hashes[name]:
            raise ValueError(f"Copied feature hash mismatch: {name}")
    protocol = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"train": 1005, "validation": 253, "test": 1068},
        "methods": list(METHODS),
        "input_dimensions": {name: spec[2] for name, spec in METHODS.items()},
        "adaptation": "CascadeScratch 12 joints; 48-channel branch replaces T1 input projection with Linear(48,256); no masked pretraining",
        "seed": [0, 1, 2],
        "epochs": 50,
        "batch": 32,
        "selection": "highest validation accuracy; earliest epoch on ties",
        "optimizer": "Adam lr=.001 weight_decay=.0001 cosine T=50 eta_min=.00001 gradient_clip=1",
        "feature_sha256": feature_hashes,
        "manifest_sha256": trainer.sha(trainer.MANIFEST),
        "script_sha256": trainer.sha(Path(__file__)),
        "training_script_sha256": trainer.sha(Path(experiment.__file__)),
        "wrapper_sha256": trainer.sha(ROOT / "src/gt_baselines/train_gt_cascade_scratch.py"),
        "author_source_sha256": {
            name: trainer.sha(SOURCE / name) for name in ("pretraining.py", "finetuning.py")
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "test_history": "Official test previously used in development; exploratory retrospective evaluation",
        "pose_conditions": "GT bounding box assists YOLO crop/person selection; offline; annotation is a separate reference input",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Prepared verified feature inputs; CUDA:", torch.cuda.get_device_name(0), flush=True)


def main():
    experiment.OUT = OUT
    experiment.METHODS = METHODS
    experiment.model = model
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "train":
        experiment.fit()
    elif sys.argv[1] == "evaluate":
        experiment.evaluate()
    else:
        raise ValueError(sys.argv[1])


if __name__ == "__main__":
    main()
