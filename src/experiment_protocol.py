"""Reproducible sequence splits and metadata for action recognition experiments."""

import random
from collections import defaultdict


def prepare_penn_splits(splits, validation_fraction=0.2, seed=42):
    """Hold out validation sequences from train; keep all official test modalities aligned.

    Split whole videos within each action class. A class needs at least two videos
    so it remains represented in both training and validation. Input caches are
    never mutated.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    indexed = {}
    for key in ("train_gt", "test_gt", "test_raw_yolo", "test_calib_yolo"):
        samples = splits[key]
        by_id = {sample["seq_id"]: sample for sample in samples}
        if not samples or len(by_id) != len(samples):
            raise ValueError(f"{key} must contain nonempty, unique sequence IDs")
        indexed[key] = by_id

    test_ids = set(indexed["test_gt"])
    if test_ids & set(indexed["train_gt"]):
        raise ValueError("Official train and test sequence IDs overlap")
    for key in ("test_raw_yolo", "test_calib_yolo"):
        if set(indexed[key]) != test_ids:
            raise ValueError(f"{key} must match the GT test sequence IDs")
        if any(indexed[key][sid]["label"] != indexed["test_gt"][sid]["label"] for sid in test_ids):
            raise ValueError(f"{key} labels do not match GT test labels")

    by_label = defaultdict(list)
    for sample in splits["train_gt"]:
        by_label[sample["label"]].append(sample)
    rng = random.Random(seed)
    train, validation = [], []
    for label in sorted(by_label):
        samples = sorted(by_label[label], key=lambda sample: sample["seq_id"])
        if len(samples) < 2:
            raise ValueError(f"Class {label} needs at least two training sequences")
        rng.shuffle(samples)
        count = max(1, min(len(samples) - 1, round(len(samples) * validation_fraction)))
        validation.extend(samples[:count])
        train.extend(samples[count:])

    result = dict(splits)
    result["train_gt"] = train
    result["validation_gt"] = validation
    return result


def protocol_metadata(splits, validation_fraction, seed):
    """Store the exact split IDs alongside metrics for later reproducibility."""
    return {
        "version": 2,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "checkpoint_selection": "validation_gt_top1",
        "train_modality": "ground_truth",
        "sequence_ids": {
            key: sorted(sample["seq_id"] for sample in splits[key])
            for key in ("train_gt", "validation_gt", "test_gt", "test_raw_yolo", "test_calib_yolo")
        },
    }
