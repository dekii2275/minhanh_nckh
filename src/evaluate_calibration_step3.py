"""Frozen 2^3 calibration ablation on Penn validation; alignment loaded from Step 2."""

import itertools
import json

import numpy as np
import pandas as pd
import torch

from data.action_dataset import ACTION_CLASSES, normalize_skeleton_sequence
from evaluate_alignment_step2 import ROOT, SWAP, as_df, classifier, coords, dump, predict, sha
from skeleton_calibrator import FullCalibrationPipeline
from train_action_recognition import compute_metrics

SOURCE = ROOT / "outputs/alignment_step2"
OUT = ROOT / "outputs/calibration_step3"
CONFIGS = {
    "A": (False, False, False),
    "B": (True, False, False),
    "C": (False, True, False),
    "D": (False, False, True),
    "E": (True, True, False),
    "F": (True, False, True),
    "G": (False, True, True),
    "H": (True, True, True),
}


def transform(raw, enabled, swap):
    """Never call the heuristic. Preserve production order and apply saved swap last."""
    pipeline = FullCalibrationPipeline(lr_mode=None)
    assert pipeline.stage4_lr is None
    result = raw.copy()
    stages = [pipeline.stage1_interp, pipeline.stage2_bone, pipeline.stage3_smooth]
    for active, stage in zip(enabled, stages):
        if active:
            result = stage.calibrate_sequence(result)
    body = coords(result)
    return body[:, SWAP].copy() if swap else body.copy()


def changed(a, b):
    return ~(np.isclose(a, b, rtol=0, atol=1e-6, equal_nan=True))


def check_sources():
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    paths = {
        "checkpoint_sha256": ROOT / provenance["checkpoint"],
        "calibrator_sha256": ROOT / "src/skeleton_calibrator.py",
        "classifier_code_sha256": ROOT / "src/models/action_bilstm.py",
        "feature_code_sha256": ROOT / "src/data/action_dataset.py",
        "script_sha256": ROOT / "src/evaluate_alignment_step2.py",
    }
    for key, path in paths.items():
        assert sha(path) == provenance[key], path
    manifest = json.loads((SOURCE / "validation_manifest.json").read_text())
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    saved = pd.read_csv(SOURCE / "per_video_predictions.csv", dtype={"id": str})
    decisions = pd.read_csv(SOURCE / "alignment_decisions.csv", dtype={"id": str})
    ids = [item["id"] for item in manifest]
    assert len(ids) == len(set(ids)) == 253
    assert sum(x["frames"] for x in manifest) == 17747
    assert set(ids) == set(protocol["validation_ids"])
    assert set(ids).isdisjoint(protocol["train_ids"])
    assert all(x["official_train"] == 1 for x in manifest)
    assert decisions.id.tolist() == saved.id.tolist() == ids
    assert np.array_equal(decisions.production_swap, saved.production_swap)
    probabilities = np.load(SOURCE / "probabilities.npz")
    assert ids == probabilities["ids"].tolist()
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in paths.values()}
    for name in [
        "provenance.json",
        "protocol.json",
        "validation_manifest.json",
        "alignment_decisions.csv",
        "per_video_predictions.csv",
        "probabilities.npz",
    ]:
        inputs[str((SOURCE / name).relative_to(ROOT))] = sha(SOURCE / name)
    for sid in ids:
        path = SOURCE / "raw_predictions" / f"{sid}.npz"
        inputs[str(path.relative_to(ROOT))] = sha(path)
    dump(
        OUT / "provenance.json",
        {
            "source": "outputs/alignment_step2",
            "frozen_inputs_sha256": inputs,
            "script_sha256": sha(__file__),
            "configurations": CONFIGS,
            "checkpoint": provenance["checkpoint"],
            "classifier": provenance["classifier"],
            "calibration": provenance["calibration"],
            "evaluation_batch_size": 32,
            "evaluation_cpu_threads": torch.get_num_threads(),
            "alignment": "saved Step-2 production_swap applied last; heuristic never called",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "training": False,
            "tuning": False,
            "official_test": False,
            "bootstrap": {
                "replicates": 20000,
                "seed": 20261029,
                "paired": True,
                "stratified_by": "action",
                "unit": "video",
                "interval": "percentile 95%, unadjusted",
            },
        },
    )
    return manifest, saved, decisions, probabilities


def evaluate(manifest, saved, decisions, reference):
    samples = {key: [] for key in CONFIGS}
    diagnostics = []
    for item, swap in zip(manifest, decisions.production_swap.tolist()):
        sid = item["id"]
        z = np.load(SOURCE / "raw_predictions" / f"{sid}.npz")
        assert z["xy"].shape == (item["frames"], 17, 2)
        raw = as_df(z["xy"], z["conf"])
        aligned = transform(raw, CONFIGS["A"], swap)
        aligned_features = normalize_skeleton_sequence(aligned)
        for key, enabled in CONFIGS.items():
            body = transform(raw, enabled, swap)
            features = normalize_skeleton_sequence(body)
            assert features.shape == (item["frames"], 24) and np.isfinite(features).all()
            samples[key].append({"seq_id": sid, "label": item["label"], "features": features})
            diagnostics.append(
                {
                    "id": sid,
                    "configuration": key,
                    "coordinate_values_changed_vs_A": int(changed(body, aligned).sum()),
                    "feature_values_changed_vs_A": int(changed(features, aligned_features).sum()),
                    "nan_coordinate_values": int(np.isnan(body).sum()),
                    "new_nan_coordinate_values_vs_A": int(
                        (np.isnan(body) & ~np.isnan(aligned)).sum()
                    ),
                    "filled_nan_coordinate_values_vs_A": int(
                        (~np.isnan(body) & np.isnan(aligned)).sum()
                    ),
                    "all_zero_feature_frames": int((features == 0).all(axis=1).sum()),
                }
            )
    model = classifier()
    wide = saved[
        [
            "id",
            "action",
            "label",
            "frames",
            "production_swap",
            "no_detection_frames",
            "multiple_people_frames",
        ]
    ].copy()
    metrics = {}
    posteriors = {}
    checks = {}
    for key in CONFIGS:
        records = predict(model, samples[key])
        assert [r["id"] for r in records] == wide.id.tolist()
        probabilities = np.array([r["probabilities"] for r in records])
        targets = np.array([r["label"] for r in records])
        pred = probabilities.argmax(1)
        assert np.array_equal(targets, wide.label)
        acc, f1, _, cm = compute_metrics(targets, pred, 15)
        wide[f"{key}_prediction"] = pred
        wide[f"{key}_predicted_action"] = [ACTION_CLASSES[x] for x in pred]
        wide[f"{key}_correct"] = pred == targets
        posteriors[key] = probabilities
        metrics[key] = {
            "top1_accuracy": acc,
            "macro_f1": f1,
            "correct": int((pred == targets).sum()),
            "videos": len(pred),
        }
        np.savetxt(OUT / f"{key}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        if key in ["A", "H"]:
            source_key = {"A": "C3", "H": "C4"}[key]
            assert np.array_equal(pred, saved[f"{source_key}_prediction"])
            np.testing.assert_allclose(probabilities, reference[source_key], rtol=0, atol=1e-7)
            checks[key] = {
                "reference": source_key,
                "same_predictions": True,
                "max_probability_difference": float(
                    np.max(abs(probabilities - reference[source_key]))
                ),
                "top1_accuracy": acc,
                "macro_f1": f1,
            }
        print(key, metrics[key], flush=True)
    dump(OUT / "reproduction_checks.json", checks)
    np.savez_compressed(
        OUT / "probabilities.npz",
        ids=wide.id.to_numpy(dtype=str),
        labels=wide.label.to_numpy(),
        **posteriors,
    )
    pd.DataFrame(diagnostics).to_csv(OUT / "transformation_diagnostics.csv", index=False)
    return wide, metrics


def bootstrap_indices(labels):
    rng = np.random.default_rng(20261029)
    return np.concatenate(
        [
            rng.choice(
                np.where(labels == k)[0], size=(20000, int((labels == k).sum())), replace=True
            )
            for k in range(15)
        ],
        axis=1,
    )


def statistics(wide, metrics):
    y = wide.label.to_numpy()
    indices = bootstrap_indices(y)
    correct = {key: wide[f"{key}_correct"].to_numpy(dtype=int) for key in CONFIGS}

    def contrast(weights):
        per_video = sum(value * correct[key] for key, value in weights.items())
        boot = per_video[indices].mean(1) * 100
        low, high = np.percentile(boot, [2.5, 97.5])
        return {
            "accuracy_delta_pp": float(per_video.mean() * 100),
            "ci95_low_pp": float(low),
            "ci95_high_pp": float(high),
        }

    baseline = []
    for key in CONFIGS:
        diff = correct[key] - correct["A"]
        row = {
            "configuration": key,
            **metrics[key],
            **contrast({key: 1, "A": -1} if key != "A" else {"A": 0}),
            "incorrect_to_correct": int((diff == 1).sum()),
            "correct_to_incorrect": int((diff == -1).sum()),
            "macro_f1_delta_pp": metrics[key]["macro_f1"] - metrics["A"]["macro_f1"],
        }
        baseline.append(row)
    pd.DataFrame(baseline).to_csv(OUT / "configuration_metrics.csv", index=False)
    by_bits = {tuple(value): key for key, value in CONFIGS.items()}
    marginal = []
    for component, axis in [("PCHIP", 0), ("Bone", 1), ("OneEuro", 2)]:
        for key, bits in CONFIGS.items():
            if bits[axis]:
                continue
            on = list(bits)
            on[axis] = True
            other = by_bits[tuple(on)]
            diff = correct[other] - correct[key]
            marginal.append(
                {
                    "component": component,
                    "comparison": f"{other}-{key}",
                    "without": key,
                    "with": other,
                    **contrast({other: 1, key: -1}),
                    "macro_f1_delta_pp": metrics[other]["macro_f1"] - metrics[key]["macro_f1"],
                    "incorrect_to_correct": int((diff == 1).sum()),
                    "correct_to_incorrect": int((diff == -1).sum()),
                }
            )
    pd.DataFrame(marginal).to_csv(OUT / "component_marginal_effects.csv", index=False)
    interactions = []
    for title, weights in [
        ("PCHIP:Bone | OneEuro=off", {"E": 1, "B": -1, "C": -1, "A": 1}),
        ("PCHIP:OneEuro | Bone=off", {"F": 1, "B": -1, "D": -1, "A": 1}),
        ("Bone:OneEuro | PCHIP=off", {"G": 1, "C": -1, "D": -1, "A": 1}),
        ("PCHIP:Bone | OneEuro=on", {"H": 1, "F": -1, "G": -1, "D": 1}),
        ("PCHIP:OneEuro | Bone=on", {"H": 1, "E": -1, "G": -1, "C": 1}),
        ("Bone:OneEuro | PCHIP=on", {"H": 1, "E": -1, "F": -1, "B": 1}),
        ("three_way", {"H": 1, "E": -1, "F": -1, "G": -1, "B": 1, "C": 1, "D": 1, "A": -1}),
    ]:
        interactions.append(
            {"interaction": title, "contrast": json.dumps(weights), **contrast(weights)}
        )
    pd.DataFrame(interactions).to_csv(OUT / "interaction_contrasts.csv", index=False)
    # Every pair: the long table makes all changed outcomes inspectable, without selecting examples.
    allpairs = []
    events = []
    for left, right in itertools.combinations(CONFIGS, 2):
        diff = correct[right] - correct[left]
        allpairs.append(
            {
                "comparison": f"{right}-{left}",
                **contrast({right: 1, left: -1}),
                "incorrect_to_correct": int((diff == 1).sum()),
                "correct_to_incorrect": int((diff == -1).sum()),
            }
        )
        for i, row in wide.iterrows():
            p = int(row[f"{left}_prediction"])
            q = int(row[f"{right}_prediction"])
            if p == q:
                continue
            events.append(
                {
                    "id": row.id,
                    "action": row.action,
                    "from_configuration": left,
                    "to_configuration": right,
                    "from_prediction": p,
                    "to_prediction": q,
                    "from_action": ACTION_CLASSES[p],
                    "to_action": ACTION_CLASSES[q],
                    "outcome": "improved"
                    if diff[i] == 1
                    else "harmed"
                    if diff[i] == -1
                    else "both_wrong",
                }
            )
    pd.DataFrame(allpairs).to_csv(OUT / "all_pairwise_comparisons.csv", index=False)
    pd.DataFrame(events).to_csv(OUT / "changed_prediction_events.csv", index=False)
    columns = [f"{key}_prediction" for key in CONFIGS]
    wide["prediction_changed_across_configurations"] = wide[columns].nunique(axis=1) > 1
    wide["correctness_changed_across_configurations"] = (
        wide[[f"{key}_correct" for key in CONFIGS]].nunique(axis=1) > 1
    )
    for key in CONFIGS:
        diff = correct[key] - correct["A"]
        wide[f"{key}_vs_A_outcome"] = np.where(
            diff == 1,
            "improved",
            np.where(
                diff == -1, "harmed", np.where(correct[key] == 1, "both_correct", "both_wrong")
            ),
        )
    wide.to_csv(OUT / "per_video_predictions.csv", index=False)
    wide[wide.prediction_changed_across_configurations].to_csv(
        OUT / "changed_videos.csv", index=False
    )
    classes = []
    for action in ACTION_CLASSES:
        part = wide[wide.action == action]
        for key in CONFIGS:
            classes.append(
                {
                    "action": action,
                    "configuration": key,
                    "videos": len(part),
                    "correct": int(part[f"{key}_correct"].sum()),
                    "top1_accuracy": float(part[f"{key}_correct"].mean() * 100),
                }
            )
    pd.DataFrame(classes).to_csv(OUT / "per_class_accuracy.csv", index=False)
    dump(
        OUT / "results.json",
        {
            "videos": len(wide),
            "frames": int(wide.frames.sum()),
            "configurations": baseline,
            "marginal_effects": marginal,
            "interactions": interactions,
            "changed_prediction_videos": int(wide.prediction_changed_across_configurations.sum()),
            "changed_correctness_videos": int(wide.correctness_changed_across_configurations.sum()),
        },
    )
    print(pd.DataFrame(baseline).to_string(index=False), flush=True)
    print(pd.DataFrame(marginal).to_string(index=False), flush=True)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    # Step 2 YOLO CPU select_device resets PyTorch threads to Ultralytics NUM_THREADS.
    from ultralytics.utils import NUM_THREADS

    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(0)
    np.random.seed(0)
    manifest, saved, decisions, reference = check_sources()
    wide, metrics = evaluate(manifest, saved, decisions, reference)
    statistics(wide, metrics)
    provenance = json.loads((OUT / "provenance.json").read_text())
    for path, expected in provenance["frozen_inputs_sha256"].items():
        assert sha(ROOT / path) == expected, path
    print("PASS: sources unchanged; A=C3, H=C4; no heuristic called.", flush=True)


if __name__ == "__main__":
    main()
