"""Small CPU runs exercise checkpoint selection and final test evaluation."""

import json
import sys

import numpy as np
import pytest
import torch

import run_action_experiments as experiments
import train_action_recognition as baseline


@pytest.fixture
def synthetic_splits():
    rng = np.random.default_rng(42)

    def sample(seq_id, label):
        return {
            "seq_id": seq_id,
            "label": label,
            "action": "synthetic",
            "features": rng.normal(size=(3 + label, 24)).astype(np.float32),
        }

    test = [sample(f"test-{label}", label) for label in range(2)]
    return {
        "train_gt": [sample(f"train-{label}-{i}", label) for label in range(2) for i in range(3)],
        "test_gt": test,
        "test_raw_yolo": test,
        "test_calib_yolo": test,
    }


def record_evaluations(monkeypatch, module, name):
    original = getattr(module, name)
    calls = []

    def evaluate(model, loader, *args):
        calls.append({sample["seq_id"] for sample in loader.dataset.samples})
        return original(model, loader, *args)

    monkeypatch.setattr(module, name, evaluate)
    return calls


def assert_protocol(calls, result):
    assert len(calls) == 4  # Validation once; three final test modalities.
    assert all(seq_id.startswith("train-") for seq_id in calls[0])
    assert calls[1:] == [{"test-0", "test-1"}] * 3
    assert result["best_epoch"] == 1
    assert result["protocol"]["checkpoint_selection"] == "validation_gt_top1"


@pytest.mark.parametrize(
    "mode", ["exp1_kinematics", "exp2_augmentation", "exp3_spatiotemporal", "exp4_combined"]
)
def test_experiments_use_validation_before_final_test(
    monkeypatch, tmp_path, synthetic_splits, mode
):
    torch.set_num_threads(1)
    monkeypatch.setattr(experiments, "build_or_load_penn_action_splits", lambda: synthetic_splits)
    calls = record_evaluations(monkeypatch, experiments, "evaluate")
    result = experiments.run_experiment(mode, epochs=1, device="cpu", output_dir=str(tmp_path))
    assert_protocol(calls, result)
    assert (tmp_path / f"{mode}_best.pt").exists()
    assert json.loads((tmp_path / f"{mode}_summary.json").read_text()) == result


def test_baseline_uses_validation_before_final_test(monkeypatch, tmp_path, synthetic_splits):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(baseline, "build_or_load_penn_action_splits", lambda: synthetic_splits)
    monkeypatch.setattr(baseline, "plot_training_curves", lambda *args: None)
    monkeypatch.setattr(baseline, "plot_confusion_matrix", lambda *args: None)
    monkeypatch.setattr(
        sys, "argv", ["train_action_recognition.py", "--epochs", "1", "--output_dir", str(tmp_path)]
    )
    calls = record_evaluations(monkeypatch, baseline, "evaluate_model")
    baseline.main()
    result = json.loads((tmp_path / "action_recognition_summary.json").read_text())
    assert_protocol(calls, result)
    assert (tmp_path / "best_bilstm_model.pt").exists()
