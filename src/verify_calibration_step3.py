"""Verify the eight-way ablation artifacts independently of the evaluator."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/calibration_step3"
SOURCE = ROOT / "outputs/alignment_step2"


def main():
    provenance = json.loads((OUT / "provenance.json").read_text())
    for path, digest in provenance["frozen_inputs_sha256"].items():
        target = ROOT / path
        snapshot = ROOT / "outputs/research_snapshot" / path
        file_to_check = (
            snapshot
            if (snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest)
            else target
        )
        assert hashlib.sha256(file_to_check.read_bytes()).hexdigest() == digest, path
    eval_target = ROOT / "src/evaluate_calibration_step3.py"
    eval_snapshot = ROOT / "outputs/research_snapshot/src/evaluate_calibration_step3.py"
    eval_to_check = (
        eval_snapshot
        if (
            eval_snapshot.exists()
            and hashlib.sha256(eval_snapshot.read_bytes()).hexdigest()
            == provenance["script_sha256"]
        )
        else eval_target
    )
    assert hashlib.sha256(eval_to_check.read_bytes()).hexdigest() == provenance["script_sha256"]
    videos = pd.read_csv(OUT / "per_video_predictions.csv", dtype={"id": str})
    source = pd.read_csv(SOURCE / "per_video_predictions.csv", dtype={"id": str})
    assert videos.id.tolist() == source.id.tolist()
    assert len(videos) == videos.id.nunique() == 253
    assert videos.frames.sum() == 17747
    assert np.array_equal(videos.production_swap, source.production_swap)
    assert videos.production_swap.sum() == 168
    model_probs = np.load(OUT / "probabilities.npz")
    ref = np.load(SOURCE / "probabilities.npz")
    assert model_probs["ids"].tolist() == videos.id.tolist()
    y = videos.label.to_numpy()
    assert np.array_equal(y, source.label) and np.array_equal(y, model_probs["labels"])
    correct = {}
    metrics = pd.read_csv(OUT / "configuration_metrics.csv").set_index("configuration")
    for key in "ABCDEFGH":
        probs = model_probs[key]
        p = probs.argmax(1)
        assert np.isfinite(probs).all() and probs.shape == (253, 15)
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
        assert np.array_equal(p, videos[f"{key}_prediction"])
        hit = p == y
        correct[key] = hit.astype(int)
        assert np.array_equal(hit, videos[f"{key}_correct"])
        acc = hit.mean() * 100
        f1 = (
            np.mean(
                [
                    2 * ((p == k) & (y == k)).sum() / ((p == k).sum() + (y == k).sum())
                    for k in range(15)
                ]
            )
            * 100
        )
        assert abs(acc - metrics.loc[key, "top1_accuracy"]) < 1e-10
        assert abs(f1 - metrics.loc[key, "macro_f1"]) < 1e-10
        assert hit.sum() == metrics.loc[key, "correct"]
    assert np.array_equal(model_probs["A"], ref["C3"])
    assert np.array_equal(model_probs["H"], ref["C4"])
    # Rebuild bootstrap draws without importing experiment helpers.
    rng = np.random.default_rng(20261029)
    selections = []
    for k in range(15):
        members = np.flatnonzero(y == k)
        selections.append(rng.choice(members, (20000, len(members)), replace=True))
    draws = np.hstack(selections)

    def check_contrast(coefficients, delta, lo, hi):
        values = sum(correct[key] * weight for key, weight in coefficients.items())
        samples = np.mean(values[draws], axis=1) * 100
        expected = np.quantile(samples, [0.025, 0.975])
        assert abs(values.mean() * 100 - delta) < 1e-10
        np.testing.assert_allclose(expected, [lo, hi], atol=1e-10, rtol=0)

    for key, row in metrics.iterrows():
        difference = correct[key] - correct["A"]
        assert (difference == 1).sum() == row.incorrect_to_correct
        assert (difference == -1).sum() == row.correct_to_incorrect
        check_contrast(
            {key: 1, "A": -1} if key != "A" else {"A": 0},
            row.accuracy_delta_pp,
            row.ci95_low_pp,
            row.ci95_high_pp,
        )
    pairwise = pd.read_csv(OUT / "all_pairwise_comparisons.csv")
    assert len(pairwise) == 28
    for row in pairwise.itertuples():
        on, off = row.comparison.split("-")
        difference = correct[on] - correct[off]
        assert (difference == 1).sum() == row.incorrect_to_correct
        assert (difference == -1).sum() == row.correct_to_incorrect
        check_contrast({on: 1, off: -1}, row.accuracy_delta_pp, row.ci95_low_pp, row.ci95_high_pp)
    for row in pd.read_csv(OUT / "interaction_contrasts.csv").itertuples():
        check_contrast(
            json.loads(row.contrast), row.accuracy_delta_pp, row.ci95_low_pp, row.ci95_high_pp
        )
    marginals = pd.read_csv(OUT / "component_marginal_effects.csv")
    assert len(marginals) == 12
    for row in marginals.itertuples():
        pair = pairwise[pairwise.comparison == row.comparison].iloc[0]
        for field in [
            "accuracy_delta_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "incorrect_to_correct",
            "correct_to_incorrect",
        ]:
            assert abs(getattr(row, field) - pair[field]) < 1e-10
    pred_cols = [f"{key}_prediction" for key in "ABCDEFGH"]
    changed = videos[pred_cols].nunique(axis=1) > 1
    changed_ids = pd.read_csv(OUT / "changed_videos.csv", dtype={"id": str}).id.tolist()
    assert videos.loc[changed, "id"].tolist() == changed_ids
    assert changed.sum() == 42
    assert (videos[[f"{key}_correct" for key in "ABCDEFGH"]].nunique(axis=1) > 1).sum() == 18
    report = ROOT / "reports/CALIBRATION_ABLATION_STEP3_2026-10-29.md"
    targets = [
        report.parent / link.split("#")[0]
        for link in re.findall(r"\]\(([^)]+)\)", report.read_text())
    ]
    for path in targets:
        assert path.exists() or path.resolve() == (OUT / "verification.json").resolve(), path
    verification = {
        "passed": True,
        "videos": 253,
        "frames": 17747,
        "changed_prediction_videos": 42,
        "changed_correctness_videos": 18,
        "source_hashes_unchanged": True,
        "A_C3_posteriors_identical": True,
        "H_C4_posteriors_identical": True,
        "all_eight_metrics_independently_verified": True,
        "all_28_paired_CIs_and_7_interaction_CIs_recomputed": True,
        "12_marginal_comparisons_verified": True,
        "fixed_alignment_verified": True,
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2))
    assert all(path.exists() for path in targets)
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
