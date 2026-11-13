"""Reload trained checkpoints, verify predictions and paired validation results."""

import json

import compare_person_selection as base
import gt_crop_pose_experiment as e
import numpy as np
import torch
import train_gt_baselines as trainer


def main():
    torch.set_num_threads(4)
    rows = [r for r in e.rows() if r["seed0_split"] == "validation"]
    data = []
    for r in rows:
        d = np.load(e.OUT / "poses" / f"{r['id']}.npz")
        data.append(base.resample(d["xy"][:, e.SWAP]))
    x = torch.tensor(np.stack(data))
    y = np.array([base.ACTION_CLASSES.index(r["action"]) for r in rows])
    source = json.loads((e.OUT / "gt_trained_validation.json").read_text(encoding="utf-8"))
    gt = {r["seed"]: r for r in source if r["variant"] == "fixed_swap"}
    records = []
    diff = []
    for seed in range(3):
        directory = e.OUT / "yolo_trained_fixed_swap" / "bilstm_attention" / f"seed{seed}"
        saved = json.loads((directory / "best_validation.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if base.sha(directory / "best.pt") != meta["checkpoint_sha256"]:
            raise ValueError("checkpoint hash")
        net = trainer.model_for("bilstm_attention")
        net.load_state_dict(
            torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        )
        net.eval()
        with torch.inference_mode():
            pred = torch.cat([net(batch).argmax(-1) for batch in x.split(32)]).numpy()
        assert [r["id"] for r in rows] == [p["id"] for p in saved["predictions"]]
        assert np.array_equal(pred, np.array([p["predicted"] for p in saved["predictions"]]))
        assert [r["id"] for r in rows] == [p["id"] for p in gt[seed]["predictions"]]
        other = np.array([p["predicted"] for p in gt[seed]["predictions"]])
        delta = (pred == y).astype(int) - (other == y).astype(int)
        diff.append(delta)
        records.append(
            {
                "seed": seed,
                "gt_accuracy": gt[seed]["accuracy_pct"],
                "trained_accuracy": saved["accuracy"],
                "trained_macro_f1": saved["macro_f1"],
                "epoch": saved["epoch"],
                "improved_videos": int((delta > 0).sum()),
                "worsened_videos": int((delta < 0).sum()),
                "checkpoint_sha256": meta["checkpoint_sha256"],
            }
        )
    diff = np.stack(diff).mean(axis=0)
    rng = np.random.default_rng(0)
    interval = np.quantile(
        diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1) * 100, [0.025, 0.975]
    ).tolist()
    output = {
        "seeds": records,
        "gt_accuracy_mean": float(np.mean([r["gt_accuracy"] for r in records])),
        "trained_accuracy_mean": float(np.mean([r["trained_accuracy"] for r in records])),
        "trained_accuracy_sd_sample": float(
            np.std([r["trained_accuracy"] for r in records], ddof=1)
        ),
        "trained_macro_f1_mean": float(np.mean([r["trained_macro_f1"] for r in records])),
        "paired_gain_pp": float(diff.mean() * 100),
        "paired_video_bootstrap_95_pp": interval,
        "validation_prediction_reload": "all 759 predictions match on CPU",
    }
    (e.OUT / "verified_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
