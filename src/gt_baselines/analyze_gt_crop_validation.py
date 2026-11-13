"""Evaluate completed GT-crop validation cache on fixed GT-trained checkpoints."""

import json

import compare_person_selection as base
import gt_crop_pose_experiment as experiment
import numpy as np
import torch
import train_gt_baselines as trainer


def main():
    torch.set_num_threads(4)
    rows = [r for r in experiment.rows() if r["seed0_split"] == "validation"]
    features = {"identity": [], "fixed_swap": []}
    coverage = np.zeros(4, int)
    target = []
    for r in rows:
        d = np.load(experiment.OUT / "poses" / f"{r['id']}.npz")
        if str(d["annotation_sha256"]) != r["label_sha256"] or len(d["xy"]) != int(r["nframes"]):
            raise ValueError(r["id"])
        coverage += np.bincount(d["status"], minlength=4)
        target.append(base.ACTION_CLASSES.index(r["action"]))
        features["identity"].append(base.resample(d["xy"]))
        features["fixed_swap"].append(base.resample(d["xy"][:, experiment.SWAP]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for method in features:
        x = torch.tensor(np.stack(features[method]), device=device)
        for seed in range(3):
            net = trainer.model_for("bilstm_attention").to(device)
            path = base.CKPTS / f"seed{seed}" / "best.pt"
            net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            net.eval()
            with torch.inference_mode():
                pred = torch.cat([net(batch).argmax(-1) for batch in x.split(32)]).cpu().tolist()
            results.append(
                {
                    "variant": method,
                    "seed": seed,
                    **base.metric(target, pred),
                    "checkpoint_sha256": base.sha(path),
                    "predictions": [
                        {"id": r["id"], "target": y, "predicted": p}
                        for r, y, p in zip(rows, target, pred)
                    ],
                }
            )
    out = {
        "videos": len(rows),
        "frames": int(coverage.sum()),
        "coverage": dict(
            zip(["invalid_bbox", "no_detection", "no_match", "matched"], map(int, coverage))
        ),
        "results": results,
    }
    (experiment.OUT / "validation_gt_trained_preview.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                if k != "results"
                else [{x: y for x, y in r.items() if x != "predictions"} for r in v]
                for k, v in out.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
