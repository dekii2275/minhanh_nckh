"""Independently reload locked checkpoints on CPU and verify saved test labels."""

import json

import confirm_confidence_penn as c
import numpy as np
import torch

torch.set_num_threads(4)
saved = json.loads((c.OUT / "test_predictions.json").read_text(encoding="utf8"))
xs = {m: [] for m in c.METHODS}
ids = []
ys = []
for row in c.test_rows():
    with np.load(c.OUT / "test_pose/poses" / f"{row['id']}.npz") as a:
        xy, conf = a["xy"][:, c.pose.SWAP], a["conf"][:, c.pose.SWAP]
        variants, _ = c.feat.make_variants(c.pose.base.resample(xy), xy, conf)
        for m in c.METHODS:
            xs[m].append(variants[m])
    ids.append(row["id"])
    ys.append(c.pose.base.ACTION_CLASSES.index(row["action"]))
results = []
for m in c.METHODS:
    x = torch.from_numpy(np.stack(xs[m]))
    for seed in c.SEEDS:
        net = c.exp.model_for(m)
        net.load_state_dict(
            torch.load(
                c.OUT / "trained" / m / f"seed{seed}" / "best.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        net.eval()
        with torch.inference_mode():
            pred = torch.cat([net(b).argmax(-1) for b in x.split(32)]).tolist()
        expected = next(r["predictions"] for r in saved if r["method"] == m and r["seed"] == seed)
        assert [(i, int(y)) for i, y in zip(ids, ys)] == [(r["id"], r["target"]) for r in expected]
        mismatches = [
            {"id": i, "target": int(y), "cpu": p, "gpu": r["predicted"]}
            for i, y, p, r in zip(ids, ys, pred, expected)
            if p != r["predicted"]
        ]
        results.append(
            {
                "method": m,
                "seed": seed,
                "matched": len(pred) - len(mismatches),
                "mismatches": mismatches,
            }
        )
        print(m, seed, len(pred) - len(mismatches), "matched", mismatches, flush=True)
c.write(
    c.OUT / "cpu_verification.json",
    {
        "runs": results,
        "predictions_matched": sum(r["matched"] for r in results),
        "script_sha256": c.pose.base.sha(__import__("pathlib").Path(__file__)),
    },
)
