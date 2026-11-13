"""Reload each seed-0 checkpoint and check saved validation predictions."""

import json

import torch
import train_gt_baselines as base
from torch.utils.data import DataLoader
from train_gt_cascade_scratch import CascadeScratch

torch.set_num_threads(4)
data = base.load_data()
loader = DataLoader(data["validation"], batch_size=32, shuffle=False)
root = base.ROOT / "outputs/gt_baselines_20261102_v1"
results = {}
for name in ["lstm", "bilstm", "gru", "bilstm_attention", "cascade_scratch"]:
    folder = root / name / "seed0"
    model = CascadeScratch() if name == "cascade_scratch" else base.model_for(name)
    model.load_state_dict(
        torch.load(folder / "best.pt", map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    actual = {}
    with torch.inference_mode():
        for x, y, ids in loader:
            for sid, p in zip(ids, model(x).softmax(-1).tolist()):
                actual[sid] = p
    saved = json.loads((folder / "best_validation.json").read_text(encoding="utf-8"))["predictions"]
    max_diff = max(
        abs(a - b) for row in saved for a, b in zip(actual[row["id"]], row["probabilities"])
    )
    mismatches = sum(
        int(max(range(15), key=lambda k: actual[row["id"]][k]) != row["predicted"]) for row in saved
    )
    # Different CPU/GPU kernels may differ slightly in floating point.
    assert mismatches == 0 and max_diff < 0.02, (name, mismatches, max_diff)
    results[name] = {
        "n": len(saved),
        "prediction_mismatches": mismatches,
        "max_probability_difference_cpu_vs_gpu": max_diff,
    }
    print(name, results[name], flush=True)
(root / "checkpoint_reload_verification.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
