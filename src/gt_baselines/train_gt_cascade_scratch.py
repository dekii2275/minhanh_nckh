"""Train authors' CascadeFormer 1.0 architecture from scratch on common 12-joint GT.

This omits their masked pretraining and differs in input and preprocessing; it
is an architecture adaptation, not a reproduction of the published method.
"""

import ast
import json
from pathlib import Path

import torch
import train_gt_baselines as base
from torch import nn

SOURCE = (
    base.ROOT
    / "third_party/CascadeFormer/baseline/action_recognition/cascadeformer_1_0/joint/penn_action"
)


def load_classes(filename, names, namespace):
    path = SOURCE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {c.name for c in classes} != set(names):
        raise RuntimeError("Missing class")
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)


namespace = {"torch": torch, "nn": nn, "POSITIONAL_UPPER_BOUND": 1000}
load_classes("pretraining.py", ["BaseT1"], namespace)
load_classes("finetuning.py", ["BaseT2", "CrossAttention", "GaitRecognitionHead"], namespace)


class CascadeScratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = namespace["BaseT1"](
            num_joints=12, three_d=False, d_model=256, nhead=8, num_layers=4
        )
        self.t2 = namespace["BaseT2"](d_model=256, nhead=8, num_layers=4)
        self.cross = namespace["CrossAttention"](d_model=256)
        self.head = namespace["GaitRecognitionHead"](input_dim=256, num_classes=15)

    def forward(self, x):
        a = self.t1.encode(x)
        b = self.t2.encode(a)
        return self.head(self.cross(a, b, b).mean(dim=1))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = base.load_data()
    base.model_for = lambda _: CascadeScratch()
    for seed in args.seeds:
        base.run("cascade_scratch", seed, data, args, out)
        path = out / "cascade_scratch" / ("seed" + str(seed)) / "provenance.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(
            {
                "wrapper_sha256": base.sha(Path(__file__)),
                "author_pretraining_py_sha256": base.sha(SOURCE / "pretraining.py"),
                "author_finetuning_py_sha256": base.sha(SOURCE / "finetuning.py"),
                "pretrained": False,
                "deviation": "12 joints, normalized 64-frame GT; no masked pretraining; same optimizer and selection as recurrent baselines",
            }
        )
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
