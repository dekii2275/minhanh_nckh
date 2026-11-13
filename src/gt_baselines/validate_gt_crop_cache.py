"""Verify complete crop-pose cache and write a provenance manifest."""

import json

import compare_person_selection as base
import gt_crop_pose_experiment as e
import numpy as np


def main():
    info = json.loads((e.OUT / "provenance.json").read_text(encoding="utf-8"))
    if info["script_sha256"] != base.sha(e.Path(e.__file__)) or info["manifest_sha256"] != base.sha(
        base.MANIFEST
    ):
        raise ValueError("Source script or split manifest changed")
    if info["model_sha256"] != base.sha(base.MODEL):
        raise ValueError("YOLO model changed")
    entries = []
    for r in e.rows():
        path = e.OUT / "poses" / f"{r['id']}.npz"
        with np.load(path) as d:
            if (
                len(d["xy"]) != int(r["nframes"])
                or str(d["annotation_sha256"]) != r["label_sha256"]
            ):
                raise ValueError("Cache mismatch " + r["id"])
            if d["xy"].shape != (int(r["nframes"]), 12, 2) or d["status"].shape != (
                int(r["nframes"]),
            ):
                raise ValueError("Bad pose shape " + r["id"])
            if not np.isin(d["status"], [0, 1, 2, 3]).all():
                raise ValueError("Bad status " + r["id"])
            entries.append(
                {
                    "id": r["id"],
                    "split": r["seed0_split"],
                    "frames": int(r["nframes"]),
                    "annotation_sha256": r["label_sha256"],
                    "image_bytes_sha256": str(d["image_bytes_sha256"]),
                    "pose_cache_sha256": base.sha(path),
                }
            )
    (e.OUT / "pose_manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Validated {len(entries)} videos, {sum(x['frames'] for x in entries)} frames")


if __name__ == "__main__":
    main()
