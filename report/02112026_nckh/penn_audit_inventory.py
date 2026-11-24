"""Read-only Penn inventory; writes only into this audit's output directory."""

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/Penn_Action"
OUT = Path(__file__).resolve().parent / "penn_audit_artifacts"
OUT.mkdir(exist_ok=True)
rows, issues = [], []


def issue(sid, kind, detail):
    issues.append(dict(id=sid, issue=kind, detail=str(detail)))


labels = {p.stem: p for p in (DATA / "labels").glob("*.mat")}
dirs = {p.name: p for p in (DATA / "frames").iterdir() if p.is_dir()}
for sid in sorted(set(labels) | set(dirs)):
    if sid not in labels or sid not in dirs:
        issue(
            sid, "missing_label_or_frame_directory", f"label={sid in labels}; frames={sid in dirs}"
        )
        continue
    try:
        m = loadmat(labels[sid])
        n = int(m["nframes"].item())
        action = str(m["action"].item())
        split = int(m["train"].item())
        dims = m["dimensions"].ravel().tolist()
        x, y, v = m["x"], m["y"], m["visibility"]
        for k, a in [("x", x), ("y", y), ("visibility", v)]:
            if a.shape != (n, 13):
                issue(sid, "annotation_shape", f"{k}: {a.shape}, expected {(n, 13)}")
        if m["bbox"].shape != (n, 4) or len(dims) != 3 or dims[2] != n:
            issue(
                sid,
                "annotation_metadata",
                f"bbox={m['bbox'].shape}, dimensions={dims}, nframes={n}",
            )
        if split not in (-1, 1):
            issue(sid, "invalid_split", split)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            issue(sid, "nonfinite_annotation", "x/y")
        if not set(np.unique(v)).issubset({0, 1}):
            issue(sid, "visibility_values", np.unique(v))
        files = list(dirs[sid].glob("*.jpg"))
        names = {p.name for p in files}
        expected = {f"{i:06}.jpg" for i in range(1, n + 1)}
        missing, extra = sorted(expected - names), sorted(names - expected)
        if missing or extra:
            issue(sid, "frame_names", f"missing={missing}; extra={extra}")
        # Decode first, middle and last frame; no claim about other frames.
        for i in sorted({1, (n + 1) // 2, n}):
            p = dirs[sid] / f"{i:06}.jpg"
            im = cv2.imread(str(p))
            if im is None:
                issue(sid, "unreadable_sampled_frame", p.name)
            elif tuple(im.shape[:2]) != tuple(dims[:2]):
                issue(sid, "sampled_frame_dimensions", f"{p.name}: {im.shape[:2]} != {dims[:2]}")
        rows.append(
            dict(
                id=sid,
                action=action,
                official_split="train" if split == 1 else "test" if split == -1 else str(split),
                nframes=n,
                frame_files=len(files),
                height=dims[0],
                width=dims[1],
                invisible=int((v == 0).sum()),
                total_joints=int(v.size),
                label_sha256=hashlib.sha256(labels[sid].read_bytes()).hexdigest(),
            )
        )
    except Exception as e:
        issue(sid, "annotation_or_io_error", repr(e))

byclass = defaultdict(list)
for r in rows:
    if r["official_split"] == "train":
        byclass[r["action"]].append(r)


def assign(seed):
    rng = random.Random(seed)
    result = {}
    # order must match sorted numeric ACTION_TO_ID, not alphabetical assumptions
    sys.path.insert(0, str(ROOT / "src"))
    from data.action_dataset import ACTION_CLASSES

    for action in ACTION_CLASSES:
        samples = sorted(byclass[action], key=lambda r: r["id"])
        rng.shuffle(samples)
        n = max(1, min(len(samples) - 1, round(len(samples) * 0.2)))
        result.update(
            {r["id"]: ("validation" if i < n else "train") for i, r in enumerate(samples)}
        )
    return result


for seed in (0, 42):
    assigned = assign(seed)
    for r in rows:
        r[f"seed{seed}_split"] = assigned.get(r["id"], "test")
old = (
    ROOT
    / "luu_tru_nghien_cuu_truoc/outputs/action_recognition/protocol_v2/multi_seed/baseline/seed0/action_recognition_summary.json"
)
oldmeta = json.loads(old.read_text(encoding="utf8"))["protocol"]["sequence_ids"]
comparison = {}
for key, field, value in [
    ("train_gt", "seed0_split", "train"),
    ("validation_gt", "seed0_split", "validation"),
    ("test_gt", "seed0_split", "test"),
]:
    a = {r["id"] for r in rows if r[field] == value}
    b = set(oldmeta[key])
    comparison[key] = {
        "current": len(a),
        "historical": len(b),
        "only_current": sorted(a - b),
        "only_historical": sorted(b - a),
    }
hashes = defaultdict(list)
for r in rows:
    hashes[r["label_sha256"]].append(r["id"])
dups = [ids for ids in hashes.values() if len(ids) > 1]
for ids in dups:
    issue(",".join(ids), "duplicate_label_bytes", "identical MAT SHA256")


def write(name, data, fields):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


write("video_manifest.csv", rows, list(rows[0]))
write("issues.csv", issues, ["id", "issue", "detail"])
stats = []
for action in sorted({r["action"] for r in rows}):
    rr = [r for r in rows if r["action"] == action]
    stats.append(
        dict(
            action=action,
            total=len(rr),
            official_train=sum(r["official_split"] == "train" for r in rr),
            official_test=sum(r["official_split"] == "test" for r in rr),
            train_seed0=sum(r["seed0_split"] == "train" for r in rr),
            validation_seed0=sum(r["seed0_split"] == "validation" for r in rr),
            frames=sum(r["nframes"] for r in rr),
            invisible=sum(r["invisible"] for r in rr),
            joints=sum(r["total_joints"] for r in rr),
        )
    )
write("class_statistics.csv", stats, list(stats[0]))
summary = {
    "labels": len(labels),
    "frame_dirs": len(dirs),
    "rows": len(rows),
    "nframes": sum(r["nframes"] for r in rows),
    "frame_files": sum(r["frame_files"] for r in rows),
    "invisible": sum(r["invisible"] for r in rows),
    "total_joints": sum(r["total_joints"] for r in rows),
    "issue_counts": dict(Counter(i["issue"] for i in issues)),
    "split_comparison_seed0": comparison,
    "duplicate_mat_hash_groups": dups,
    "decoding_scope": "first, middle, last JPEG per sequence only; no original video available",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
print(json.dumps(summary, indent=2))
