"""Decode and hash every JPEG, without modifying the dataset."""

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "penn_audit_artifacts"
FRAME = ROOT / "data/raw/Penn_Action/frames"
manifest = list(csv.DictReader((OUT / "video_manifest.csv").open(encoding="utf-8-sig")))


def check(row):
    sid = row["id"]
    h = hashlib.sha256()
    issues = []
    dims = set()
    decoded = 0
    for i in range(1, int(row["nframes"]) + 1):
        p = FRAME / sid / f"{i:06}.jpg"
        try:
            b = p.read_bytes()
            h.update(len(b).to_bytes(8, "big"))
            h.update(b)
            img = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                issues.append({"id": sid, "frame": i, "issue": "decode_failed"})
            else:
                decoded += 1
                dims.add((int(img.shape[0]), int(img.shape[1])))
        except Exception as e:
            issues.append({"id": sid, "frame": i, "issue": repr(e)})
    if len(dims) > 1:
        issues.append({"id": sid, "frame": "", "issue": f"varying_dimensions {sorted(dims)}"})
    return {
        "id": sid,
        "decoded": decoded,
        "sha256": h.hexdigest(),
        "dimensions": sorted(dims),
        "issues": issues,
    }


with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(check, manifest))
issues = [issue for r in results for issue in r["issues"]]
groups = defaultdict(list)
for r in results:
    groups[r["sha256"]].append(r["id"])
duplicates = [v for v in groups.values() if len(v) > 1]
with (OUT / "frame_decode_issues.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "frame", "issue"])
    w.writeheader()
    w.writerows(issues)
summary = {
    "videos": len(results),
    "decoded_frames": sum(r["decoded"] for r in results),
    "issue_count": len(issues),
    "duplicate_exact_jpeg_sequence_groups": duplicates,
    "method": "SHA256 over ordered JPEG bytes and lengths; not perceptual or cross-encoding duplicate detection",
}
(OUT / "full_frame_check.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
print(json.dumps(summary, indent=2))
