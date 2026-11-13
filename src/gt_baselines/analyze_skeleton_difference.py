"""Compare saved YOLO skeletons with Penn GT on seed-0 validation.

Diagnostic only. Selection by GT bbox and fixed L/R swap both use GT or are
counterfactuals, not autonomous deployable estimators.
"""

import argparse
import csv
import json
from pathlib import Path

import compare_person_selection as base
import numpy as np
import scipy.io as sio

SOURCE = base.ROOT / "outputs/person_selection_validation_20261102"
JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
SWAP = np.arange(12) ^ 1
METHODS = ("confidence", "gt_assisted", "gt_assisted_swap")


def nanmean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def savecsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(out):
    if out.exists():
        raise FileExistsError("Choose a new output folder")
    out.mkdir(parents=True)
    source = json.loads((SOURCE / "provenance.json").read_text(encoding="utf-8"))
    if source["manifest_sha256"] != base.sha(base.MANIFEST) or source["model_sha256"] != base.sha(
        base.MODEL
    ):
        raise ValueError("Manifest or model hash differs from saved inference")
    frame_rows = {}
    with (SOURCE / "frame_matches_iou_0_3.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame_rows.setdefault(row["id"], []).append(row)
    videos = []
    joints = []
    global_counts = {
        m: {"gt": 0, "valid": 0, "ok": 0, "distance": 0.0, "normalized": 0.0} for m in METHODS
    }
    global_joints = {
        m: [{"gt": 0, "valid": 0, "ok": 0, "distance": 0.0, "normalized": 0.0} for _ in JOINTS]
        for m in METHODS
    }
    paired_global = {m: {"count": 0, "ok": 0, "distance": 0.0, "normalized": 0.0} for m in METHODS}
    for number, row in enumerate(base.rows(), 1):
        sid = row["id"]
        label = base.LABELS / f"{sid}.mat"
        if base.sha(label) != row["label_sha256"]:
            raise ValueError("Annotation hash mismatch " + sid)
        mat = sio.loadmat(label)
        T = int(row["nframes"])
        d = np.load(SOURCE / "detections" / f"{sid}.npz")
        if str(d["annotation_sha256"]) != row["label_sha256"]:
            raise ValueError("Cache annotation mismatch " + sid)
        gt = np.stack((mat["x"][:, 1:13], mat["y"][:, 1:13]), axis=-1).astype(np.float64)
        vis = mat["visibility"][:, 1:13]
        bbox = d["gt_bbox"].astype(np.float64)
        wh = bbox[:, 2:] - bbox[:, :2]
        diagonal = np.linalg.norm(wh, axis=1)
        box_ok = np.isfinite(bbox).all(1) & (wh[:, 0] > 0) & (wh[:, 1] > 0)
        gt_ok = (
            (vis > 0)
            & np.isfinite(gt).all(2)
            & (gt[:, :, 0] > 0)
            & (gt[:, :, 1] > 0)
            & box_ok[:, None]
        )
        if len(frame_rows[sid]) != T or len(d["offsets"]) != T + 1 or gt.shape != (T, 12, 2):
            raise ValueError("Frame count mismatch " + sid)
        pred = {m: np.full((T, 12, 2), np.nan, np.float32) for m in METHODS}
        pconf = {m: np.zeros((T, 12), np.float32) for m in METHODS}
        for t, r in enumerate(frame_rows[sid]):
            if int(r["frame"]) != t + 1:
                raise ValueError("Frame order mismatch " + sid)
            lo, hi = d["offsets"][t : t + 2]
            for method, key in (("confidence", "max_index"), ("gt_assisted", "gt_index")):
                idx = int(r[key])
                n = int(hi - lo)
                if not -1 <= idx < n:
                    raise ValueError("Invalid detection index")
                if idx >= 0:
                    pred[method][t] = d["xy"][lo + idx, 5:17]
                    pconf[method][t] = d["conf"][lo + idx, 5:17]
            pred["gt_assisted_swap"][t] = pred["gt_assisted"][t, SWAP]
            pconf["gt_assisted_swap"][t] = pconf["gt_assisted"][t, SWAP]
        valid = {}
        dist = {}
        nd = {}
        for m in METHODS:
            v = pred[m]
            valid[m] = (
                gt_ok
                & np.isfinite(v).all(2)
                & (v[:, :, 0] > 0)
                & (v[:, :, 1] > 0)
                & np.isfinite(pconf[m])
                & (pconf[m] > 0)
            )
            dist[m] = np.linalg.norm(v - gt, axis=2)
            nd[m] = dist[m] / np.maximum(diagonal[:, None], 1e-8)
        common = valid["confidence"] & valid["gt_assisted"] & valid["gt_assisted_swap"]
        v = {
            "id": sid,
            "action": row["action"],
            "frames": T,
            "gt_visible_joints": int(gt_ok.sum()),
            "paired_joints": int(common.sum()),
        }
        for m in METHODS:
            den = int(gt_ok.sum())
            n = int(valid[m].sum())
            hits = int((valid[m] & (nd[m] <= 0.2)).sum())
            pixel_sum = float(dist[m][valid[m]].sum())
            norm_sum = float(nd[m][valid[m]].sum())
            c = global_counts[m]
            c["gt"] += den
            c["valid"] += n
            c["ok"] += hits
            c["distance"] += pixel_sum
            c["normalized"] += norm_sum
            v[m + "_valid_joints"] = n
            v[m + "_mean_error_px"] = pixel_sum / n if n else None
            v[m + "_mean_norm_error"] = norm_sum / n if n else None
            v[m + "_pck20_all_pct"] = 100 * hits / den if den else None
            v[m + "_coverage_pct"] = 100 * n / den if den else None
            cn = int(common.sum())
            ch = int((common & (nd[m] <= 0.2)).sum())
            cp = float(dist[m][common].sum())
            cr = float(nd[m][common].sum())
            x = paired_global[m]
            x["count"] += cn
            x["ok"] += ch
            x["distance"] += cp
            x["normalized"] += cr
            v[m + "_paired_error_px"] = cp / cn if cn else None
            v[m + "_paired_norm_error"] = cr / cn if cn else None
            v[m + "_paired_pck20_pct"] = 100 * ch / cn if cn else None
            for j, name in enumerate(JOINTS):
                gj = gt_ok[:, j]
                ok = valid[m][:, j]
                count = int(ok.sum())
                obj = global_joints[m][j]
                obj["gt"] += int(gj.sum())
                obj["valid"] += count
                obj["ok"] += int((ok & (nd[m][:, j] <= 0.2)).sum())
                obj["distance"] += float(dist[m][:, j][ok].sum())
                obj["normalized"] += float(nd[m][:, j][ok].sum())
        videos.append(v)
        if number % 50 == 0:
            print(f"{number}/253 videos", flush=True)
    summary = {}
    for m in METHODS:
        c = global_counts[m]
        pair = paired_global[m]
        v_errors = np.asarray(
            [v[m + "_mean_error_px"] for v in videos if v[m + "_mean_error_px"] is not None]
        )
        v_nerrors = np.asarray(
            [v[m + "_mean_norm_error"] for v in videos if v[m + "_mean_norm_error"] is not None]
        )
        summary[m] = {
            "visible_gt_joints": c["gt"],
            "valid_predicted_joints": c["valid"],
            "coverage_pct": 100 * c["valid"] / c["gt"],
            "mean_error_px_among_predicted": c["distance"] / c["valid"],
            "mean_error_norm_bbox_diagonal_among_predicted": c["normalized"] / c["valid"],
            "pck20_missing_incorrect_pct": 100 * c["ok"] / c["gt"],
            "pck20_among_predicted_pct": 100 * c["ok"] / c["valid"],
            "video_mean_error_px_median": float(np.median(v_errors)),
            "video_mean_error_px_p90": float(np.percentile(v_errors, 90)),
            "video_equal_mean_norm_error": float(v_nerrors.mean()),
            "paired_common_joints": pair["count"],
            "paired_error_px": pair["distance"] / pair["count"],
            "paired_norm_error": pair["normalized"] / pair["count"],
            "paired_pck20_pct": 100 * pair["ok"] / pair["count"],
        }
        for name, cj in zip(JOINTS, global_joints[m]):
            n = cj["valid"]
            joints.append(
                {
                    "method": m,
                    "joint": name,
                    "gt_visible_joints": cj["gt"],
                    "valid_predicted_joints": n,
                    "coverage_pct": 100 * n / cj["gt"] if cj["gt"] else None,
                    "mean_error_px": cj["distance"] / n if n else None,
                    "mean_norm_error": cj["normalized"] / n if n else None,
                    "pck20_missing_incorrect_pct": 100 * cj["ok"] / cj["gt"] if cj["gt"] else None,
                }
            )
    by_action = []
    for action in base.ACTION_CLASSES:
        subset = [v for v in videos if v["action"] == action]
        for m in METHODS:
            by_action.append(
                {
                    "action": action,
                    "method": m,
                    "videos": len(subset),
                    "mean_video_error_px": nanmean([x[m + "_mean_error_px"] for x in subset]),
                    "mean_video_norm_error": nanmean([x[m + "_mean_norm_error"] for x in subset]),
                    "mean_video_pck20_all_pct": nanmean([x[m + "_pck20_all_pct"] for x in subset]),
                }
            )
    rng = np.random.default_rng(0)
    comparisons = []
    for a, b in [("confidence", "gt_assisted"), ("gt_assisted", "gt_assisted_swap")]:
        subset = [
            v
            for v in videos
            if v[a + "_paired_norm_error"] is not None and v[b + "_paired_norm_error"] is not None
        ]
        delta = np.asarray(
            [v[b + "_paired_norm_error"] - v[a + "_paired_norm_error"] for v in subset]
        )
        boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        pdelta = np.asarray(
            [v[b + "_paired_pck20_pct"] - v[a + "_paired_pck20_pct"] for v in subset]
        )
        pboot = pdelta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        comparisons.append(
            {
                "from": a,
                "to": b,
                "videos": len(delta),
                "video_equal_mean_norm_error_difference": float(delta.mean()),
                "bootstrap_95_ci_norm_error": np.quantile(boot, [0.025, 0.975]).tolist(),
                "video_equal_mean_pck20_difference_pp": float(pdelta.mean()),
                "bootstrap_95_ci_pck20_pp": np.quantile(pboot, [0.025, 0.975]).tolist(),
            }
        )
    result = {
        "videos": len(videos),
        "frames": sum(v["frames"] for v in videos),
        "methods": summary,
        "paired_video_comparisons": comparisons,
        "definition": "GT visible 12 joints, finite x/y >0, valid bbox; YOLO coordinates finite and positive with keypoint confidence >0; PCK20 uses 0.2 * GT bbox diagonal; missing counted incorrect for all-joint PCK; pixel coordinates directly compared without 1px shift.",
        "paired_definition": "identical GT joints where all three methods have valid predicted coordinates.",
        "caution": "GT-assisted bbox selection and fixed swap are diagnostic; neither establishes target identity or true left/right convention. Validation is exploratory.",
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "script_sha256": base.sha(Path(__file__)),
                "source_provenance_sha256": base.sha(SOURCE / "provenance.json"),
                "source_frame_csv_sha256": base.sha(SOURCE / "frame_matches_iou_0_3.csv"),
                "manifest_sha256": base.sha(base.MANIFEST),
                "source_model_sha256": source["model_sha256"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    savecsv(out / "per_video.csv", videos)
    savecsv(out / "per_joint.csv", joints)
    savecsv(out / "per_action.csv", by_action)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    analyze(Path(args.output))
