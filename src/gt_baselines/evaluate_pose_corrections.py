"""Exploratory 5-frame smoothing and short-gap imputation on YOLO validation.

All postprocessors use predicted coordinates/confidence only, with future frames
for centered operations. The saved per-frame target selection uses GT bbox,
therefore these are diagnostic results, not autonomous end-to-end evaluation.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import compare_person_selection as base
import numpy as np
import scipy.io as sio
import temporal_swap_analysis as source_loader
import torch

ROOT = base.ROOT
SWAP = np.arange(12) ^ 1
THRESHOLD = 0.30
MAX_GAP = 3
RADIUS = 2
MIN_NEIGHBORS = 3
METHODS = ("raw_identity", "fixed_swap", "interpolate3", "mean5", "median5", "interpolate3_mean5")


def valid_joints(xy, conf, threshold=THRESHOLD):
    return (
        np.isfinite(xy).all(axis=2) & (xy > 0).all(axis=2) & np.isfinite(conf) & (conf >= threshold)
    )


def interpolate_short(xy, conf, max_gap=MAX_GAP):
    """Interpolate internal runs of up to max_gap invalid samples per joint.

    Unfilled coordinates, including low-confidence predictions, remain as in
    the input. No endpoint extrapolation or GT-derived visibility is used.
    """
    result = xy.copy()
    outconf = conf.copy()
    valid = valid_joints(xy, conf)
    filled = np.zeros(valid.shape, bool)
    T, J = valid.shape
    for j in range(J):
        t = 0
        while t < T:
            if valid[t, j]:
                t += 1
                continue
            end = t
            while end < T and not valid[end, j]:
                end += 1
            if t > 0 and end < T and end - t <= max_gap:
                for s in range(t, end):
                    a = (s - t + 1) / (end - t + 1)
                    result[s, j] = (1 - a) * xy[t - 1, j] + a * xy[end, j]
                    outconf[s, j] = min(conf[t - 1, j], conf[end, j])
                    filled[s, j] = True
            t = end
    return result, outconf, filled


def smooth5(xy, conf, kind):
    """Centered mean/median of reliable observations; never fill missing target."""
    if kind not in ("mean", "median"):
        raise ValueError(kind)
    valid = valid_joints(xy, conf)
    result = xy.copy()
    T, J = valid.shape
    for t in range(T):
        lo = max(0, t - RADIUS)
        hi = min(T, t + RADIUS + 1)
        for j in range(J):
            if not valid[t, j]:
                continue
            points = xy[lo:hi, j][valid[lo:hi, j]]
            if len(points) >= MIN_NEIGHBORS:
                result[t, j] = (
                    np.mean(points, axis=0) if kind == "mean" else np.median(points, axis=0)
                )
    return result


def displacement_error(pose, gt, diag):
    """GT may be uint16; cast before differencing to avoid unsigned wrap."""
    pose = np.asarray(pose, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    delta = (pose[1:] - pose[:-1]) - (gt[1:] - gt[:-1])
    return np.linalg.norm(delta, axis=2) / np.maximum(
        np.asarray(diag[1:, None], dtype=np.float64), 1e-8
    )


def make_variants(xy, conf):
    """All processed variants use the same fixed-swap diagnostic baseline."""
    sx = xy[:, SWAP].copy()
    sc = conf[:, SWAP].copy()
    interp, ic, filled = interpolate_short(sx, sc)
    return {
        "raw_identity": (xy.copy(), conf.copy()),
        "fixed_swap": (sx, sc),
        "interpolate3": (interp, ic),
        "mean5": (smooth5(sx, sc, "mean"), sc.copy()),
        "median5": (smooth5(sx, sc, "median"), sc.copy()),
        "interpolate3_mean5": (smooth5(interp, ic, "mean"), ic.copy()),
    }, filled


def writecsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(out):
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    out.mkdir(parents=True)
    (out / "per_video").mkdir()
    source = json.loads((source_loader.SOURCE / "provenance.json").read_text(encoding="utf-8"))
    if source["manifest_sha256"] != base.sha(base.MANIFEST) or source["model_sha256"] != base.sha(
        base.MODEL
    ):
        raise ValueError("Source changed")
    for seed in (0, 1, 2):
        if source["checkpoints"][str(seed)] != base.sha(base.CKPTS / f"seed{seed}" / "best.pt"):
            raise ValueError("Checkpoint changed")
    frame_rows = {}
    with (source_loader.SOURCE / "frame_matches_iou_0_3.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        for r in csv.DictReader(f):
            frame_rows.setdefault(r["id"], []).append(r)
    features = {m: [] for m in METHODS}
    ids = []
    labels = []
    video_rows = []
    total_filled = 0
    totals = {
        m: {
            "n": 0,
            "error_px": 0.0,
            "error_norm": 0.0,
            "pck": 0,
            "velocity_n": 0,
            "velocity_norm": 0.0,
            "all_gt": 0,
            "all_pred": 0,
            "all_pck": 0,
        }
        for m in METHODS
    }
    manifest = []
    for index, row in enumerate(base.rows(), 1):
        sid = row["id"]
        xy, conf, boxes, d = source_loader.load_inputs(row, frame_rows[sid])
        # Confirm the confidence array used by the filter matches original detector output.
        for t, r in enumerate(frame_rows[sid]):
            i = int(r["gt_index"])
            lo, hi = d["offsets"][t : t + 2]
            if i >= 0 and not np.array_equal(conf[t], d["conf"][lo + i, 5:17]):
                raise ValueError("Confidence mismatch " + sid)
        start = time.perf_counter()
        variants, filled = make_variants(xy, conf)
        transform_time = time.perf_counter() - start
        total_filled += int(filled.sum())
        label = base.LABELS / f"{sid}.mat"
        if base.sha(label) != row["label_sha256"]:
            raise ValueError("GT changed " + sid)
        mat = sio.loadmat(label)
        gt = np.stack((mat["x"][:, 1:13], mat["y"][:, 1:13]), axis=-1).astype(np.float64)
        vis = mat["visibility"][:, 1:13] > 0
        wh = d["gt_bbox"][:, 2:] - d["gt_bbox"][:, :2]
        diag = np.linalg.norm(wh, axis=1)
        gt_valid = (
            vis
            & np.isfinite(gt).all(2)
            & (gt > 0).all(2)
            & (np.isfinite(wh).all(1) & (wh[:, 0] > 0) & (wh[:, 1] > 0))[:, None]
        )
        original_valid = np.isfinite(xy).all(2) & (xy > 0).all(2) & np.isfinite(conf) & (conf > 0)
        maxxy = np.full_like(xy, np.nan)
        maxconf = np.zeros_like(conf)
        for t, r in enumerate(frame_rows[sid]):
            j = int(r["max_index"])
            lo, hi = d["offsets"][t : t + 2]
            if j >= 0:
                maxxy[t] = d["xy"][lo + j, 5:17]
                maxconf[t] = d["conf"][lo + j, 5:17]
        max_valid = (
            np.isfinite(maxxy).all(2) & (maxxy > 0).all(2) & np.isfinite(maxconf) & (maxconf > 0)
        )
        common = gt_valid & original_valid & original_valid[:, SWAP] & max_valid
        # Every variant preserves valid baseline joints; enforce fixed denominators.
        for name, (pose, confidence) in variants.items():
            if not np.all(np.isfinite(pose[common])) or not np.all((pose[common] > 0).all(axis=1)):
                raise ValueError("Postprocessing invalidated common set: " + sid + " " + name)
        vel_common = common[1:] & common[:-1]
        vrow = {
            "id": sid,
            "action": row["action"],
            "frames": len(xy),
            "paired_joints": int(common.sum()),
            "filled_joint_frames": int(filled.sum()),
            "total_variant_compute_seconds": transform_time,
        }
        saved = {"filled": filled}
        for name, (pose, confidence) in variants.items():
            gt_all_pred = (
                gt_valid
                & np.isfinite(pose).all(2)
                & (pose > 0).all(2)
                & np.isfinite(confidence)
                & (confidence > 0)
            )
            dist = np.linalg.norm(pose - gt, axis=2)
            norm = dist / np.maximum(diag[:, None], 1e-8)
            pcount = int(common.sum())
            velocity = displacement_error(pose, gt, diag)
            a = totals[name]
            a["n"] += pcount
            a["error_px"] += float(dist[common].sum())
            a["error_norm"] += float(norm[common].sum())
            a["pck"] += int((common & (norm <= 0.2)).sum())
            a["velocity_n"] += int(vel_common.sum())
            a["velocity_norm"] += float(velocity[vel_common].sum())
            a["all_gt"] += int(gt_valid.sum())
            a["all_pred"] += int(gt_all_pred.sum())
            a["all_pck"] += int((gt_all_pred & (norm <= 0.2)).sum())
            vrow[name + "_mean_error_px"] = float(dist[common].mean()) if pcount else None
            vrow[name + "_mean_norm_error"] = float(norm[common].mean()) if pcount else None
            vrow[name + "_pck20_pct"] = (
                100 * float((norm[common] <= 0.2).mean()) if pcount else None
            )
            vrow[name + "_velocity_norm_error"] = (
                float(velocity[vel_common].mean()) if vel_common.any() else None
            )
            features[name].append(base.resample(pose))
            saved[name] = pose
            saved[name + "_conf"] = confidence
        np.savez_compressed(out / "per_video" / f"{sid}.npz", **saved)
        manifest.append(
            {
                "id": sid,
                "annotation_sha256": row["label_sha256"],
                "source_selection_npz_sha256": base.sha(
                    source_loader.SOURCE / "detections" / f"{sid}.npz"
                ),
                "source_pose_npz_sha256": base.sha(source_loader.INPUT / f"{sid}.npz"),
            }
        )
        video_rows.append(vrow)
        labels.append(base.ACTION_CLASSES.index(row["action"]))
        ids.append(sid)
        if index % 50 == 0:
            print(f"{index}/253 videos", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = {m: {} for m in METHODS}
    prediction_rows = []
    for name in METHODS:
        X = torch.tensor(np.stack(features[name]), device=device)
        for seed in (0, 1, 2):
            net = base.ActionBiLSTMAttentionClassifier().to(device)
            net.load_state_dict(
                torch.load(
                    base.CKPTS / f"seed{seed}" / "best.pt", map_location=device, weights_only=True
                )
            )
            net.eval()
            pred = []
            with torch.inference_mode():
                for t in range(0, len(X), 32):
                    pred.extend(net(X[t : t + 32]).argmax(-1).cpu().tolist())
            preds[name][seed] = pred
            prediction_rows.extend(
                {"id": sid, "seed": seed, "method": name, "target": target, "predicted": p}
                for sid, target, p in zip(ids, labels, pred)
            )
    result = {}
    rng = np.random.default_rng(20261103)
    Y = np.asarray(labels)
    boot_indices = rng.integers(0, len(Y), size=(10000, len(Y)))
    for name in METHODS:
        a = totals[name]
        seed_metrics = []
        per_video_diffs = []
        for seed in (0, 1, 2):
            P = np.asarray(preds[name][seed])
            B = np.asarray(preds["fixed_swap"][seed])
            diff = (P == Y).astype(int) - (B == Y).astype(int)
            mets = base.metric(Y, P)
            seed_metrics.append(
                {
                    "seed": seed,
                    **mets,
                    "improved_vs_fixed": int((diff > 0).sum()),
                    "worsened_vs_fixed": int((diff < 0).sum()),
                }
            )
            per_video_diffs.append(diff)
        mean_diff = np.mean(per_video_diffs, axis=0)
        boot = np.mean(mean_diff[boot_indices], axis=1) * 100
        pose_diff = [
            float(v[name + "_mean_error_px"]) - float(v["fixed_swap_mean_error_px"])
            for v in video_rows
            if v[name + "_mean_error_px"] is not None
        ]
        result[name] = {
            "accuracy_mean_pct": float(np.mean([x["accuracy_pct"] for x in seed_metrics])),
            "macro_f1_mean_pct": float(np.mean([x["macro_f1_pct"] for x in seed_metrics])),
            "accuracy_difference_vs_fixed_pp": 100 * float(mean_diff.mean()),
            "accuracy_paired_video_bootstrap_95_pct_pp": np.quantile(boot, [0.025, 0.975]).tolist(),
            "seeds": seed_metrics,
            "paired_gt_joints": a["n"],
            "mean_error_px": a["error_px"] / a["n"],
            "mean_error_norm_bbox": a["error_norm"] / a["n"],
            "pck20_common_pct": 100 * a["pck"] / a["n"],
            "velocity_common_joint_pairs": a["velocity_n"],
            "mean_velocity_error_norm_per_frame": a["velocity_norm"] / a["velocity_n"],
            "all_gt_visible_joints": a["all_gt"],
            "all_predicted_joints": a["all_pred"],
            "coverage_all_gt_pct": 100 * a["all_pred"] / a["all_gt"],
            "pck20_missing_incorrect_pct": 100 * a["all_pck"] / a["all_gt"],
            "video_pose_error_better_than_fixed": sum(x < -1e-7 for x in pose_diff),
            "video_pose_error_worse_than_fixed": sum(x > 1e-7 for x in pose_diff),
        }
    summary = {
        "videos": len(ids),
        "frames": sum(v["frames"] for v in video_rows),
        "filled_joint_frames": total_filled,
        "methods": result,
        "definitions": "All GT-selected YOLO poses; GT-bbox supported selection IoU>=0.3. Raw identity and fixed swap are controls. Mean/median5 only change reliable target joints (xy finite positive, conf>=0.30) with >=3 reliable observations in centered +/-2 frame window. Linear interpolation fills confidence<0.30 or invalid coordinates only for bounded gaps of 1-3 frames; no extrapolation, unresolved values unchanged. Combo interpolates then mean-filters. All use 12 joints, same classifier/normalization/resampling; no GT in postprocessing.",
        "pose_comparison": "Common original 160594 visible GT joints valid for both orientations and confidence-selected detection; pixel Euclidean error; PCK20 threshold 0.2 GT bbox diagonal. Velocity is difference in 2D pixel displacement per frame vs GT, divided by GT bbox diagonal; no FPS assumption.",
        "caution": "Exploratory validation; GT assistance at target selection, no tracking; symmetric mapping uncertainty; centered filter uses future frames; classifier trained on GT not predicted pose. GT test not run. Motion errors do not prove action quality.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "script_sha256": base.sha(Path(__file__)),
                "selection_sha256": base.sha(source_loader.SOURCE / "frame_matches_iou_0_3.csv"),
                "source_provenance_sha256": base.sha(source_loader.SOURCE / "provenance.json"),
                "manifest_sha256": base.sha(base.MANIFEST),
                "yolo_model_sha256": source["model_sha256"],
                "classifier_sha256": source["checkpoints"],
                "torch": torch.__version__,
                "device": device,
                "versions": "see source provenance",
                "postprocess": "no GT; centered windows; confidence controls filters only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    writecsv(out / "per_video_metrics.csv", video_rows)
    writecsv(out / "predictions.csv", prediction_rows)
    writecsv(out / "input_manifest.csv", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(Path(args.output))
