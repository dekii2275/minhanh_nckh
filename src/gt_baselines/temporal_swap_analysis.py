"""Exploratory four-neighbor correction of isolated left/right flips.

Input is the prior per-frame GT-bbox-assisted YOLO selection. GT is NOT read
by temporal_decisions(), but the pose source uses GT, so this is a diagnostic,
not an autonomous tracker or deployable end-to-end result.
"""

import argparse
import csv
import json
from pathlib import Path

import compare_person_selection as base
import numpy as np
import scipy.io as sio
import torch

ROOT = base.ROOT
SOURCE = ROOT / "outputs/person_selection_validation_20261102"
INPUT = ROOT / "outputs/missing_pose_validation_20261102/per_video"
ANALYSIS = ROOT / "outputs/skeleton_gt_comparison_validation_20261103"
SWAP = np.arange(12) ^ 1
JOINT_CONF = 0.30
MARGIN = 0.02
MIN_JOINTS = 4
MAX_CENTER_RATIO = 0.75


def compare(reference, candidate, conf_ref, conf_cand, box_ref, box_cand):
    """Compare same vs swapped candidate against reference without GT."""
    if not np.isfinite(box_ref).all() or not np.isfinite(box_cand).all():
        return None
    wh1 = box_ref[2:] - box_ref[:2]
    wh2 = box_cand[2:] - box_cand[:2]
    if np.any(wh1 <= 0) or np.any(wh2 <= 0):
        return None
    scale = float((np.linalg.norm(wh1) + np.linalg.norm(wh2)) / 2)
    if scale <= 0:
        return None
    center1 = (box_ref[:2] + box_ref[2:]) / 2
    center2 = (box_cand[:2] + box_cand[2:]) / 2
    if np.linalg.norm(center1 - center2) > MAX_CENTER_RATIO * scale:
        return None
    ok1 = (
        np.isfinite(reference).all(axis=1)
        & (reference > 0).all(axis=1)
        & np.isfinite(conf_ref)
        & (conf_ref >= JOINT_CONF)
    )
    ok2 = (
        np.isfinite(candidate).all(axis=1)
        & (candidate > 0).all(axis=1)
        & np.isfinite(conf_cand)
        & (conf_cand >= JOINT_CONF)
    )
    common = ok1 & ok2 & ok2[SWAP]
    n = int(common.sum())
    if n < MIN_JOINTS:
        return None
    keep = float(np.linalg.norm(reference[common] - candidate[common], axis=1).mean() / scale)
    flip = float(np.linalg.norm(reference[common] - candidate[SWAP][common], axis=1).mean() / scale)
    return keep, flip, n


def temporal_decisions(xy, conf, boxes):
    """Start with a global fixed swap, undo it on isolated opposite frames.

    The four neighbors must all exist and agree internally; 3 of 4 must prefer
    flipping the center by MARGIN, with at least one vote on either side.
    Decisions are simultaneous, using original neighboring observations.
    """
    T = len(xy)
    reverse = np.zeros(T, bool)
    reasons = []
    for t in range(T):
        rec = {
            "frame": t + 1,
            "reverse_fixed_swap": False,
            "reason": "boundary",
            "opposite_votes": 0,
        }
        if t < 2 or t + 2 >= T:
            reasons.append(rec)
            continue
        neighbor = [t - 2, t - 1, t + 1, t + 2]
        # Compare only adjacent neighbors + the cross-gap pair.
        checks = [
            compare(xy[a], xy[b], conf[a], conf[b], boxes[a], boxes[b])
            for a, b in ((t - 2, t - 1), (t - 1, t + 1), (t + 1, t + 2))
        ]
        if any(c is None for c in checks):
            rec["reason"] = "neighbor_missing_or_ambiguous"
            reasons.append(rec)
            continue
        if not all(keep + MARGIN < flip for keep, flip, _ in checks):
            rec["reason"] = "neighbors_not_consistent"
            reasons.append(rec)
            continue
        votes = [compare(xy[n], xy[t], conf[n], conf[t], boxes[n], boxes[t]) for n in neighbor]
        if any(c is None for c in votes):
            rec["reason"] = "center_missing_or_ambiguous"
            reasons.append(rec)
            continue
        opposite = [flip + MARGIN < keep for keep, flip, _ in votes]
        rec["opposite_votes"] = int(sum(opposite))
        if sum(opposite) >= 3 and any(opposite[:2]) and any(opposite[2:]):
            reverse[t] = True
            rec["reverse_fixed_swap"] = True
            rec["reason"] = "temporal_reverse"
        else:
            rec["reason"] = "keep_fixed_swap"
        reasons.append(rec)
    return reverse, reasons


def load_inputs(row, frame_rows):
    sid = row["id"]
    d = np.load(SOURCE / "detections" / f"{sid}.npz")
    z = np.load(INPUT / f"{sid}.npz")
    xy = z["baseline"]
    conf = z["baseline_conf"]
    T = int(row["nframes"])
    if len(xy) != T or len(frame_rows) != T or str(d["annotation_sha256"]) != row["label_sha256"]:
        raise ValueError("Input mismatch " + sid)
    boxes = np.full((T, 4), np.nan, np.float32)
    for t, r in enumerate(frame_rows):
        idx = int(r["gt_index"])
        lo, hi = d["offsets"][t : t + 2]
        if int(r["frame"]) != t + 1 or not -1 <= idx < int(hi - lo):
            raise ValueError("Frame index mismatch")
        if idx >= 0:
            boxes[t] = d["boxes"][lo + idx, :4]
            if not np.array_equal(xy[t], d["xy"][lo + idx, 5:17], equal_nan=True):
                raise ValueError("Pose mismatch")
    return xy, conf, boxes, d


def summarize(out):
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    out.mkdir(parents=True)
    (out / "per_video").mkdir()
    source = json.loads((SOURCE / "provenance.json").read_text(encoding="utf-8"))
    if source["manifest_sha256"] != base.sha(base.MANIFEST):
        raise ValueError("Manifest mismatch")
    for seed in (0, 1, 2):
        if base.sha(base.CKPTS / f"seed{seed}" / "best.pt") != source["checkpoints"][str(seed)]:
            raise ValueError("Checkpoint mismatch")
    frames = {}
    with (SOURCE / "frame_matches_iou_0_3.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            frames.setdefault(r["id"], []).append(r)
    with (ANALYSIS / "per_video.csv").open(newline="", encoding="utf-8") as f:
        prior = {r["id"]: r for r in csv.DictReader(f)}
    ids11 = {
        sid
        for sid, r in prior.items()
        if r["gt_assisted_paired_error_px"]
        and float(r["gt_assisted_swap_paired_error_px"]) > float(r["gt_assisted_paired_error_px"])
    }
    if len(ids11) != 11:
        raise ValueError("Previous exception set changed")
    features = {k: [] for k in ("identity", "fixed_swap", "temporal")}
    labels = []
    ids = []
    vrows = []
    frows = []
    accum = {
        group: {m: {"n": 0, "error": 0.0, "pck": 0} for m in features}
        for group in ("all", "eleven")
    }
    changed_gt = {"better": 0, "worse": 0, "tie": 0, "no_gt": 0}
    reason_counts = {}
    for i, row in enumerate(base.rows(), 1):
        sid = row["id"]
        xy, conf, boxes, d = load_inputs(row, frames[sid])
        reverse, reasons = temporal_decisions(xy, conf, boxes)
        fixed = xy[:, SWAP, :].copy()
        adaptive = fixed.copy()
        adaptive[reverse] = xy[reverse]
        oriented = {"identity": xy, "fixed_swap": fixed, "temporal": adaptive}
        matpath = base.LABELS / f"{sid}.mat"
        if base.sha(matpath) != row["label_sha256"]:
            raise ValueError("GT changed " + sid)
        mat = sio.loadmat(matpath)
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], axis=-1)
        vis = mat["visibility"][:, 1:13] > 0
        wh = d["gt_bbox"][:, 2:] - d["gt_bbox"][:, :2]
        diag = np.linalg.norm(wh, axis=1)
        bbox_ok = np.isfinite(wh).all(1) & (wh[:, 0] > 0) & (wh[:, 1] > 0)
        gtvalid = vis & np.isfinite(gt).all(2) & (gt > 0).all(2) & bbox_ok[:, None]
        valid = np.isfinite(xy).all(2) & (xy > 0).all(2) & np.isfinite(conf) & (conf > 0)
        # Match the previous report's exact paired subset, including confidence selection.
        maxxy = np.full_like(xy, np.nan)
        maxconf = np.zeros_like(conf)
        for t, r in enumerate(frames[sid]):
            idx = int(r["max_index"])
            lo, hi = d["offsets"][t : t + 2]
            if idx >= 0:
                maxxy[t] = d["xy"][lo + idx, 5:17]
                maxconf[t] = d["conf"][lo + idx, 5:17]
        maxvalid = (
            np.isfinite(maxxy).all(2) & (maxxy > 0).all(2) & np.isfinite(maxconf) & (maxconf > 0)
        )
        common = gtvalid & valid & valid[:, SWAP] & maxvalid
        n = int(common.sum())
        v = {
            "id": sid,
            "action": row["action"],
            "in_eleven": sid in ids11,
            "frames": len(xy),
            "common_joints": n,
            "changed_frames": int(reverse.sum()),
        }
        for name, data in oriented.items():
            distance = np.linalg.norm(data - gt, axis=2)
            norm = distance / np.maximum(diag[:, None], 1e-8)
            count = int(common.sum())
            error = float(distance[common].sum())
            pck = int((common & (norm <= 0.2)).sum())
            v[name + "_mean_error_px"] = error / count if count else None
            v[name + "_pck20_pct"] = 100 * pck / count if count else None
            for group in ["all", "eleven"] if sid in ids11 else ["all"]:
                a = accum[group][name]
                a["n"] += count
                a["error"] += error
                a["pck"] += pck
            features[name].append(base.resample(data))
        for t, rec in enumerate(reasons):
            reason_counts[rec["reason"]] = reason_counts.get(rec["reason"], 0) + 1
            rec.update(id=sid, in_eleven=sid in ids11)
            if reverse[t]:
                mask = common[t]
                count = int(mask.sum())
                if count:
                    fixederr = float(np.linalg.norm(fixed[t] - gt[t], axis=1)[mask].mean())
                    adapt_err = float(np.linalg.norm(adaptive[t] - gt[t], axis=1)[mask].mean())
                    status = (
                        "better"
                        if adapt_err < fixederr - 1e-6
                        else ("worse" if adapt_err > fixederr + 1e-6 else "tie")
                    )
                    rec.update(
                        gt_valid_common_joints=count,
                        gt_fixed_error_px=fixederr,
                        gt_temporal_error_px=adapt_err,
                        gt_outcome=status,
                    )
                else:
                    status = "no_gt"
                    rec.update(
                        gt_valid_common_joints=0,
                        gt_fixed_error_px=None,
                        gt_temporal_error_px=None,
                        gt_outcome=status,
                    )
                changed_gt[status] += 1
            else:
                rec.update(
                    gt_valid_common_joints=None,
                    gt_fixed_error_px=None,
                    gt_temporal_error_px=None,
                    gt_outcome="",
                )
            frows.append(rec)
        np.savez_compressed(
            out / "per_video" / f"{sid}.npz",
            identity=xy,
            fixed_swap=fixed,
            temporal=adaptive,
            confidence=conf,
            changed=reverse,
        )
        ids.append(sid)
        labels.append(base.ACTION_CLASSES.index(row["action"]))
        vrows.append(v)
        if i % 50 == 0:
            print(f"{i}/{len(base.rows())} videos", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prediction_rows = []
    preds = {}
    for method, values in features.items():
        X = torch.tensor(np.stack(values), device=device)
        preds[method] = {}
        for seed in (0, 1, 2):
            model = base.ActionBiLSTMAttentionClassifier().to(device)
            model.load_state_dict(
                torch.load(
                    base.CKPTS / f"seed{seed}" / "best.pt", map_location=device, weights_only=True
                )
            )
            model.eval()
            p = []
            with torch.inference_mode():
                for t in range(0, len(X), 32):
                    p.extend(model(X[t : t + 32]).argmax(-1).cpu().tolist())
            preds[method][seed] = p
            prediction_rows.extend(
                {"id": sid, "seed": seed, "method": method, "target": y, "predicted": pr}
                for sid, y, pr in zip(ids, labels, p)
            )
    result = {}
    for group in ("all", "eleven"):
        mask = np.asarray([True if group == "all" else sid in ids11 for sid in ids])
        y = np.asarray(labels)[mask]
        rows = []
        for method in features:
            a = accum[group][method]
            metrics = []
            for seed in (0, 1, 2):
                pp = np.asarray(preds[method][seed])[mask]
                met = base.metric(y, pp)
                old = np.asarray(preds["fixed_swap"][seed])[mask]
                diff = (pp == y).astype(int) - (old == y).astype(int)
                metrics.append(
                    {
                        "seed": seed,
                        **met,
                        "improved_vs_fixed": int((diff > 0).sum()),
                        "worsened_vs_fixed": int((diff < 0).sum()),
                    }
                )
            rows.append(
                {
                    "method": method,
                    "paired_joints": a["n"],
                    "mean_error_px": a["error"] / a["n"] if a["n"] else None,
                    "pck20_common_pct": 100 * a["pck"] / a["n"] if a["n"] else None,
                    "seeds": metrics,
                    "mean_accuracy_pct": float(np.mean([m["accuracy_pct"] for m in metrics])),
                    "mean_macro_f1_pct": float(np.mean([m["macro_f1_pct"] for m in metrics])),
                }
            )
        result[group] = {
            "videos": int(mask.sum()),
            "changed_frames": sum(
                v["changed_frames"] for v in vrows if group == "all" or v["in_eleven"]
            ),
            "videos_with_changes": sum(
                v["changed_frames"] > 0 for v in vrows if group == "all" or v["in_eleven"]
            ),
            "methods": rows,
        }
    summary = {
        "groups": result,
        "decision_reasons": reason_counts,
        "changed_frames_gt_outcome": changed_gt,
        "protocol": "Four original neighbors; all three neighbor comparisons favor same label by >0.02 bbox diagonal; at least 3/4 neighbors favor reversing center by >0.02, at least 1 from each side; min 4 valid joints conf>=0.3 and bbox-center difference<=0.75 bbox diagonal. Start from fixed swap, reverse only isolated frames. Decisions simultaneous; no GT read by rule.",
        "caution": "GT-assisted per-frame selection uses GT and may change person; full validation was used to develop earlier baselines. Noncausal method uses future frames. Temporally smooth states cannot identify absolute left/right convention.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "script_sha256": base.sha(Path(__file__)),
                "source_provenance_sha256": base.sha(SOURCE / "provenance.json"),
                "frame_selection_sha256": base.sha(SOURCE / "frame_matches_iou_0_3.csv"),
                "prior_per_video_sha256": base.sha(ANALYSIS / "per_video.csv"),
                "checkpoint_sha256": source["checkpoints"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for filename, data in [
        ("per_video.csv", vrows),
        ("per_frame_decisions.csv", frows),
        ("predictions.csv", prediction_rows),
    ]:
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    summarize(Path(args.output))
