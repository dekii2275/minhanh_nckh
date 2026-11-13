"""Compare temporal swap windows on the existing GT-assisted validation poses.

The 2-neighbor setting exactly reproduces temporal_swap_analysis.py. Larger
windows preserve the same thresholds, strict neighbor-consistency checks and
75% opposite-vote rule. Future frames are used; this is a diagnostic only.
"""

import argparse
import csv
import json
from functools import lru_cache
from math import ceil
from pathlib import Path

import compare_person_selection as base
import numpy as np
import scipy.io as sio
import temporal_swap_analysis as previous
import torch

ROOT = base.ROOT
RADII = (2, 3, 5, 10)


def decisions_for_radii(xy, conf, boxes, radii=RADII):
    """Compute all windows from the same original observations, without GT."""
    T = len(xy)
    output = {r: np.zeros(T, bool) for r in radii}
    reasons = {r: [] for r in radii}

    @lru_cache(maxsize=None)
    def pair(a, b):
        return previous.compare(xy[a], xy[b], conf[a], conf[b], boxes[a], boxes[b])

    for radius in radii:
        needed = ceil(0.75 * 2 * radius)
        for t in range(T):
            reason = "boundary"
            if t - radius >= 0 and t + radius < T:
                neighbors = list(range(t - radius, t)) + list(range(t + 1, t + radius + 1))
                links = [pair(a, b) for a, b in zip(neighbors, neighbors[1:])]
                if any(c is None for c in links):
                    reason = "neighbor_missing_or_ambiguous"
                elif not all(keep + previous.MARGIN < flip for keep, flip, _ in links):
                    reason = "neighbors_not_consistent"
                else:
                    votes = [pair(n, t) for n in neighbors]
                    if any(c is None for c in votes):
                        reason = "center_missing_or_ambiguous"
                    else:
                        opposite = [flip + previous.MARGIN < keep for keep, flip, _ in votes]
                        if (
                            sum(opposite) >= needed
                            and any(opposite[:radius])
                            and any(opposite[radius:])
                        ):
                            output[radius][t] = True
                            reason = "temporal_reverse"
                        else:
                            reason = "keep_fixed_swap"
            reasons[radius].append(reason)
    return output, reasons


def run(out):
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    out.mkdir(parents=True)
    (out / "per_video").mkdir()
    source = json.loads((previous.SOURCE / "provenance.json").read_text(encoding="utf-8"))
    if source["manifest_sha256"] != base.sha(base.MANIFEST):
        raise ValueError("Manifest mismatch")
    for seed in (0, 1, 2):
        if base.sha(base.CKPTS / f"seed{seed}" / "best.pt") != source["checkpoints"][str(seed)]:
            raise ValueError("Checkpoint mismatch")
    frames = {}
    with (previous.SOURCE / "frame_matches_iou_0_3.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            frames.setdefault(r["id"], []).append(r)
    with (previous.ANALYSIS / "per_video.csv").open(newline="", encoding="utf-8") as f:
        prior = {r["id"]: r for r in csv.DictReader(f)}
    ids11 = {
        sid
        for sid, r in prior.items()
        if r["gt_assisted_paired_error_px"]
        and float(r["gt_assisted_swap_paired_error_px"]) > float(r["gt_assisted_paired_error_px"])
    }
    if len(ids11) != 11:
        raise ValueError("Previous exception set changed")
    methods = ("fixed_swap",) + tuple("radius_" + str(r) for r in RADII)
    features = {m: [] for m in methods}
    ids = []
    labels = []
    video_rows = []
    frame_rows = []
    totals = {g: {m: {"n": 0, "error": 0.0, "pck": 0} for m in methods} for g in ("all", "eleven")}
    outcome = {r: {k: 0 for k in ("better", "worse", "tie", "no_gt")} for r in RADII}
    reason_counts = {r: {} for r in RADII}
    for i, row in enumerate(base.rows(), 1):
        sid = row["id"]
        xy, conf, boxes, d = previous.load_inputs(row, frames[sid])
        decisions, reasons = decisions_for_radii(xy, conf, boxes)
        original, _ = previous.temporal_decisions(xy, conf, boxes)
        if not np.array_equal(original, decisions[2]):
            raise AssertionError("Radius 2 differs from previous rule: " + sid)
        fixed = xy[:, previous.SWAP, :].copy()
        variants = {"fixed_swap": fixed}
        for radius in RADII:
            x = fixed.copy()
            x[decisions[radius]] = xy[decisions[radius]]
            variants["radius_" + str(radius)] = x
        matpath = base.LABELS / f"{sid}.mat"
        if base.sha(matpath) != row["label_sha256"]:
            raise ValueError("Annotation changed " + sid)
        mat = sio.loadmat(matpath)
        gt = np.stack((mat["x"][:, 1:13], mat["y"][:, 1:13]), axis=-1)
        vis = mat["visibility"][:, 1:13] > 0
        wh = d["gt_bbox"][:, 2:] - d["gt_bbox"][:, :2]
        diag = np.linalg.norm(wh, axis=1)
        bbox_ok = np.isfinite(wh).all(1) & (wh[:, 0] > 0) & (wh[:, 1] > 0)
        gtvalid = vis & np.isfinite(gt).all(2) & (gt > 0).all(2) & bbox_ok[:, None]
        valid = np.isfinite(xy).all(2) & (xy > 0).all(2) & np.isfinite(conf) & (conf > 0)
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
        common = gtvalid & valid & valid[:, previous.SWAP] & maxvalid
        v = {
            "id": sid,
            "action": row["action"],
            "in_eleven": sid in ids11,
            "frames": len(xy),
            "common_joints": int(common.sum()),
        }
        for name, pose in variants.items():
            dist = np.linalg.norm(pose - gt, axis=2)
            norm = dist / np.maximum(diag[:, None], 1e-8)
            n = int(common.sum())
            error = float(dist[common].sum())
            pck = int((common & (norm <= 0.2)).sum())
            v[name + "_mean_error_px"] = error / n if n else None
            v[name + "_pck20_pct"] = 100 * pck / n if n else None
            for group in ["all", "eleven"] if sid in ids11 else ["all"]:
                acc = totals[group][name]
                acc["n"] += n
                acc["error"] += error
                acc["pck"] += pck
            features[name].append(base.resample(pose))
        for radius in RADII:
            mask = decisions[radius]
            v["radius_" + str(radius) + "_changed_frames"] = int(mask.sum())
            for t, reason in enumerate(reasons[radius]):
                reason_counts[radius][reason] = reason_counts[radius].get(reason, 0) + 1
                if not mask[t]:
                    continue
                valid_gt = common[t]
                n = int(valid_gt.sum())
                status = "no_gt"
                baseerr = None
                newerr = None
                if n:
                    baseerr = float(np.linalg.norm(fixed[t] - gt[t], axis=1)[valid_gt].mean())
                    newerr = float(np.linalg.norm(xy[t] - gt[t], axis=1)[valid_gt].mean())
                    status = (
                        "better"
                        if newerr < baseerr - 1e-6
                        else ("worse" if newerr > baseerr + 1e-6 else "tie")
                    )
                outcome[radius][status] += 1
                frame_rows.append(
                    {
                        "id": sid,
                        "in_eleven": sid in ids11,
                        "frame": t + 1,
                        "radius": radius,
                        "gt_common_joints": n,
                        "fixed_error_px": baseerr,
                        "temporal_error_px": newerr,
                        "gt_outcome": status,
                    }
                )
        np.savez_compressed(
            out / "per_video" / f"{sid}.npz", **{"changed_" + str(r): decisions[r] for r in RADII}
        )
        ids.append(sid)
        labels.append(base.ACTION_CLASSES.index(row["action"]))
        video_rows.append(v)
        if i % 50 == 0:
            print(f"{i}/253 videos", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = {m: {} for m in methods}
    pred_rows = []
    for m, features_m in features.items():
        X = torch.tensor(np.stack(features_m), device=device)
        for seed in (0, 1, 2):
            model = base.ActionBiLSTMAttentionClassifier().to(device)
            model.load_state_dict(
                torch.load(
                    base.CKPTS / f"seed{seed}" / "best.pt", map_location=device, weights_only=True
                )
            )
            model.eval()
            pred = []
            with torch.inference_mode():
                for t in range(0, len(X), 32):
                    pred.extend(model(X[t : t + 32]).argmax(-1).cpu().tolist())
            predictions[m][seed] = pred
            pred_rows.extend(
                {"id": sid, "seed": seed, "method": m, "target": target, "predicted": yp}
                for sid, target, yp in zip(ids, labels, pred)
            )
    summary = {}
    for group in ("all", "eleven"):
        mask = np.asarray([group == "all" or sid in ids11 for sid in ids])
        y = np.asarray(labels)[mask]
        entries = []
        for m in methods:
            acc = totals[group][m]
            seed_scores = []
            for seed in (0, 1, 2):
                pred = np.asarray(predictions[m][seed])[mask]
                reference = np.asarray(predictions["fixed_swap"][seed])[mask]
                diff = (pred == y).astype(int) - (reference == y).astype(int)
                seed_scores.append(
                    {
                        "seed": seed,
                        "accuracy_pct": 100 * float(np.mean(pred == y)),
                        "improved_vs_fixed": int((diff > 0).sum()),
                        "worsened_vs_fixed": int((diff < 0).sum()),
                    }
                )
            changes = (
                0
                if m == "fixed_swap"
                else sum(
                    v[m + "_changed_frames"] for v in video_rows if group == "all" or v["in_eleven"]
                )
            )
            changed_videos = (
                0
                if m == "fixed_swap"
                else sum(
                    v[m + "_changed_frames"] > 0
                    for v in video_rows
                    if group == "all" or v["in_eleven"]
                )
            )
            entries.append(
                {
                    "method": m,
                    "changed_frames": changes,
                    "videos_with_changes": changed_videos,
                    "paired_joints": acc["n"],
                    "mean_error_px": acc["error"] / acc["n"] if acc["n"] else None,
                    "pck20_common_pct": 100 * acc["pck"] / acc["n"] if acc["n"] else None,
                    "mean_accuracy_pct": float(np.mean([x["accuracy_pct"] for x in seed_scores])),
                    "seeds": seed_scores,
                }
            )
        summary[group] = {"videos": int(mask.sum()), "methods": entries}
    result = {
        "groups": summary,
        "radii": list(RADII),
        "reason_counts": reason_counts,
        "gt_outcome_changed_frames": outcome,
        "protocol": "Radius r uses all r preceding and r following frames; all adjacent neighbor pairs must favor same orientation by >0.02 bbox diagonal. At least ceil(0.75*2r) neighbors must favor reversing center, including both sides. Min 4 joints confidence>=0.3, predicted bbox center movement<=0.75 bbox diagonal. Start from fixed swap; missing/ambiguous -> unchanged.",
        "caution": "Exploratory comparison on previously used validation; per-frame person selected with GT. More neighbors impose stricter data coverage, so the effects of window size and eligibility are coupled. Future frames are used.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "script_sha256": base.sha(Path(__file__)),
                "previous_rule_sha256": base.sha(Path(previous.__file__)),
                "source_provenance_sha256": base.sha(previous.SOURCE / "provenance.json"),
                "frame_selection_sha256": base.sha(previous.SOURCE / "frame_matches_iou_0_3.csv"),
                "prior_per_video_sha256": base.sha(previous.ANALYSIS / "per_video.csv"),
                "checkpoint_sha256": source["checkpoints"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for filename, rows in [
        ("per_video_metrics.csv", video_rows),
        ("changed_frames.csv", frame_rows),
        ("predictions.csv", pred_rows),
    ]:
        if rows:
            with (out / filename).open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(Path(args.output))
