"""Complete YOLO controls and matched video-level pose/motion scoring."""

import csv
import json
import sys
from pathlib import Path

import compare_person_selection as base
import evaluate_abrupt_joint_interpolation_20261113 as abrupt
import evaluate_pose_corrections as correction
import numpy as np
import retrain_unified_20261115 as run
import scipy.io as sio
import targeted_pose_pilot_20261113 as target
import torch
import train_gt_baselines as trainer

ROOT = trainer.ROOT
OUT = ROOT / "outputs/pose_controls_20261115"
OLD = ROOT / "outputs/unified_retrain_20261115"
TEST = ROOT / "outputs/confidence_confirmation_penn_20261105/test_pose/poses"
NEW = ("interpolate3", "mean5", "median5")
ALL = ("raw", "interpolate3", "mean5", "median5", "abrupt", "combined")


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def prepare():
    assert torch.cuda.is_available(), "CUDA required for training/inference"
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    rows = sorted(
        (
            r
            for r in csv.DictReader(base.MANIFEST.open(encoding="utf-8-sig", newline=""))
            if r["seed0_split"] == "test"
        ),
        key=lambda r: r["id"],
    )
    assert len(rows) == 1068 and len({r["id"] for r in rows}) == 1068
    with np.load(OLD / "features_raw.npz") as d:
        old_ids = d["test_ids"].copy()
        old_y = d["test_y"].copy()
        old_raw = d["test_x"].copy()
    assert np.array_equal(old_ids, [r["id"] for r in rows])
    refs = {}
    for name in ("abrupt", "combined"):
        with np.load(OLD / f"features_{name}.npz") as d:
            assert np.array_equal(d["test_ids"], old_ids) and np.array_equal(d["test_y"], old_y)
            refs[name] = d["test_x"].copy()
    features = {m: [] for m in NEW}
    per = []
    hashes = []
    for i, r in enumerate(rows):
        path = TEST / (r["id"] + ".npz")
        label = base.LABELS / (r["id"] + ".mat")
        assert trainer.sha(label) == r["label_sha256"]
        mat = sio.loadmat(label)
        assert int(mat["train"][0, 0]) == -1 and str(mat["action"][0]) == r["action"]
        with np.load(path) as d:
            assert str(d["annotation_sha256"]) == r["label_sha256"] and len(d["xy"]) == int(
                r["nframes"]
            )
            xy, conf, status = d["xy"].copy(), d["conf"].copy(), d["status"].copy()
        variants, filled = correction.make_variants(xy, conf)
        raw, rc = variants["fixed_swap"]
        j, jk = abrupt.correct(raw, rc)
        c, ck = target.correct(raw, rc)
        points = {
            "raw": raw,
            "interpolate3": variants["interpolate3"][0],
            "mean5": variants["mean5"][0],
            "median5": variants["median5"][0],
            "abrupt": j,
            "combined": c,
        }
        assert np.array_equal(base.resample(raw), old_raw[i])
        assert np.array_equal(base.resample(j), refs["abrupt"][i]) and np.array_equal(
            base.resample(c), refs["combined"][i]
        )
        for m in NEW:
            features[m].append(base.resample(points[m]))
        gt = np.stack([mat["x"][:, 1:13], mat["y"][:, 1:13]], axis=-1).astype(float)
        vis = mat["visibility"][:, 1:13] > 0
        bb = np.full((len(raw), 4), np.nan)
        n = min(len(raw), len(mat["bbox"]))
        bb[:n] = mat["bbox"][:n]
        wh = bb[:, 2:] - bb[:, :2]
        diag = np.linalg.norm(wh, axis=1)
        bok = np.isfinite(bb).all(1) & (wh > 0).all(1)
        eligible = vis & target.valid(gt) & bok[:, None]
        rawobs = target.valid(raw) & (status[:, None] == 3)
        observed = {
            m: target.valid(points[m])
            & (
                (status[:, None] == 3)
                | (
                    filled
                    if m == "interpolate3"
                    else (ck > 0 if m == "combined" else np.zeros_like(filled))
                )
            )
            for m in ALL
        }
        common = eligible & rawobs
        for m in ALL:
            common &= observed[m]
        pair = common[1:] & common[:-1]
        gt_delta = gt[1:] - gt[:-1]
        row = {
            "id": r["id"],
            "eligible": int(eligible.sum()),
            "common": int(common.sum()),
            "motion_n": int(pair.sum()),
        }
        raw_good = eligible & rawobs & (np.linalg.norm(raw - gt, axis=2) <= 0.2 * diag[:, None])
        raw_err = np.linalg.norm(raw - gt, axis=2)
        for m in ALL:
            p = points[m]
            obs = observed[m]
            dist = np.linalg.norm(p - gt, axis=2)
            good = eligible & obs & (dist <= 0.2 * diag[:, None])
            delta = (p[1:] - p[:-1]) - gt_delta
            motion = np.linalg.norm(delta, axis=2) / diag[1:, None]
            changed = ~np.all((p == raw) | (np.isnan(p) & np.isnan(raw)), axis=2)
            changed_gt = changed & eligible
            was = changed_gt & rawobs & obs
            row.update(
                {
                    m + "_pck": int(good.sum()),
                    m + "_covered": int((eligible & obs).sum()),
                    m + "_error_sum": float(dist[common].sum()),
                    m + "_motion_sum": float(motion[pair].sum()),
                    m + "_changed_gt": int(changed_gt.sum()),
                    m + "_closer": int((was & (dist < raw_err)).sum()),
                    m + "_farther": int((was & (dist > raw_err)).sum()),
                    m + "_pck_better": int((good & ~raw_good & changed_gt).sum()),
                    m + "_pck_worse": int((~good & raw_good & changed_gt).sum()),
                    m + "_original_missing": int((changed_gt & ~rawobs).sum()),
                }
            )
        per.append(row)
        hashes.append(
            {
                "id": r["id"],
                "annotation_sha256": r["label_sha256"],
                "pose_sha256": trainer.sha(path),
            }
        )
        if (i + 1) % 100 == 0:
            print("prepared", i + 1, "/", len(rows), flush=True)
    for m in NEW:
        path = ROOT / "outputs/gt_crop_corrections_20261103" / f"features_{m}.npz"
        expected = json.loads((path.parent / "prepared.json").read_text())["feature_sha256"][m]
        assert trainer.sha(path) == expected
        with np.load(path) as z:
            train = {k: z[k].copy() for k in ("x", "ids", "y", "split")}
        with np.load(OLD / "features_raw.npz") as z:
            assert (
                np.array_equal(train["ids"], z["ids"])
                and np.array_equal(train["y"], z["y"])
                and np.array_equal(train["split"], z["split"])
            )
        np.savez_compressed(
            OUT / f"features_{m}.npz",
            **train,
            test_x=np.stack(features[m]),
            test_y=old_y,
            test_ids=old_ids,
        )
    with (OUT / "pose_per_video.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    write(OUT / "input_hashes.json", hashes)
    summary = {"videos": len(rows), "methods": {}, "comparisons": {}, "bootstrap_seed": 20261115}
    rng = np.random.default_rng(20261115)
    idx = rng.integers(0, len(per), (10000, len(per)))

    def col(k):
        return np.array([r[k] for r in per], dtype=float)

    denominators = {
        "eligible": col("eligible"),
        "common": col("common"),
        "motion_n": col("motion_n"),
    }
    for m in ALL:
        sums = {
            k: int(col(m + "_" + k).sum())
            for k in (
                "pck",
                "covered",
                "changed_gt",
                "closer",
                "farther",
                "pck_better",
                "pck_worse",
                "original_missing",
            )
        }
        sums.update(
            {
                "pck_pct": float(100 * sums["pck"] / denominators["eligible"].sum()),
                "coverage_pct": float(100 * sums["covered"] / denominators["eligible"].sum()),
                "error_px": float(col(m + "_error_sum").sum() / denominators["common"].sum()),
                "motion_error_norm": float(
                    col(m + "_motion_sum").sum() / denominators["motion_n"].sum()
                ),
                "motion_video_mean": float(
                    np.mean(
                        [r[m + "_motion_sum"] / r["motion_n"] for r in per if r["motion_n"] > 0]
                    )
                ),
            }
        )
        summary["methods"][m] = sums
        if m != "raw":
            p = 100 * (col(m + "_pck") - col("raw_pck"))
            q = col(m + "_motion_sum") - col("raw_motion_sum")
            summary["comparisons"][m] = {
                "pck_delta_pp": float(p.sum() / denominators["eligible"].sum()),
                "pck_delta_bootstrap95_pp": np.quantile(
                    p[idx].sum(1) / denominators["eligible"][idx].sum(1), [0.025, 0.975]
                ).tolist(),
                "motion_delta": float(q.sum() / denominators["motion_n"].sum()),
                "motion_delta_bootstrap95": np.quantile(
                    q[idx].sum(1) / denominators["motion_n"][idx].sum(1), [0.025, 0.975]
                ).tolist(),
            }
    summary["denominators"] = {k: int(v.sum()) for k, v in denominators.items()}
    write(OUT / "pose_summary.json", summary)
    write(
        OUT / "protocol.json",
        {
            "base_protocol_sha256": trainer.sha(
                ROOT / "report/13112026_nckh/PROTOCOL_THONG_NHAT_PENN_2026-11-15.md"
            ),
            "manifest_sha256": trainer.sha(base.MANIFEST),
            "script_sha256": trainer.sha(Path(__file__)),
            "correction_sha256": trainer.sha(Path(correction.__file__)),
            "abrupt_sha256": trainer.sha(Path(abrupt.__file__)),
            "combined_sha256": trainer.sha(Path(target.__file__)),
            "yolo_provenance_sha256": trainer.sha(TEST.parent / "provenance.json"),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "scope": "Historical test exploratory, 1005 train / 253 validation / 1068 test; GT bbox assisted, offline; paired common mask, missing counts as incorrect; 3 new branches 3 seeds; no method selection on test.",
        },
    )
    print("pose preparation complete", flush=True)


def fit():
    run.OUT = OUT
    run.METHODS = {m: (None, None, 24) for m in NEW}
    run.fit()


def evaluate():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    allpred = {}
    scores = []
    for m in NEW:
        with np.load(OUT / f"features_{m}.npz") as z:
            datasets = {
                "validation": (
                    z["x"][z["split"] == "validation"].copy(),
                    z["y"][z["split"] == "validation"].copy(),
                    z["ids"][z["split"] == "validation"].copy(),
                ),
                "test": (z["test_x"].copy(), z["test_y"].copy(), z["test_ids"].copy()),
            }
        for mode in ("frozen", "retrained"):
            for seed in range(3):
                ck = (
                    (OLD / "trained/raw" if mode == "frozen" else OUT / "trained" / m)
                    / f"seed{seed}"
                    / "best.pt"
                )
                meta = json.loads((ck.parent / "provenance.json").read_text())
                assert trainer.sha(ck) == meta["checkpoint_sha256"]
                net = trainer.model_for("bilstm_attention").cuda()
                net.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
                net.eval()
                for split, (x, y, ids) in datasets.items():
                    with torch.inference_mode():
                        p = torch.cat(
                            [net(b.cuda()).argmax(-1).cpu() for b in torch.from_numpy(x).split(32)]
                        ).numpy()
                    scores.append(
                        {
                            "method": m,
                            "mode": mode,
                            "seed": seed,
                            "split": split,
                            "epoch": meta["best_epoch"],
                            "checkpoint_sha256": meta["checkpoint_sha256"],
                            **base.metric(y, p),
                        }
                    )
                    allpred[m, mode, seed, split] = (ids, y, p)
                del net
    prior = list(csv.DictReader((OLD / "predictions.csv").open(encoding="utf-8-sig", newline="")))
    summary = {"methods": {}, "comparisons": {}, "device": torch.cuda.get_device_name(0)}
    rng = np.random.default_rng(20261115)
    for m in NEW:
        for mode in ("frozen", "retrained"):
            for split in ("validation", "test"):
                group = [
                    r
                    for r in scores
                    if r["method"] == m and r["mode"] == mode and r["split"] == split
                ]
                name = f"{m}/{mode}/{split}"
                summary["methods"][name] = {
                    "accuracy_mean_pct": float(np.mean([r["accuracy_pct"] for r in group])),
                    "accuracy_sd_pct": float(np.std([r["accuracy_pct"] for r in group], ddof=1)),
                    "macro_f1_mean_pct": float(np.mean([r["macro_f1_pct"] for r in group])),
                    "seeds": group,
                }
                diffs = []
                for seed in range(3):
                    ids, y, p = allpred[m, mode, seed, split]
                    raw = [
                        r
                        for r in prior
                        if r["method"] == "raw"
                        and r["scenario"] == "retrained"
                        and r["split"] == split
                        and int(r["seed"]) == seed
                    ]
                    assert np.array_equal(ids, [r["id"] for r in raw]) and np.array_equal(
                        y, [int(r["target"]) for r in raw]
                    )
                    rp = np.array([int(r["predicted"]) for r in raw])
                    diffs.append((p == y).astype(int) - (rp == y).astype(int))
                d = np.mean(diffs, axis=0)
                idx = rng.integers(0, len(d), (10000, len(d)))
                summary["comparisons"][name] = {
                    "delta_pp": float(100 * d.mean()),
                    "bootstrap95_pp": np.quantile(100 * d[idx].mean(1), [0.025, 0.975]).tolist(),
                }
    with (OUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mode", "seed", "split", "id", "target", "prediction"])
        for (m, mode, seed, split), (ids, y, p) in allpred.items():
            w.writerows(
                zip([m] * len(y), [mode] * len(y), [seed] * len(y), [split] * len(y), ids, y, p)
            )
    write(OUT / "classification_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "train": fit, "evaluate": evaluate}[sys.argv[1]]()
