"""Inventory the 2026-11-15 Penn results and pair predictions by video and seed."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/unified_retrain_20261115"
CONTROLS = ROOT / "outputs/pose_controls_20261115"
MANIFEST = ROOT / "report/02112026_nckh/penn_audit_artifacts/video_manifest.csv"
OUT = ROOT / "outputs/penn_prediction_inventory_20261123"
METHODS = ("interpolate3", "mean5", "median5", "abrupt", "combined")
CLASSES = (
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowl",
    "clean_and_jerk",
    "golf_swing",
    "jump_rope",
    "jumping_jacks",
    "pullup",
    "pushup",
    "situp",
    "squat",
    "strum_guitar",
    "tennis_forehand",
    "tennis_serve",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest_rows = read_csv(MANIFEST)
    manifest = {row["id"]: row for row in manifest_rows}
    assert len(manifest) == len(manifest_rows)
    test_ids = {r["id"] for r in manifest_rows if r["official_split"] == "test"}
    assert len(test_ids) == 1068
    summaries = {
        OLD: json.loads((OLD / "summary.json").read_text(encoding="utf-8")),
        CONTROLS: json.loads(
            (CONTROLS / "classification_summary.json").read_text(encoding="utf-8")
        ),
    }
    groups = defaultdict(dict)
    sources = {}
    for directory in (OLD, CONTROLS):
        for row in read_csv(directory / "predictions.csv"):
            method = row["method"]
            mode = row.get("scenario", row.get("mode"))
            key = (method, mode, row["split"], int(row["seed"]))
            video = row["id"]
            assert video not in groups[key], (key, video)
            target = int(row["target"])
            prediction = int(row.get("predicted", row.get("prediction")))
            assert manifest[video]["action"] == CLASSES[target]
            assert 0 <= prediction < len(CLASSES)
            groups[key][video] = (target, prediction)
            sources[key] = directory

    pose = {r["id"]: r for r in read_csv(CONTROLS / "pose_per_video.csv")}
    assert set(pose) == test_ids
    inventory = []
    for key, predictions in sorted(groups.items()):
        method, mode, split, seed = key
        expected = {r["id"] for r in manifest_rows if r["seed0_split"] == split}
        assert set(predictions) == expected, (key, len(predictions), len(expected))
        directory = sources[key]
        meta = next(
            s
            for s in summaries[directory]["methods"][f"{method}/{mode}/{split}"]["seeds"]
            if int(s["seed"]) == seed
        )
        ckdir = OLD if mode == "frozen" else directory
        ckmethod = "raw" if mode == "frozen" else method
        checkpoint = ckdir / "trained" / ckmethod / f"seed{seed}" / "best.pt"
        actual_hash = digest(checkpoint)
        assert actual_hash == meta["checkpoint_sha256"], key
        correct = sum(y == p for y, p in predictions.values())
        assert abs(100 * correct / len(predictions) - meta["accuracy_pct"]) < 1e-8
        inventory.append(
            dict(
                method=method,
                mode=mode,
                split=split,
                seed=seed,
                count=len(predictions),
                correct=correct,
                accuracy_pct=100 * correct / len(predictions),
                epoch=meta["epoch"],
                checkpoint=str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                checkpoint_sha256=actual_hash,
                predictions=str((directory / "predictions.csv").relative_to(ROOT)).replace(
                    "\\", "/"
                ),
            )
        )

    paired = []
    per_video = []
    summary = []
    class_summary = []
    transition_names = ("correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong")
    for method in METHODS:
        for mode in ("frozen", "retrained"):
            counts = Counter()
            grouped = defaultdict(list)
            by_class = defaultdict(Counter)
            for seed in range(3):
                raw = groups["raw", "retrained", "test", seed]
                treatment = groups[method, mode, "test", seed]
                assert set(raw) == set(treatment) == test_ids
                if mode == "frozen":
                    ref = next(
                        r
                        for r in inventory
                        if (r["method"], r["mode"], r["split"], r["seed"])
                        == ("raw", "retrained", "test", seed)
                    )
                    trial = next(
                        r
                        for r in inventory
                        if (r["method"], r["mode"], r["split"], r["seed"])
                        == (method, mode, "test", seed)
                    )
                    assert ref["checkpoint_sha256"] == trial["checkpoint_sha256"]
                for video in sorted(test_ids):
                    target, before = raw[video]
                    target2, after = treatment[video]
                    assert target == target2
                    transition = (
                        ("correct" if before == target else "wrong")
                        + "_"
                        + ("correct" if after == target else "wrong")
                    )
                    counts[transition] += 1
                    by_class[CLASSES[target]][transition] += 1
                    grouped[video].append(transition)
                    p = pose[video]
                    common = int(p["common"])
                    paired.append(
                        dict(
                            method=method,
                            mode=mode,
                            seed=seed,
                            id=video,
                            action=CLASSES[target],
                            target=target,
                            raw_prediction=before,
                            raw_label=CLASSES[before],
                            method_prediction=after,
                            method_label=CLASSES[after],
                            transition=transition,
                            common_joints=common,
                            raw_pose_error_px=float(p["raw_error_sum"]) / common if common else "",
                            method_pose_error_px=float(p[method + "_error_sum"]) / common
                            if common
                            else "",
                            gt_eligible=int(p["eligible"]),
                            raw_covered=int(p["raw_covered"]),
                            changed_gt=int(p[method + "_changed_gt"]),
                            closer=int(p[method + "_closer"]),
                            farther=int(p[method + "_farther"]),
                            pck_better=int(p[method + "_pck_better"]),
                            pck_worse=int(p[method + "_pck_worse"]),
                        )
                    )
            assert sum(counts.values()) == 1068 * 3
            for video, transitions in sorted(grouped.items()):
                per_video.append(
                    dict(
                        method=method,
                        mode=mode,
                        id=video,
                        action=manifest[video]["action"],
                        **{
                            name + "_seeds": ",".join(
                                str(i) for i, x in enumerate(transitions) if x == name
                            )
                            for name in transition_names
                        },
                        **{name + "_count": transitions.count(name) for name in transition_names},
                    )
                )
            summary.append(
                dict(
                    method=method,
                    mode=mode,
                    video_seed_pairs=3192,
                    **{name: counts[name] for name in transition_names},
                    any_correct_wrong=sum("correct_wrong" in t for t in grouped.values()),
                    any_wrong_correct=sum("wrong_correct" in t for t in grouped.values()),
                    both_directions=sum(
                        "correct_wrong" in t and "wrong_correct" in t for t in grouped.values()
                    ),
                    all_seeds_wrong_wrong=sum(
                        t.count("wrong_wrong") == 3 for t in grouped.values()
                    ),
                )
            )
            for action, values in sorted(by_class.items()):
                class_summary.append(
                    dict(
                        method=method,
                        mode=mode,
                        action=action,
                        videos=sum(manifest[v]["action"] == action for v in test_ids),
                        **{name: values[name] for name in transition_names},
                    )
                )

    OUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUT / "inventory.csv", inventory)
    write_csv(OUT / "paired_video_seed.csv", paired)
    write_csv(OUT / "video_summary.csv", per_video)
    write_csv(OUT / "transition_summary.csv", summary)
    write_csv(OUT / "class_summary.csv", class_summary)
    inputs = [
        MANIFEST,
        OLD / "predictions.csv",
        CONTROLS / "predictions.csv",
        OLD / "summary.json",
        CONTROLS / "classification_summary.json",
        CONTROLS / "pose_per_video.csv",
        Path(__file__),
    ]
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "input_sha256": {
                    str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in inputs
                },
                "scope": "Penn historical official test exploratory; 1068 videos and three seeds; GT bbox assisted; offline",
                "validation": "ID, split, target, uniqueness, accuracy, checkpoint hashes and frozen baseline checkpoints checked",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
