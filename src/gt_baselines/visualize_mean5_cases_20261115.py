"""List video-level raw/mean5 paired errors and render real-frame evidence."""

import csv
import html
import json

import compare_person_selection as base
import cv2
import evaluate_pose_corrections as corr
import numpy as np
import targeted_pose_pilot_20261113 as target
from PIL import Image
from previous_joint_visual_pilot_20261113 import render
from scipy.io import loadmat

ROOT = base.ROOT
OUT = ROOT / "outputs/mean5_case_review_20261115"
POSE = ROOT / "outputs/pose_controls_20261115/pose_per_video.csv"
RAW = ROOT / "outputs/unified_retrain_20261115/predictions.csv"
MEAN = ROOT / "outputs/pose_controls_20261115/predictions.csv"
CACHE = ROOT / "outputs/confidence_confirmation_penn_20261105/test_pose/poses"
FRAMES = ROOT / "data/raw/Penn_Action/frames"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    if (OUT / "index.html").exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thumbnails").mkdir(exist_ok=True)
    (OUT / "clips").mkdir(exist_ok=True)
    pose = {r["id"]: r for r in read(POSE)}
    manifest = {r["id"]: r for r in read(base.MANIFEST) if r["seed0_split"] == "test"}
    raw = {
        (r["id"], int(r["seed"])): r
        for r in read(RAW)
        if r["method"] == "raw" and r["scenario"] == "retrained" and r["split"] == "test"
    }
    means = {
        (r["id"], int(r["seed"]), r["mode"]): r
        for r in read(MEAN)
        if r["method"] == "mean5" and r["split"] == "test"
    }
    assert len(raw) == 3204 and len(manifest) == 1068 and len(means) == 6408
    rows = []
    for sid in sorted(manifest):
        q = pose[sid]
        common = int(q["common"])
        eligible = int(q["eligible"])
        d = (float(q["mean5_error_sum"]) - float(q["raw_error_sum"])) / common if common else None
        pck = (int(q["mean5_pck"]) - int(q["raw_pck"])) / eligible * 100 if eligible else None
        row = {
            "id": sid,
            "action": manifest[sid]["action"],
            "pose_error_raw_px": float(q["raw_error_sum"]) / common if common else None,
            "pose_error_mean_px": float(q["mean5_error_sum"]) / common if common else None,
            "pose_error_delta_px": d,
            "pck_delta_pp": pck,
            "common_joints": common,
            "missing_gt_joints": eligible - int(q["raw_covered"]),
        }
        for mode in ("retrained", "frozen"):
            plus = []
            minus = []
            fields = []
            for seed in range(3):
                a = raw[sid, seed]
                b = means[sid, seed, mode]
                assert a["target"] == b["target"]
                y = int(a["target"])
                pa = int(a["predicted"])
                pb = int(b["prediction"])
                if pa == y and pb != y:
                    minus.append(seed)
                if pa != y and pb == y:
                    plus.append(seed)
                fields.append(f"{seed}:{base.ACTION_CLASSES[pa]}>{base.ACTION_CLASSES[pb]}")
            row[mode + "_worse_seeds"] = ",".join(map(str, minus))
            row[mode + "_better_seeds"] = ",".join(map(str, plus))
            row[mode + "_worse_count"] = len(minus)
            row[mode + "_better_count"] = len(plus)
            row[mode + "_predictions"] = "; ".join(fields)
        row["pose_better_recognition_worse"] = bool(
            d is not None and d < -1e-7 and row["retrained_worse_count"]
        )
        row["recognition_better"] = bool(row["retrained_better_count"])
        if row["pose_better_recognition_worse"] or row["recognition_better"]:
            rows.append(row)
    # Every selected video gets a labeled image; six also get short animations.
    bad = sorted(
        (r for r in rows if r["pose_better_recognition_worse"]),
        key=lambda r: (
            -r["retrained_worse_count"],
            r["retrained_better_count"],
            r["pose_error_delta_px"],
        ),
    )
    good = sorted(
        (r for r in rows if r["recognition_better"]),
        key=lambda r: (
            -r["retrained_better_count"],
            r["retrained_worse_count"],
            r["pose_error_delta_px"] if r["pose_error_delta_px"] is not None else 0,
        ),
    )
    picks = []
    for group, items in [("worse", bad), ("better", good)]:
        for r in items:
            if r["id"] not in [p[1] for p in picks]:
                picks.append((group, r["id"]))
            if sum(v[0] == group for v in picks) == 3:
                break
    selected = dict((sid, group) for group, sid in picks)
    for i, r in enumerate(rows):
        sid = r["id"]
        path = CACHE / (sid + ".npz")
        label = base.LABELS / (sid + ".mat")
        assert target.sha(label) == manifest[sid]["label_sha256"]
        with np.load(path) as z:
            assert str(z["annotation_sha256"]) == manifest[sid]["label_sha256"]
            xy = z["xy"].copy()
            conf = z["conf"].copy()
            status = z["status"].copy()
        variants, _ = corr.make_variants(xy, conf)
        x, c = variants["fixed_swap"]
        y = variants["mean5"][0]
        mat = loadmat(label)
        gt = np.stack((mat["x"][:, 1:13], mat["y"][:, 1:13]), axis=-1).astype(float)
        vis = mat["visibility"][:, 1:13] > 0
        bb = np.full((len(x), 4), np.nan)
        n = min(len(x), len(mat["bbox"]))
        bb[:n] = mat["bbox"][:n]
        wh = bb[:, 2:] - bb[:, :2]
        diag = np.linalg.norm(wh, axis=1)
        eligible = (
            vis
            & target.valid(gt)
            & target.valid(x)
            & (status[:, None] == 3)
            & np.isfinite(bb).all(1)[:, None]
            & (wh > 0).all(1)[:, None]
        )
        e0 = np.linalg.norm(x - gt, axis=2)
        e1 = np.linalg.norm(y - gt, axis=2)
        change = (e1 - e0) / np.maximum(diag[:, None], 1e-8)
        change = np.where(eligible, change, np.nan)
        worse = r["pose_better_recognition_worse"] and (
            r["retrained_worse_count"] >= r["retrained_better_count"]
        )
        # Frame chosen post hoc to expose the strongest local change, not typicality.
        if np.isfinite(change).any():
            flat = np.nanargmax(change) if worse else np.nanargmin(change)
            t, j = np.unravel_index(flat, change.shape)
        else:
            t = len(x) // 2
            j = 0
        r.update(
            {
                "example_frame": int(t + 1),
                "example_joint": target.NAMES[j],
                "example_error_before_px": float(e0[t, j]) if eligible[t, j] else None,
                "example_error_after_px": float(e1[t, j]) if eligible[t, j] else None,
                "example_frame_selected": "largest local error increase"
                if worse
                else "largest local error decrease",
                "notes": "Example selected after observing GT and outcomes; not proof of cause.",
            }
        )

        def frame_image(index):
            im = cv2.imread(str(FRAMES / sid / f"{index + 1:06d}.jpg"))
            if im is None:
                raise FileNotFoundError(str(FRAMES / sid / f"{index + 1:06d}.jpg"))
            marks = np.zeros(12, dtype=bool)
            marks[j] = True
            left = render(
                im,
                x[index],
                c[index],
                gt[index],
                vis[index],
                bb[index],
                marks,
                f"{sid} f{index + 1} | RAW",
            )
            right = render(
                im,
                y[index],
                c[index],
                gt[index],
                vis[index],
                bb[index],
                marks,
                f"{sid} f{index + 1} | MEAN 5",
            )
            return np.hstack((left, right))

        image = frame_image(t)
        h, w = image.shape[:2]
        image = cv2.resize(image, (900, max(1, int(h * 900 / w))))
        cv2.imwrite(str(OUT / "thumbnails" / f"{sid}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if sid in selected:
            first = max(0, t - 4)
            last = min(len(x), t + 5)
            images = []
            for f in range(first, last):
                im = frame_image(f)
                h, w = im.shape[:2]
                im = cv2.resize(im, (800, max(1, int(h * 800 / w))))
                images.append(
                    Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).convert(
                        "P", palette=Image.Palette.ADAPTIVE, colors=128
                    )
                )
            images[0].save(
                OUT / "clips" / f"{sid}.gif",
                save_all=True,
                append_images=images[1:],
                duration=250,
                loop=0,
            )
        if (i + 1) % 25 == 0:
            print("rendered", i + 1, "/", len(rows), flush=True)
    fields = list(rows[0])
    with (OUT / "video_cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    counts = {
        "all_test_videos": 1068,
        "worse_pose_better": len(bad),
        "recognition_better": len(good),
        "union": len(rows),
        "overlap": sum(
            r["pose_better_recognition_worse"] and r["recognition_better"] for r in rows
        ),
        "selection": picks,
        "definition": "at least one seed correct-to-wrong or wrong-to-correct; pose improved is average Euclidean error on common eligible joints lower; train raw vs independently retrained mean5; videos may belong to both groups",
        "limitations": "observational, groups selected after test; does not establish cause. Fixed pose left/right mapping, GT bbox assistance, labels may have error.",
    }
    (OUT / "summary.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf8"
    )

    def table(title, items):
        chunks = [f"<h2>{html.escape(title)} ({len(items)} video)</h2>"]
        for r in items:
            sid = r["id"]
            link = f'<a href="clips/{sid}.gif">Xem GIF</a>' if sid in selected else ""
            delta = (
                f"{r['pose_error_delta_px']:.2f} px"
                if r["pose_error_delta_px"] is not None
                else "không đủ khớp hợp lệ"
            )
            chunks.append(
                f'<article><h3>{sid} — {html.escape(r["action"])} {link}</h3><p>Seed đúng→sai: {r["retrained_worse_seeds"] or "không"}; sai→đúng: {r["retrained_better_seeds"] or "không"}; Δ sai số pose: {delta}; điểm minh họa: frame {r["example_frame"]}, {r["example_joint"]}.</p><img loading="lazy" src="thumbnails/{sid}.jpg" alt="{sid}"></article>'
            )
        return "\n".join(chunks)

    page = (
        '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Đối chiếu lọc skeleton</title><style>body{font:16px Arial;max-width:1050px;margin:auto;color:#203044}article{border-top:1px solid #ccd;padding:10px}img{max-width:100%;height:auto}h2{background:#e9f0fa;padding:10px}</style><h1>Video test: pose gốc và lọc trung bình 5 frame</h1><p>Ảnh bên trái: gốc; bên phải: lọc. Xanh lá: annotation GT; đỏ: pose YOLO; vòng xanh lam: khớp chọn minh họa. Frame được chọn sau khi xem lỗi, không đại diện toàn video. Mỗi kết quả so sánh hai classifier train riêng, không chứng minh bộ lọc là nguyên nhân trực tiếp. Một video có thể xuất hiện trong cả hai nhóm vì seed khác nhau.</p>'
        + table("Pose gần GT hơn nhưng có seed nhận diện đúng thành sai", bad)
        + table("Có seed nhận diện sai thành đúng", good)
        + "</html>"
    )
    (OUT / "index.html").write_text(page, encoding="utf8")
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
