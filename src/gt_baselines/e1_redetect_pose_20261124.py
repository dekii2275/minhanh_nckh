"""E1 - Giam mat nguoi o buoc trich pose: chay lai YOLO voi nhieu cau hinh tren 91 video
cua phieu buoc 1, roi danh gia frozen bang checkpoint raw da co (BiLSTM + CascadeFormer, 3 seed).

Chay tai thu muc goc du an (PowerShell):
    .\\.venv\\Scripts\\python.exe .\\src\\gt_baselines\\e1_redetect_pose_20261124.py infer
    .\\.venv\\Scripts\\python.exe .\\src\\gt_baselines\\e1_redetect_pose_20261124.py evaluate

`infer` co the chay lai nhieu lan: video da co cache se bo qua. `--configs` de chon cau hinh,
`--all-test` de chay toan bo 1068 video test (lau hon nhieu).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics_config"))
sys.path.insert(0, str(ROOT / "src/gt_baselines"))
import compare_person_selection as base  # noqa: E402
import cv2  # noqa: E402
import train_gt_baselines as trainer  # noqa: E402
import ultralytics  # noqa: E402
from train_gt_cascade_scratch import CascadeScratch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

OUT = ROOT / "outputs/e1_redetect_20261124"
QUEUE = ROOT / "outputs/error_attribution_20261124/phieu_gan_nhan_loi_buoc1.csv"
LABELS_REVIEWED = ROOT / "outputs/error_attribution_20261124/nhan_loi_buoc1_claude.csv"
BASE_CACHE = ROOT / "outputs/confidence_confirmation_penn_20261105/test_pose/poses"
TEST_FEATURES = ROOT / "outputs/requested_test_gpu_20261115/features.npz"
CKPT_BILSTM = ROOT / "outputs/unified_retrain_20261115/trained/raw"
CKPT_CASCADE = ROOT / "outputs/cascade_three_inputs_20261123/trained/raw"
SWAP = np.arange(12) ^ 1

# ten: (model, conf, margin moi canh, nguong IoU, fallback toan anh khi khong khop)
CONFIGS = {
    "n_base": ("yolo11n-pose.pt", 0.25, 0.25, 0.30, False),  # tai lap pipeline goc
    "n_relaxed": ("yolo11n-pose.pt", 0.10, 0.50, 0.10, True),
    "m_base": ("yolo11m-pose.pt", 0.25, 0.25, 0.30, False),
    "x_base": ("yolo11x-pose.pt", 0.25, 0.25, 0.30, False),
    "x_relaxed": ("yolo11x-pose.pt", 0.10, 0.50, 0.10, True),
}
# Bien the khong can suy luan them: giu cache goc, chi thay frame bi mat nguoi bang cau hinh khac.
FILLS = {"fill_n_relaxed": "n_relaxed", "fill_x_relaxed": "x_relaxed"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def manifest_rows(all_test):
    rows = {r["id"]: r for r in read_csv(base.MANIFEST) if r["official_split"] == "test"}
    assert len(rows) == 1068
    if all_test:
        attr = ROOT / "outputs/error_attribution_20261124/video_error_attribution.csv"
        cat = {r["id"].zfill(4): r["cat"] for r in read_csv(attr)} if attr.exists() else {}
        short = {
            "A_sai_do_khung_xuong": "A",
            "B_sai_ca_hai": "B",
            "C_chi_GT_sai": "C",
            "D_dung_ca_hai": "D",
        }
        return [dict(rows[k], nhom=short.get(cat.get(k, ""), "")) for k in sorted(rows)]
    queue = read_csv(QUEUE)
    return [dict(rows[q["id"]], nhom=q["nhom"]) for q in queue]


def crop_rect(box, w, h, margin):
    box = np.asarray(box, float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    dx, dy = (box[2:] - box[:2]) * margin
    x1, y1 = max(0, int(np.floor(box[0] - dx))), max(0, int(np.floor(box[1] - dy)))
    x2, y2 = min(w, int(np.ceil(box[2] + dx))), min(h, int(np.ceil(box[3] + dy)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def load_model(name):
    path = ROOT / "models" / name
    if not path.is_file():
        print(f"Chua co {path.name}, Ultralytics se tai ve (can internet)...", flush=True)
        m = YOLO(name)  # tai ve thu muc hien tai
        src = Path(name)
        if src.is_file():
            src.replace(path)
            m = YOLO(str(path))
        return m
    return YOLO(str(path))


def pick(result, offset, gt_box, iou_min):
    """Chon detection co IoU lon nhat voi bbox GT (giong pipeline goc)."""
    if not len(result.boxes):
        return 1, None, None
    boxes = result.boxes.xyxy.cpu().numpy().copy() + np.array([*offset, *offset])
    scores = base.ious(boxes, gt_box)
    if not np.isfinite(scores).all() or scores.max() < iou_min:
        return 2, None, None
    i = int(np.argmax(scores))
    p = result.keypoints.xy[i, 5:17].cpu().numpy().copy()
    c = result.keypoints.conf[i, 5:17].cpu().numpy().copy()
    valid = np.isfinite(p).all(1) & (p > 0).all(1)
    p[valid] += np.array(offset)
    p[~valid] = np.nan
    return 3, p, c


def infer(args):
    rows = manifest_rows(args.all_test)
    for name in args.configs:
        model_name, conf, margin, iou_min, fallback = CONFIGS[name]
        out = OUT / name / "poses"
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(
            model=model_name,
            conf=conf,
            margin_each_side=margin,
            iou_min=iou_min,
            fallback_full_frame=fallback,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            ultralytics=ultralytics.__version__,
            torch=torch.__version__,
            select="max IoU vs GT bbox; keypoints COCO 5:17; fixed swap applied at evaluation",
        )
        mf = OUT / name / "config.json"
        if mf.exists() and json.loads(mf.read_text(encoding="utf-8")) != meta:
            raise ValueError(f"Cau hinh {name} khac lan chay truoc; doi ten hoac xoa {OUT / name}")
        mf.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        model = load_model(model_name)
        t0 = time.time()
        for k, row in enumerate(rows, 1):
            sid = row["id"]
            dest = out / f"{sid}.npz"
            if dest.exists():
                continue
            T = int(row["nframes"])
            bbox = np.asarray(sio.loadmat(base.LABELS / f"{sid}.mat")["bbox"], float)
            xy = np.full((T, 12, 2), np.nan, np.float32)
            cf = np.zeros((T, 12), np.float32)
            st = np.zeros(T, np.uint8)
            frames = {}
            for s in range(0, T, args.batch):
                imgs, data = [], []
                for t in range(s, min(T, s + args.batch)):
                    img = cv2.imdecode(
                        np.fromfile(str(base.FRAMES / sid / f"{t + 1:06d}.jpg"), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    frames[t] = img
                    rect = (
                        crop_rect(bbox[t], img.shape[1], img.shape[0], margin)
                        if t < len(bbox)
                        else None
                    )
                    if rect is None:
                        continue
                    x1, y1, x2, y2 = rect
                    imgs.append(img[y1:y2, x1:x2])
                    data.append((t, x1, y1))
                if not imgs:
                    continue
                res = model.predict(
                    source=imgs, imgsz=640, conf=conf, device=args.device, verbose=False
                )
                for r, (t, x1, y1) in zip(res, data):
                    st[t], p, c = pick(r, (x1, y1), bbox[t], iou_min)
                    if p is not None:
                        xy[t], cf[t] = p, c
            if fallback:  # frame chua khop: thu tren toan anh
                miss = [
                    t
                    for t in range(T)
                    if st[t] != 3 and t < len(bbox) and np.isfinite(bbox[t]).all()
                ]
                for s in range(0, len(miss), args.batch):
                    ts = miss[s : s + args.batch]
                    res = model.predict(
                        source=[frames[t] for t in ts],
                        imgsz=640,
                        conf=conf,
                        device=args.device,
                        verbose=False,
                    )
                    for r, t in zip(res, ts):
                        code, p, c = pick(r, (0, 0), bbox[t], iou_min)
                        if p is not None:
                            xy[t], cf[t], st[t] = p, c, 3
                        else:
                            st[t] = max(st[t], code)
            tmp = out / f"{sid}.tmp.npz"
            np.savez_compressed(tmp, xy=xy, conf=cf, status=st)
            os.replace(tmp, dest)
            print(
                f"[{name}] {k}/{len(rows)} {sid} frames={T} matched={(st == 3).mean():.0%} {time.time() - t0:.0f}s",
                flush=True,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_pose(source, sid):
    with np.load(source / f"{sid}.npz") as d:
        return (
            d["xy"].astype(np.float32),
            d["conf"].astype(np.float32),
            d["status"].astype(np.uint8),
        )


def pose_metrics(sid, xy_s, status):
    """xy_s da fixed swap. Tra ve tong so de cong don: frame co GT bbox, frame khop, khop du dieu kien, PCK@0.2."""
    m = sio.loadmat(base.LABELS / f"{sid}.mat")
    gt = np.stack([m["x"][:, 1:13], m["y"][:, 1:13]], -1).astype(float)
    vis = m["visibility"][:, 1:13] > 0
    n = min(len(gt), len(xy_s))
    bb = np.full((n, 4), np.nan)
    k = min(n, len(m["bbox"]))
    bb[:k] = m["bbox"][:k]
    wh = bb[:, 2:] - bb[:, :2]
    bok = np.isfinite(bb).all(1) & (wh > 0).all(1)
    diag = np.linalg.norm(wh, axis=1)
    elig = vis[:n] & (gt[:n] > 0).all(-1) & bok[:, None]
    obs = (status[:n, None] == 3) & np.isfinite(xy_s[:n]).all(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.linalg.norm(xy_s[:n] - gt[:n], axis=-1) / diag[:, None]
    return np.array(
        [bok.sum(), (bok & (status[:n] == 3)).sum(), elig.sum(), (elig & obs & (d <= 0.2)).sum()]
    )


def load_nets(device):
    nets = []
    for kind, folder in (("bilstm", CKPT_BILSTM), ("cascade", CKPT_CASCADE)):
        for seed in range(3):
            ck = folder / f"seed{seed}" / "best.pt"
            meta = json.loads((ck.parent / "provenance.json").read_text(encoding="utf-8"))
            assert trainer.sha(ck) == meta["checkpoint_sha256"], ck
            net = (
                CascadeScratch()
                if kind == "cascade"
                else trainer.ActionBiLSTMAttentionClassifier(input_size=24)
            )
            net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            nets.append((f"{kind}_s{seed}", net.to(device).eval()))
    return nets


def predict(nets, feats, device):
    x = torch.from_numpy(np.stack(feats))
    out = {}
    with torch.inference_mode():
        for name, net in nets:
            out[name] = torch.cat([net(b.to(device)).argmax(-1).cpu() for b in x.split(32)]).numpy()
    return out


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = manifest_rows(args.all_test)
    ids = [r["id"] for r in rows]
    y = np.array([base.ACTION_CLASSES.index(r["action"]) for r in rows])
    nhom = {r["id"]: r.get("nhom", "") for r in rows}
    reviewed = {r["id"]: r for r in read_csv(LABELS_REVIEWED)} if LABELS_REVIEWED.exists() else {}
    nets = load_nets(device)

    variants = {"goc_cache": BASE_CACHE}
    variants.update({n: OUT / n / "poses" for n in CONFIGS if (OUT / n / "poses").is_dir()})
    variants.update({f: s for f, s in FILLS.items() if (OUT / s / "poses").is_dir()})

    # Kiem tra tai lap: dac trung tu cache goc phai trung dac trung test da dung trong cac thi nghiem truoc.
    with np.load(TEST_FEATURES) as z:
        ref = dict(zip(z["ids"].tolist(), z["raw"]))
    per_video, summary = [], []
    base_ok = None
    for name, src in variants.items():
        feats, pm, missing = [], np.zeros(4), []
        for sid in ids:
            if name in FILLS:
                xy0, cf0, st0 = load_pose(BASE_CACHE, sid)
                p = OUT / src / "poses" / f"{sid}.npz"
                if not p.exists():
                    missing.append(sid)
                    continue
                xy1, cf1, st1 = load_pose(OUT / src / "poses", sid)
                fill = (st0 != 3) & (st1 == 3)
                xy, st = xy0.copy(), st0.copy()
                xy[fill], st[fill] = xy1[fill], 3
            else:
                p = Path(src) / f"{sid}.npz"
                if not p.exists():
                    missing.append(sid)
                    continue
                xy, _, st = load_pose(Path(src), sid)
            xs = xy[:, SWAP]
            f = base.resample(xs)
            if name == "goc_cache":
                assert np.allclose(f, ref[sid], atol=1e-5), f"Dac trung khong tai lap duoc: {sid}"
            feats.append(f)
            pm += pose_metrics(sid, xs, st)
        if missing:
            print(f"[{name}] bo qua: thieu {len(missing)} video (chua chay infer)")
            continue
        pred = predict(nets, feats, device)
        acc_video = np.mean([pred[k] == y for k in pred], axis=0)  # ti le dung qua 6 luot
        ok = acc_video >= 0.5
        if name == "goc_cache":
            base_ok = ok
        rec = dict(
            bien_the=name,
            ti_le_frame_khop=round(pm[1] / pm[0], 4),
            pck02=round(pm[3] / pm[2], 4),
            acc_luot_tb=round(float(np.mean(acc_video)) * 100, 2),
            acc_video_da_so=round(float(ok.mean()) * 100, 2),
        )
        for k in pred:
            rec[f"acc_{k}"] = round(float(np.mean(pred[k] == y)) * 100, 2)
        if base_ok is not None:
            rec["sua_duoc"] = int((~base_ok & ok).sum())
            rec["lam_hong"] = int((base_ok & ~ok).sum())
            for g in sorted({v for v in nhom.values() if v}):
                sel = np.array([nhom[i] == g for i in ids])
                if sel.any():
                    rec[f"{g}_dung"] = f"{int(ok[sel].sum())}/{int(sel.sum())}"
            real = np.array(
                [
                    nhom[i] == "A"
                    and reviewed.get(i, {}).get("loi_chinh", "") not in ("", "khong_loi_ro")
                    for i in ids
                ]
            )
            if real.any():
                rec["A_loi_khung_xuong_that_dung"] = f"{int(ok[real].sum())}/{int(real.sum())}"
        summary.append(rec)
        for i, sid in enumerate(ids):
            per_video.append(
                dict(
                    bien_the=name,
                    id=sid,
                    nhom=nhom[sid],
                    action=rows[i]["action"],
                    loi_chinh=reviewed.get(sid, {}).get("loi_chinh", ""),
                    ti_le_dung_6_luot=round(float(acc_video[i]), 3),
                    thay_doi=(
                        "sua_duoc"
                        if base_ok is not None and ok[i] and not base_ok[i]
                        else "lam_hong"
                        if base_ok is not None and base_ok[i] and not ok[i]
                        else ""
                    ),
                )
            )
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "all_test" if args.all_test else "91video"
    for fname, data in ((f"summary_{tag}.csv", summary), (f"per_video_{tag}.csv", per_video)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with (OUT / fname).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(data)
    print("Da luu", OUT / f"summary_{tag}.csv")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["infer", "evaluate"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument(
        "--all-test", action="store_true", help="Chay 1068 video test thay vi 91 video phieu"
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if args.step == "infer":
        if args.device != "cpu" and not torch.cuda.is_available():
            sys.exit("Khong thay CUDA; dung --device cpu (rat cham)")
        infer(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
