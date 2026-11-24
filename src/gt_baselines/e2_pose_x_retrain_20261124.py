"""E2 - Train lai tren pose YOLO11x va xu ly 3 loai loi con lai (Penn Action).

Cac buoc (chay tai thu muc goc du an, xem run_e1_e2_full.ps1):
  infer   --split trainval|test   Trich pose x_relaxed (YOLO11x, conf .1, margin .5, IoU .1, fallback toan anh).
  fix     --split trainval|test   Tao x_fix tu cache x_relaxed:
            (a) tu the nam/conf thap: frame mat nguoi hoac conf TB < 0.5 -> chay them crop xoay 90 do
                (2 chieu), chon ung vien co conf TB cao nhat;
            (b) annotation/bbox GT lech: video co < 50% frame khop nguoi -> chon nguoi bang tracking tren
                toan anh, khong dua vao bbox GT.
  features                        Tao dac trung 64x24 cho x_relaxed va x_fix (train/val/test).
  train                           Train tu dau BiLSTM + CascadeFormer, 3 seed, 50 epoch, protocol nhu
                                  unified_retrain_20261115; them bien the lr = hoan doi trai/phai ngau nhien
                                  khi train + trung binh logit goc/hoan doi khi suy luan (loi dao trai-phai).
  evaluate                        Bang ket qua so voi baseline raw (YOLO11n) cung protocol.
Khong dung toa do khop GT o bat ky buoc nao; bbox GT van duoc dung de crop nhu pipeline goc.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics_config"))
sys.path.insert(0, str(ROOT / "src/gt_baselines"))
import compare_person_selection as base  # noqa: E402
import cv2  # noqa: E402
import e1_redetect_pose_20261124 as e1  # noqa: E402
import train_gt_baselines as trainer  # noqa: E402
from train_gt_cascade_scratch import CascadeScratch  # noqa: E402

E1 = ROOT / "outputs/e1_redetect_20261124"
OUT = ROOT / "outputs/e2_pose_x_retrain_20261124"
SWAP = np.arange(12) ^ 1
POSE = {"x_relaxed": E1 / "x_relaxed/poses", "x_fix": OUT / "x_fix/poses"}
# ten phuong phap: (nguon pose, mang, hoan doi trai/phai)
METHODS = {
    "xr_bilstm": ("x_relaxed", "bilstm", False),
    "xr_cascade": ("x_relaxed", "cascade", False),
    "xf_bilstm": ("x_fix", "bilstm", False),
    "xf_cascade": ("x_fix", "cascade", False),
    "xf_lr_bilstm": ("x_fix", "bilstm", True),
    "xf_lr_cascade": ("x_fix", "cascade", True),
}
BASELINE = {
    "bilstm": ROOT / "outputs/unified_retrain_20261115/predictions.csv",
    "cascade": ROOT / "outputs/cascade_three_inputs_20261123/predictions.csv",
}
LOW_CONF, TRACK_BELOW = 0.5, 0.5


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def rows(split):
    allr = e1.read_csv(base.MANIFEST)
    if split == "test":
        r = [x for x in allr if x["official_split"] == "test"]
        assert len(r) == 1068
    elif split == "trainval":
        r = [x for x in allr if x["seed0_split"] in ("train", "validation")]
        assert len(r) == 1258
    else:
        raise ValueError(split)
    return sorted(r, key=lambda x: x["id"])


def frame(sid, t):
    return cv2.imdecode(
        np.fromfile(str(base.FRAMES / sid / f"{t + 1:06d}.jpg"), np.uint8), cv2.IMREAD_COLOR
    )


# ---------------------------------------------------------------- infer (x_relaxed cho train/val)
def infer(args):
    model_name, conf, margin, iou_min, fallback = e1.CONFIGS["x_relaxed"]
    out = POSE["x_relaxed"]
    out.mkdir(parents=True, exist_ok=True)
    model = e1.load_model(model_name)
    rs = rows(args.split)
    t0 = time.time()
    for k, row in enumerate(rs, 1):
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
                img = frame(sid, t)
                frames[t] = img
                rect = (
                    e1.crop_rect(bbox[t], img.shape[1], img.shape[0], margin)
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
            for r, (t, x1, y1) in zip(
                model.predict(source=imgs, imgsz=640, conf=conf, device=args.device, verbose=False),
                data,
            ):
                st[t], p, c = e1.pick(r, (x1, y1), bbox[t], iou_min)
                if p is not None:
                    xy[t], cf[t] = p, c
        if fallback:
            miss = [
                t for t in range(T) if st[t] != 3 and t < len(bbox) and np.isfinite(bbox[t]).all()
            ]
            for s in range(0, len(miss), args.batch):
                ts = miss[s : s + args.batch]
                for r, t in zip(
                    model.predict(
                        source=[frames[t] for t in ts],
                        imgsz=640,
                        conf=conf,
                        device=args.device,
                        verbose=False,
                    ),
                    ts,
                ):
                    code, p, c = e1.pick(r, (0, 0), bbox[t], iou_min)
                    if p is not None:
                        xy[t], cf[t], st[t] = p, c, 3
                    else:
                        st[t] = max(st[t], code)
        tmp = out / f"{sid}.tmp.npz"
        np.savez_compressed(tmp, xy=xy, conf=cf, status=st)
        os.replace(tmp, dest)
        print(
            f"[infer x_relaxed {args.split}] {k}/{len(rs)} {sid} matched={(st == 3).mean():.0%} {time.time() - t0:.0f}s",
            flush=True,
        )


# ---------------------------------------------------------------- fix
def unrotate(pts, mode, h, w):
    """Diem trong anh crop da xoay -> toa do crop goc (h, w la kich thuoc crop goc)."""
    x, y = pts[..., 0], pts[..., 1]
    if mode == cv2.ROTATE_90_CLOCKWISE:
        return np.stack([y, h - 1 - x], -1)
    return np.stack([w - 1 - y, x], -1)  # ROTATE_90_COUNTERCLOCKWISE


def candidates(result, mode, h, w, offset):
    """Tra ve list (box_xyxy, kp12, conf12) da doi ve toa do anh goc."""
    out = []
    if result.keypoints is None or not len(result.boxes):
        return out
    boxes = result.boxes.xyxy.cpu().numpy()
    kps = result.keypoints.xy.cpu().numpy()
    kc = result.keypoints.conf.cpu().numpy()
    for b, p, c in zip(boxes, kps, kc):
        p = p[5:17].copy()
        c = c[5:17].copy()
        valid = np.isfinite(p).all(1) & (p > 0).all(1)
        if mode is not None:
            corners = unrotate(
                np.array([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]]), mode, h, w
            )
            b = np.array([*corners.min(0), *corners.max(0)])
            p = unrotate(p, mode, h, w)
        p = p + np.array(offset)
        p[~valid] = np.nan
        out.append((b + np.array([*offset, *offset]), p.astype(np.float32), c.astype(np.float32)))
    return out


def best_match(cands, gt_box, iou_min):
    if not cands:
        return None
    iou = base.ious(np.stack([c[0] for c in cands]), gt_box)
    if not np.isfinite(iou).all() or iou.max() < iou_min:
        return None
    return cands[int(np.argmax(iou))]


def track(model, sid, T, conf, device, batch):
    """Chon nguoi khong dung bbox GT: neo o frame co detection diem cao nhat, lan theo IoU hai chieu."""
    dets = []
    for s in range(0, T, batch):
        ts = list(range(s, min(T, s + batch)))
        for r in model.predict(
            source=[frame(sid, t) for t in ts], imgsz=640, conf=conf, device=device, verbose=False
        ):
            if r.keypoints is None or not len(r.boxes):
                dets.append(None)
                continue
            p = r.keypoints.xy.cpu().numpy()[:, 5:17].copy()
            c = r.keypoints.conf.cpu().numpy()[:, 5:17].copy()
            p[~((p > 0).all(-1))] = np.nan
            dets.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), p, c))
    xy = np.full((T, 12, 2), np.nan, np.float32)
    cf = np.zeros((T, 12), np.float32)
    st = np.full(T, 1, np.uint8)
    scores = [d[1].max() if d is not None else -1 for d in dets]
    t0 = int(np.argmax(scores))
    if scores[t0] < 0:
        return xy, cf, st
    for direction in (range(t0, T), range(t0, -1, -1)):
        last = None
        for t in direction:
            d = dets[t]
            if d is None:
                continue
            if last is None:
                i = int(np.argmax(d[1]))
            else:
                iou = base.ious(d[0], last)
                if iou.max() < 0.2:
                    st[t] = 2
                    continue
                i = int(np.argmax(iou))
            last = d[0][i]
            xy[t], cf[t], st[t] = d[2][i], d[3][i], 3
    return xy, cf, st


def fix(args):
    model_name, conf, margin, iou_min, _ = e1.CONFIGS["x_relaxed"]
    out = POSE["x_fix"]
    out.mkdir(parents=True, exist_ok=True)
    model = e1.load_model(model_name)
    rs = rows(args.split)
    t0 = time.time()
    log = []
    for k, row in enumerate(rs, 1):
        sid = row["id"]
        dest = out / f"{sid}.npz"
        if dest.exists():
            continue
        with np.load(POSE["x_relaxed"] / f"{sid}.npz") as d:
            xy, cf, st = d["xy"].copy(), d["conf"].copy(), d["status"].copy()
        T = len(st)
        bbox = np.asarray(sio.loadmat(base.LABELS / f"{sid}.mat")["bbox"], float)
        rot = np.zeros(T, np.uint8)  # 0 giu nguyen, 1 xoay CW, 2 xoay CCW
        tracked = False
        if (st == 3).mean() < TRACK_BELOW:  # (b) bbox GT nghi lech -> tracking khong dung GT
            txy, tcf, tst = track(model, sid, T, conf, args.device, args.batch)
            if (tst == 3).mean() > (st == 3).mean():
                xy, cf, st, tracked = txy, tcf, tst, True
        if not tracked:  # (a) xoay crop cho frame mat nguoi / conf thap
            need = [
                t
                for t in range(T)
                if t < len(bbox)
                and np.isfinite(bbox[t]).all()
                and (st[t] != 3 or np.nanmean(cf[t]) < LOW_CONF)
            ]
            for s in range(0, len(need), args.batch):
                ts = need[s : s + args.batch]
                jobs = []
                for t in ts:
                    img = frame(sid, t)
                    rect = e1.crop_rect(bbox[t], img.shape[1], img.shape[0], margin)
                    if rect is None:
                        continue
                    x1, y1, x2, y2 = rect
                    crop = img[y1:y2, x1:x2]
                    for code, mode in (
                        (1, cv2.ROTATE_90_CLOCKWISE),
                        (2, cv2.ROTATE_90_COUNTERCLOCKWISE),
                    ):
                        jobs.append(
                            (
                                t,
                                code,
                                mode,
                                crop.shape[0],
                                crop.shape[1],
                                (x1, y1),
                                cv2.rotate(crop, mode),
                            )
                        )
                if not jobs:
                    continue
                res = model.predict(
                    source=[j[-1] for j in jobs],
                    imgsz=640,
                    conf=conf,
                    device=args.device,
                    verbose=False,
                )
                for r, (t, code, mode, h, w, off, _) in zip(res, jobs):
                    m = best_match(candidates(r, mode, h, w, off), bbox[t], iou_min)
                    if m is None:
                        continue
                    cur = float(np.nanmean(cf[t])) if st[t] == 3 else -1.0
                    if float(np.mean(m[2])) > cur:
                        xy[t], cf[t], st[t], rot[t] = m[1], m[2], 3, code
        tmp = out / f"{sid}.tmp.npz"
        np.savez_compressed(tmp, xy=xy, conf=cf, status=st, rotated=rot, tracked=tracked)
        os.replace(tmp, dest)
        log.append(
            dict(
                id=sid,
                split=args.split,
                tracked=int(tracked),
                rotated_frames=int((rot > 0).sum()),
                frames=T,
                matched=round(float((st == 3).mean()), 4),
            )
        )
        print(
            f"[fix {args.split}] {k}/{len(rs)} {sid} tracked={tracked} rotated={(rot > 0).sum()}/{T} {time.time() - t0:.0f}s",
            flush=True,
        )
    if log:
        p = OUT / f"fix_log_{args.split}.csv"
        new = not p.exists()
        with p.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, list(log[0]))
            new and w.writeheader()
            w.writerows(log)


# ---------------------------------------------------------------- features
def features(args):
    OUT.mkdir(parents=True, exist_ok=True)
    tv, te = rows("trainval"), rows("test")
    for name, src in POSE.items():
        dest = OUT / f"features_{name}.npz"
        if dest.exists():
            print("da co", dest)
            continue

        def feat(rs):
            return np.stack(
                [base.resample(np.load(src / f"{r['id']}.npz")["xy"][:, SWAP]) for r in rs]
            )

        x, tx = feat(tv), feat(te)
        assert (
            np.isfinite(x).all()
            and np.isfinite(tx).all()
            and x.shape == (1258, 64, 24)
            and tx.shape == (1068, 64, 24)
        )
        np.savez_compressed(
            dest,
            x=x,
            y=np.array([base.ACTION_CLASSES.index(r["action"]) for r in tv]),
            ids=np.array([r["id"] for r in tv]),
            split=np.array([r["seed0_split"] for r in tv]),
            test_x=tx,
            test_y=np.array([base.ACTION_CLASSES.index(r["action"]) for r in te]),
            test_ids=np.array([r["id"] for r in te]),
        )
        print("da tao", dest, flush=True)


# ---------------------------------------------------------------- train / evaluate
def make_net(kind):
    return (
        CascadeScratch()
        if kind == "cascade"
        else trainer.ActionBiLSTMAttentionClassifier(input_size=24)
    )


def swap_lr(x):
    s = x.shape
    return x.reshape(*s[:-1], 12, 2)[..., torch.from_numpy(SWAP).to(x.device), :].reshape(s)


def logits(net, x, lr):
    out = []
    for b in x.split(32):
        b = b.cuda()
        o = net(b)
        if lr:
            o = (o.softmax(-1) + net(swap_lr(b)).softmax(-1)) / 2
        out.append(o.cpu())
    return torch.cat(out)


def train(args):
    assert torch.cuda.is_available(), "Can CUDA"
    torch.set_num_threads(4)
    for name, (src, kind, lr) in METHODS.items():
        with np.load(OUT / f"features_{src}.npz") as z:
            tr = z["split"] == "train"
            va = z["split"] == "validation"
            train_x = torch.from_numpy(z["x"][tr].copy())
            train_y = torch.from_numpy(z["y"][tr].copy())
            val_x = torch.from_numpy(z["x"][va].copy())
            val_y = z["y"][va].copy()
        assert len(train_y) == 1005 and len(val_y) == 253
        for seed in range(3):
            dest = OUT / "trained" / name / f"seed{seed}"
            if (dest / "provenance.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            net = make_net(kind).cuda()
            loader = DataLoader(
                TensorDataset(train_x, train_y),
                batch_size=32,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0.00001)
            crit = nn.CrossEntropyLoss()
            best = -1
            history = []
            start = time.time()
            best_epoch = 0
            for epoch in range(1, 51):
                net.train()
                total = 0.0
                for x, y in loader:
                    x, y = x.cuda(), y.cuda()
                    if lr:  # hoan doi nhan trai/phai ngau nhien cho tung mau
                        m = torch.rand(len(x), device=x.device) < 0.5
                        x = torch.where(m[:, None, None], swap_lr(x), x)
                    opt.zero_grad()
                    loss = crit(net(x), y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    total += loss.item() * len(y)
                sched.step()
                net.eval()
                with torch.inference_mode():
                    p = logits(net, val_x, lr).argmax(-1).numpy()
                mt = base.metric(val_y, p)
                history.append({"epoch": epoch, "loss": total / len(train_y), **mt})
                if mt["accuracy_pct"] > best:
                    best = mt["accuracy_pct"]
                    best_epoch = epoch
                    torch.save(net.state_dict(), dest / "best.pt")
                if epoch % 10 == 0:
                    print(
                        name,
                        seed,
                        "epoch",
                        epoch,
                        "best val",
                        round(best, 2),
                        round(time.time() - start),
                        "s",
                        flush=True,
                    )
            write(dest / "history.json", history)
            write(
                dest / "provenance.json",
                {
                    "seed": seed,
                    "method": name,
                    "source": src,
                    "net": kind,
                    "lr_swap": lr,
                    "best_epoch": best_epoch,
                    "checkpoint_sha256": trainer.sha(dest / "best.pt"),
                    "feature_sha256": trainer.sha(OUT / f"features_{src}.npz"),
                    "seconds": time.time() - start,
                },
            )
            del net
            torch.cuda.empty_cache()


def baseline_preds():
    """Du doan raw (YOLO11n) da co: {net: {(split, seed): {id: pred}}}, cung protocol."""
    out = {}
    for kind, path in BASELINE.items():
        d = {}
        for r in e1.read_csv(path):
            if r["method"] == "raw" and r["scenario"] == "retrained":
                d.setdefault((r["split"], int(r["seed"])), {})[r["id"]] = int(r["predicted"])
        out[kind] = d
    return out


def evaluate(args):
    assert torch.cuda.is_available()
    base_p = baseline_preds()
    attr = {
        r["id"].zfill(4): r["cat"][0]
        for r in e1.read_csv(
            ROOT / "outputs/error_attribution_20261124/video_error_attribution.csv"
        )
    }
    rev = {
        r["id"]: r["loi_chinh"]
        for r in e1.read_csv(ROOT / "outputs/error_attribution_20261124/nhan_loi_buoc1_claude.csv")
    }
    preds, rec = {}, []
    for name, (src, kind, lr) in METHODS.items():
        with np.load(OUT / f"features_{src}.npz") as z:
            sets = {
                "validation": (
                    z["x"][z["split"] == "validation"],
                    z["y"][z["split"] == "validation"],
                    z["ids"][z["split"] == "validation"],
                ),
                "test": (z["test_x"], z["test_y"], z["test_ids"]),
            }
        for seed in range(3):
            ck = OUT / "trained" / name / f"seed{seed}" / "best.pt"
            net = make_net(kind).cuda()
            net.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
            net.eval()
            for split, (x, y, ids) in sets.items():
                with torch.inference_mode():
                    p = logits(net, torch.from_numpy(x.copy()), lr).argmax(-1).numpy()
                preds[(name, split, seed)] = dict(zip(ids.tolist(), p.tolist()))
                rec.append({"method": name, "split": split, "seed": seed, **base.metric(y, p)})
            del net
    # bang tong hop
    with np.load(OUT / "features_x_relaxed.npz") as z:
        test_ids, test_y = z["test_ids"].tolist(), z["test_y"]
        val_ids, val_y = (
            z["ids"][z["split"] == "validation"].tolist(),
            z["y"][z["split"] == "validation"],
        )
    rows_out = []

    def add(label, kind, getp):
        r = {"phuong_phap": label, "mang": kind}
        for split, ids, y in (("validation", val_ids, val_y), ("test", test_ids, test_y)):
            accs, f1s = [], []
            for s in range(3):
                p = np.array([getp(split, s)[i] for i in ids])
                m = base.metric(y, p)
                accs.append(m["accuracy_pct"])
                f1s.append(m["macro_f1_pct"])
            r[f"acc_{split}"] = f"{np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f}"
            r[f"f1_{split}"] = round(float(np.mean(f1s)), 2)
        ok = np.mean(
            [[getp("test", s)[i] == y for i, y in zip(test_ids, test_y)] for s in range(3)], 0
        )
        r["_ok"] = ok
        rows_out.append(r)

    for kind in ("bilstm", "cascade"):
        add("raw YOLO11n (cu)", kind, lambda sp, s, k=kind: base_p[k][(sp, s)])
        for name, (src, k2, lr) in METHODS.items():
            if k2 == kind:
                add(name, kind, lambda sp, s, n=name: preds[(n, sp, s)])
    # so sanh tung video voi raw cung mang: dung da so 3 seed; bootstrap chenh lech accuracy
    rng = np.random.default_rng(20261124)
    final = []
    for r in rows_out:
        ref = next(
            x for x in rows_out if x["mang"] == r["mang"] and x["phuong_phap"].startswith("raw")
        )
        a, b = ref["_ok"], r["_ok"]
        diff = b - a
        idx = rng.integers(0, len(diff), (10000, len(diff)))
        ci = np.quantile(100 * diff[idx].mean(1), [0.025, 0.975])
        r2 = {k: v for k, v in r.items() if k != "_ok"}
        r2["chenh_test_pp"] = round(100 * float(diff.mean()), 2)
        r2["ci95"] = f"[{ci[0]:.2f}; {ci[1]:.2f}]"
        r2["sua_duoc"] = int(((a < 0.5) & (b >= 0.5)).sum())
        r2["lam_hong"] = int(((a >= 0.5) & (b < 0.5)).sum())
        for g in "ABCD":
            sel = np.array([attr.get(i) == g for i in test_ids])
            r2[f"{g}_dung"] = f"{int((b[sel] >= 0.5).sum())}/{int(sel.sum())}"
        real = np.array(
            [attr.get(i) == "A" and rev.get(i, "khong_loi_ro") != "khong_loi_ro" for i in test_ids]
        )
        r2["A_loi_that_dung"] = f"{int((b[real] >= 0.5).sum())}/{int(real.sum())}"
        final.append(r2)
    with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(final[0]))
        w.writeheader()
        w.writerows(final)
    with (OUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "split", "seed", "id", "predicted"])
        for (n, sp, s), d in preds.items():
            w.writerows([n, sp, s, i, p] for i, p in d.items())
    write(OUT / "per_seed.json", rec)
    for r in final:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    print("Da luu", OUT / "summary.csv")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["infer", "fix", "features", "train", "evaluate"])
    ap.add_argument("--split", choices=["trainval", "test"], default="trainval")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    {"infer": infer, "fix": fix, "features": features, "train": train, "evaluate": evaluate}[
        a.step
    ](a)


if __name__ == "__main__":
    main()
