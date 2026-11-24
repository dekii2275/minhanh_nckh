"""Gan nhan loai loi khung xuong theo video (buoc 1) - Penn Action official test.

Mot cua so OpenCV: ben trai la video co ve khung xuong annotation (xanh la) va
YOLO (cam) da luu cua pipeline (crop bbox GT, fixed swap); ben phai la bang tick
loai loi. Khong chay lai YOLO.
"""

import argparse
import base64
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
ATTR = ROOT / "outputs/error_attribution_20261124"
QUEUE = ATTR / "phieu_gan_nhan_loi_buoc1.csv"
OUT = ATTR / "nhan_loi_buoc1.csv"
POSES = ROOT / "outputs/confidence_confirmation_penn_20261105/test_pose/poses"
LABELS = ROOT / "data/raw/Penn_Action/labels"
FRAMES = ROOT / "data/raw/Penn_Action/frames"
MANIFEST = ROOT / "report/02112026_nckh/penn_audit_artifacts/video_manifest.csv"
PRED_BILSTM = ROOT / "outputs/unified_retrain_20261115/predictions.csv"
PRED_CASCADE = ROOT / "outputs/cascade_three_inputs_20261123/predictions.csv"

SWAP = np.arange(12) ^ 1  # fixed swap trai/phai, giong pipeline nghien cuu
EDGES = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
    (6, 8),
    (8, 10),
    (7, 9),
    (9, 11),
]
SHORT = [
    "L_sh",
    "R_sh",
    "L_el",
    "R_el",
    "L_wr",
    "R_wr",
    "L_hip",
    "R_hip",
    "L_kn",
    "R_kn",
    "L_an",
    "R_an",
]
STATUS = ("invalid_gt_bbox", "no_person", "no_iou_match", "matched")

ERRORS = [
    ("mat_khop", "Mat khop / conf thap"),
    ("nhay_khop", "Khop nhay dot ngot"),
    ("dao_trai_phai", "Dao trai - phai"),
    ("nham_nguoi", "Nham / doi nguoi"),
    ("che_khuat", "Che khuat (dung cu)"),
    ("tu_the_kho", "Tu the kho / lech ca cum"),
    ("rung_nhe", "Rung nhe lien tuc"),
    ("khong_loi_ro", "Khung xuong on, khong loi ro"),
    ("gt_lech", "Annotation/bbox GT lech"),
    ("cat_khung", "Nguoi bi cat khoi khung"),
]
ERR_KEYS = "1234567890"
JOINT_GROUPS = [
    ("vai", "Vai"),
    ("khuyu", "Khuyu"),
    ("co_tay", "Co tay"),
    ("hong", "Hong"),
    ("goi", "Goi"),
    ("co_chan", "Co chan"),
    ("toan_than", "Toan than"),
]
FIELDS = [
    "id",
    "nhom",
    "action",
    "loi_chinh",
    "loi_phu",
    "frame_bat_dau",
    "frame_ket_thuc",
    "khop_bi_anh_huong",
    "muc_do_1_3",
    "ghi_chu",
    "da_xong",
    "thoi_gian_utc",
]

HEADER, VIDEO_H, TIMELINE, FOOTER, PANEL = 84, 540, 44, 40, 340
SPEEDS = [0.25, 0.5, 1.0, 2.0]
KEY_LEFT, KEY_RIGHT = {2424832, 65361, 81}, {2555904, 65363, 83}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def text(img, s, x, y, color=(235, 235, 235), scale=0.45, thick=1, bg=False):
    # Ve mot lan (vien den kieu cu bi lech chu tren OpenCV 5); bg=True them nen den de doc tren video.
    if bg:
        (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def load_predictions():
    """{id: {'YOLO-BiL': [names], 'YOLO-Cas': ..., 'GT-BiL': ..., 'GT-Cas': ...}} tren test."""
    names = {}
    for r in read_csv(MANIFEST):
        names[r["id"]] = r["action"]
    target_name = {}
    out = {}
    sources = [
        (PRED_BILSTM, {"raw": "YOLO-BiL", "gt_bilstm": "GT-BiL"}),
        (PRED_CASCADE, {"raw": "YOLO-Cas", "gt_cascade": "GT-Cas"}),
    ]
    rows_all = []
    for path, methods in sources:
        if not path.is_file():
            print("Khong thay file du doan:", path)
            continue
        for r in read_csv(path):
            if r["split"] != "test" or r["method"] not in methods:
                continue
            target_name[int(r["target"])] = names.get(r["id"], "?")
            rows_all.append((methods[r["method"]], r))
    for tag, r in sorted(rows_all, key=lambda t: int(t[1]["seed"])):
        out.setdefault(r["id"], {}).setdefault(tag, []).append(int(r["predicted"]))
    return {
        sid: {tag: [target_name.get(p, str(p)) for p in ps] for tag, ps in d.items()}
        for sid, d in out.items()
    }


class Video:
    def __init__(self, row):
        self.id = row["id"]
        self.row = row
        mat = loadmat(LABELS / f"{self.id}.mat")
        x, y, vis = mat["x"].astype(float), mat["y"].astype(float), mat["visibility"].astype(bool)
        if x.shape[1] != 13 and x.shape[0] == 13:
            x, y, vis = x.T, y.T, vis.T
        self.gt = np.stack([x, y], -1)  # (n,13,2), joint 0 = head
        self.gt_vis = vis & (self.gt > 0).all(-1)
        self.action = str(mat["action"].flat[0]) if "action" in mat else row.get("action", "")
        bbox = np.full((len(self.gt), 4), np.nan)
        if "bbox" in mat:
            b = mat["bbox"]
            k = min(len(b), len(bbox))
            bbox[:k] = b[:k]
        self.bbox = bbox
        with np.load(POSES / f"{self.id}.npz") as d:
            self.xy = d["xy"][:, SWAP].astype(float)
            self.conf = d["conf"][:, SWAP].astype(float)
            self.status = d["status"].astype(int)
        self.files = sorted(
            p for p in (FRAMES / self.id).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.n = min(len(self.files), len(self.gt), len(self.xy))
        if self.n == 0:
            raise ValueError(f"Video {self.id} khong co frame")
        if not (len(self.files) == len(self.gt) == len(self.xy)):
            print(
                f"Canh bao {self.id}: {len(self.files)} anh, {len(self.gt)} GT, {len(self.xy)} pose -> xem {self.n} frame chung"
            )
        self.cache = {}
        # sai so tung frame de to mau timeline
        g = self.gt[: self.n, 1:13]
        p = self.xy[: self.n]
        ok = (
            self.gt_vis[: self.n, 1:13]
            & (self.status[: self.n, None] == 3)
            & (p > 0).all(-1)
            & np.isfinite(p).all(-1)
        )
        err = np.linalg.norm(p - g, axis=-1)
        wh = self.bbox[: self.n, 2:] - self.bbox[: self.n, :2]
        diag = np.linalg.norm(wh, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.err_px = np.where(
                ok.any(1), np.nansum(np.where(ok, err, 0), 1) / ok.sum(1), np.nan
            )
            self.err_norm = self.err_px / diag
        self.ok = ok
        self.has_gt = self.gt_vis[: self.n, 1:13].any(1)
        h, w = self.frame(0).shape[:2]
        self.scale = VIDEO_H / h
        self.w = int(round(w * self.scale))

    def frame(self, t):
        if t not in self.cache:
            if len(self.cache) > 600:
                self.cache.clear()
            img = cv2.imdecode(np.fromfile(str(self.files[t]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Khong doc duoc {self.files[t]}")
            self.cache[t] = img
        return self.cache[t]


def draw_gt(img, v, t, s, show_names):
    pts = v.gt[t] * s
    vis = v.gt_vis[t]
    body = pts[1:13]
    bv = vis[1:13]
    for a, b in EDGES:
        if bv[a] and bv[b]:
            cv2.line(
                img,
                tuple(np.rint(body[a]).astype(int)),
                tuple(np.rint(body[b]).astype(int)),
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )
    for k in range(13):
        if vis[k]:
            cv2.circle(img, tuple(np.rint(pts[k]).astype(int)), 4, (80, 255, 200), -1, cv2.LINE_AA)


def draw_yolo(img, v, t, s, show_names):
    if v.status[t] != 3:
        return
    pts = v.xy[t] * s
    conf = v.conf[t]
    valid = (v.xy[t] > 0).all(1) & np.isfinite(v.xy[t]).all(1)
    for a, b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(
                img,
                tuple(np.rint(pts[a]).astype(int)),
                tuple(np.rint(pts[b]).astype(int)),
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
    for k in np.flatnonzero(valid):
        p = tuple(np.rint(pts[k]).astype(int))
        if conf[k] >= 0.5:
            cv2.circle(img, p, 4, (0, 160, 255), -1, cv2.LINE_AA)
        else:  # conf thap: vong rong, do neu rat thap
            cv2.circle(img, p, 6, (0, 0, 255) if conf[k] < 0.2 else (0, 200, 255), 2, cv2.LINE_AA)
        if show_names:
            text(img, f"{SHORT[k]} {conf[k]:.2f}", p[0] + 6, p[1] - 5, (0, 200, 255), 0.38, bg=True)


def draw_err_lines(img, v, t, s):
    for k in np.flatnonzero(v.ok[t]):
        a = tuple(np.rint(v.gt[t, k + 1] * s).astype(int))
        b = tuple(np.rint(v.xy[t, k] * s).astype(int))
        cv2.line(img, a, b, (255, 0, 255), 1, cv2.LINE_AA)


class App:
    def __init__(self, queue, out_path, preds, start):
        self.queue = queue
        self.out_path = out_path
        self.preds = preds
        self.records = {}
        if out_path.is_file():
            for r in read_csv(out_path):
                self.records[r["id"]] = r
        self.i = start
        self.show = dict(gt=True, yolo=True, err=False, names=False, bbox=False, side=False)
        self.clicks = []
        self.drag = False
        self.feedback = ""
        self.window = "Gan nhan loi khung xuong - buoc 1"

    # ---------- trang thai nhan cua video hien tai ----------
    def load_state(self):
        r = self.records.get(self.v.id, {})

        def split(s):
            return [x for x in (s or "").split(";") if x]

        codes = split(r.get("loi_chinh")) + [
            c for c in split(r.get("loi_phu")) if c not in split(r.get("loi_chinh"))
        ]
        self.state = dict(
            codes=codes,
            main=(split(r.get("loi_chinh")) or [None])[0],
            joints=split(r.get("khop_bi_anh_huong")),
            start=r.get("frame_bat_dau") or "",
            end=r.get("frame_ket_thuc") or "",
            level=r.get("muc_do_1_3") or "",
            note=r.get("ghi_chu") or "",
            done=r.get("da_xong") == "1",
        )

    def empty(self):
        s = self.state
        return not (s["codes"] or s["joints"] or s["start"] or s["end"] or s["level"] or s["note"])

    def store(self, done=None):
        s = self.state
        if self.empty() and not s["done"]:
            return
        if done is not None:
            s["done"] = done
        main = s["main"] if s["main"] in s["codes"] else (s["codes"][0] if s["codes"] else "")
        self.records[self.v.id] = {
            "id": self.v.id,
            "nhom": self.v.row.get("nhom", ""),
            "action": self.v.action,
            "loi_chinh": main,
            "loi_phu": ";".join(c for c in s["codes"] if c != main),
            "frame_bat_dau": s["start"],
            "frame_ket_thuc": s["end"],
            "khop_bi_anh_huong": ";".join(s["joints"]),
            "muc_do_1_3": s["level"],
            "ghi_chu": s["note"],
            "da_xong": "1" if s["done"] else "0",
            "thoi_gian_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        order = {r["id"]: k for k, r in enumerate(self.queue)}
        tmp = self.out_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, FIELDS)
            wr.writeheader()
            for r in sorted(self.records.values(), key=lambda r: order.get(r["id"], 10**6)):
                wr.writerow({k: r.get(k, "") for k in FIELDS})
        os.replace(tmp, self.out_path)

    # ---------- bo cuc panel ----------
    def layout(self):
        x0 = self.video_w + 12
        w = PANEL - 24
        L = {}
        y = 44
        for k, (code, _) in enumerate(ERRORS):
            L[("err", code)] = (x0, y, x0 + w, y + 26)
            y += 30
        y += 26
        bw = (w - 16) // 3
        for k in range(3):
            L[("level", str(k + 1))] = (x0 + k * (bw + 8), y, x0 + k * (bw + 8) + bw, y + 26)
        y += 54
        cw = (w - 8) // 2
        for k, (code, _) in enumerate(JOINT_GROUPS):
            c, r = k % 2, k // 2
            L[("joint", code)] = (
                x0 + c * (cw + 8),
                y + r * 30,
                x0 + c * (cw + 8) + cw,
                y + r * 30 + 26,
            )
        y += 4 * 30 + 6
        L[("start", "")] = (x0, y, x0 + cw, y + 26)
        L[("end", "")] = (x0 + cw + 8, y, x0 + w, y + 26)
        y += 32
        L[("note", "")] = (x0, y, x0 + w, y + 26)
        y += 40
        L[("prev", "")] = (x0, y, x0 + cw, y + 32)
        L[("next", "")] = (x0 + cw + 8, y, x0 + w, y + 32)
        return L

    def button(self, img, rect, label, on=False, color=(0, 200, 120)):
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), color if on else (70, 70, 70), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 1)
        text(img, label, x1 + 7, y1 + 18, (255, 255, 255), 0.42)

    # ---------- ve toan bo canvas ----------
    def render(self):
        v, t, s = self.v, self.t, self.v.scale
        base = cv2.resize(v.frame(t), (v.w, VIDEO_H), interpolation=cv2.INTER_LINEAR)
        if self.show["side"]:
            left, right = base.copy(), base.copy()
            if self.show["gt"]:
                draw_gt(left, v, t, s, self.show["names"])
            if self.show["yolo"]:
                draw_yolo(right, v, t, s, self.show["names"])
            view = np.hstack([left, right])
        else:
            view = base.copy()
            if self.show["gt"]:
                draw_gt(view, v, t, s, self.show["names"])
            if self.show["yolo"]:
                draw_yolo(view, v, t, s, self.show["names"])
            if self.show["err"]:
                draw_err_lines(view, v, t, s)
        if self.show["bbox"] and np.isfinite(v.bbox[t]).all():
            b = np.rint(v.bbox[t] * s).astype(int)
            for off in [0, v.w] if self.show["side"] else [0]:
                cv2.rectangle(view, (b[0] + off, b[1]), (b[2] + off, b[3]), (200, 200, 200), 1)
        self.video_w = view.shape[1]
        H = HEADER + VIDEO_H + TIMELINE + FOOTER
        canvas = np.zeros((max(H, 700), self.video_w + PANEL, 3), np.uint8)
        canvas[HEADER : HEADER + VIDEO_H, : self.video_w] = view
        if self.show["side"]:
            text(canvas, "ANNOTATION", 8, HEADER + 18, (60, 220, 60), bg=True)
            text(canvas, "YOLO", v.w + 8, HEADER + 18, (0, 160, 255), bg=True)

        # header
        done = sum(1 for r in self.queue if self.records.get(r["id"], {}).get("da_xong") == "1")
        spd = SPEEDS[self.speed]
        text(
            canvas,
            f"[{self.i + 1}/{len(self.queue)}] id {v.id} | {v.action} | nhom {v.row.get('nhom', '')} | "
            f"frame {t + 1}/{v.n} | {'PAUSE' if self.paused else 'PLAY'} x{spd:g} | da xong {done}/{len(self.queue)}",
            6,
            18,
            (255, 255, 255),
            0.48,
        )
        p = self.preds.get(v.id, {})

        def fmt(tag):
            return ",".join(n[:12] for n in p.get(tag, [])) or "-"

        text(
            canvas,
            f"Du doan YOLO  BiLSTM: {fmt('YOLO-BiL')}  | Cascade: {fmt('YOLO-Cas')}",
            6,
            40,
            (0, 190, 255),
            0.42,
        )
        text(
            canvas,
            f"Du doan GT    BiLSTM: {fmt('GT-BiL')}  | Cascade: {fmt('GT-Cas')}",
            6,
            58,
            (90, 230, 90),
            0.42,
        )
        st = STATUS[v.status[t]] if 0 <= v.status[t] < 4 else str(v.status[t])
        cm = np.nanmean(v.conf[t]) if v.status[t] == 3 else float("nan")
        e = f"{v.err_px[t]:.1f}px ({v.err_norm[t]:.2f} bbox)" if np.isfinite(v.err_px[t]) else "N/A"
        text(
            canvas,
            f"YOLO: {st} | conf TB {cm:.2f} | sai so TB voi GT: {e}",
            6,
            77,
            (230, 230, 230),
            0.42,
        )

        # timeline
        ty = HEADER + VIDEO_H + 8
        tw = self.video_w - 16
        th = 22
        self.tl = (8, ty, 8 + tw, ty + th)
        for k in range(v.n):
            x1 = 8 + int(k * tw / v.n)
            x2 = max(x1 + 1, 8 + int((k + 1) * tw / v.n))
            if not v.has_gt[k]:
                c = (90, 50, 20)
            elif v.status[k] != 3:
                c = (110, 110, 110)
            elif not np.isfinite(v.err_norm[k]):
                c = (60, 60, 60)
            elif v.err_norm[k] < 0.1:
                c = (60, 170, 60)
            elif v.err_norm[k] < 0.2:
                c = (0, 200, 230)
            else:
                c = (40, 40, 230)
            cv2.rectangle(canvas, (x1, ty), (x2, ty + th), c, -1)
        st_, en_ = self.state["start"], self.state["end"]
        if st_ or en_:
            a = int(st_ or 1) - 1
            b = int(en_ or v.n) - 1
            cv2.rectangle(
                canvas,
                (8 + int(a * tw / v.n), ty - 6),
                (8 + int((b + 1) * tw / v.n), ty - 2),
                (255, 255, 0),
                -1,
            )
        cx = 8 + int((t + 0.5) * tw / v.n)
        cv2.line(canvas, (cx, ty - 4), (cx, ty + th + 4), (255, 255, 255), 2)
        text(
            canvas,
            "timeline: xanh <0.1 | vang <0.2 | do >=0.2 bbox | xam = YOLO mat nguoi | nau = khong co GT  (click/keo de tua)",
            8,
            ty + th + 14,
            (200, 200, 200),
            0.36,
        )
        text(
            canvas,
            "Space phat/dung | A/D hoac <-/-> 1 frame | J/L 10 frame | R ve dau | +/- toc do | G/Y GT/YOLO | E duong sai so | "
            "C ten khop | B bbox | V song song",
            8,
            H - 22,
            (170, 170, 170),
            0.36,
        )
        text(
            canvas,
            "1-9,0 tick loi | [ ] doan loi | T ghi chu | Enter/N luu & tiep | P video truoc | Q luu & thoat",
            8,
            H - 6,
            (170, 170, 170),
            0.36,
        )

        # panel
        px = self.video_w
        cv2.rectangle(canvas, (px, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (35, 35, 35), -1)
        self.rects = self.layout()
        S = self.state
        text(
            canvas,
            "LOAI LOI  (chuot trai: tick | phai: loi chinh)",
            px + 12,
            22,
            (0, 210, 255),
            0.42,
        )
        for k, (code, label) in enumerate(ERRORS):
            on = code in S["codes"]
            main = on and (
                S["main"] == code
                or (S["main"] not in S["codes"] and S["codes"] and S["codes"][0] == code)
            )
            self.button(
                canvas,
                self.rects[("err", code)],
                f"{ERR_KEYS[k]}. {'[x]' if on else '[ ]'} {label}{'  *CHINH' if main else ''}",
                on,
                (0, 120, 200) if main else (0, 150, 90),
            )
        r = self.rects[("level", "1")]
        text(canvas, "MUC DO (1 nhe - 3 hong gan het video)", px + 12, r[1] - 8, (0, 210, 255), 0.4)
        for k in "123":
            self.button(canvas, self.rects[("level", k)], f"  {k}", S["level"] == k)
        r = self.rects[("joint", "vai")]
        text(canvas, "KHOP BI ANH HUONG", px + 12, r[1] - 8, (0, 210, 255), 0.4)
        for code, label in JOINT_GROUPS:
            self.button(
                canvas,
                self.rects[("joint", code)],
                f"{'[x]' if code in S['joints'] else '[ ]'} {label}",
                code in S["joints"],
            )
        self.button(
            canvas,
            self.rects[("start", "")],
            f"[ Bat dau: {S['start'] or '-'}",
            bool(S["start"]),
            (150, 130, 0),
        )
        self.button(
            canvas,
            self.rects[("end", "")],
            f"] Ket thuc: {S['end'] or '-'}",
            bool(S["end"]),
            (150, 130, 0),
        )
        note = S["note"].replace("\n", " ")
        note_ascii = note.encode("ascii", "replace").decode()
        self.button(
            canvas,
            self.rects[("note", "")],
            f"T. Ghi chu: {note_ascii[:26] + ('..' if len(note) > 26 else '') if note else '(trong)'}",
            bool(note),
            (120, 80, 160),
        )
        self.button(canvas, self.rects[("prev", "")], "<< Truoc (P)")
        self.button(canvas, self.rects[("next", "")], "Luu & tiep >>", True, (0, 120, 200))
        y = self.rects[("next", "")][3] + 22
        if S["done"]:
            text(canvas, "Video nay DA XONG", px + 12, y, (80, 230, 120), 0.42)
        if self.feedback:
            text(canvas, self.feedback[:46], px + 12, y + 20, (80, 200, 255), 0.4)
        return canvas

    # ---------- su kien ----------
    def on_mouse(self, event, x, y, flags, _):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            x1, y1, x2, y2 = getattr(self, "tl", (0, 0, 0, 0))
            if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 - 8 <= y <= y2 + 8:
                self.drag = True
                self.seek_x(x)
                return
            for key, (a, b, c, d) in getattr(self, "rects", {}).items():
                if a <= x <= c and b <= y <= d:
                    self.clicks.append((key, event == cv2.EVENT_RBUTTONDOWN))
                    return
        elif event == cv2.EVENT_MOUSEMOVE and self.drag and flags & cv2.EVENT_FLAG_LBUTTON:
            self.seek_x(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = False

    def seek_x(self, x):
        x1, _, x2, _ = self.tl
        self.t = int(np.clip((x - x1) / max(1, x2 - x1) * self.v.n, 0, self.v.n - 1))
        self.paused = True

    def toggle_code(self, code, make_main=False):
        S = self.state
        if make_main:
            if code not in S["codes"]:
                S["codes"].append(code)
            S["main"] = code
        elif code in S["codes"]:
            S["codes"].remove(code)
            if S["main"] == code:
                S["main"] = None
        else:
            S["codes"].append(code)
        S["done"] = False if not S["codes"] else S["done"]

    def ask_note(self):
        env = os.environ.copy()
        env["LABEL_VIDEO"] = f"{self.v.id} {self.v.action}"
        env["LABEL_NOTE_B64"] = base64.b64encode(self.state["note"].encode("utf-8")).decode()
        try:
            res = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-STA",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(Path(__file__).with_name("note_dialog.ps1")),
                ],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.feedback = "Khong mo duoc hop ghi chu"
            print(exc)
            return
        out = res.stdout.strip()
        if out.startswith("SAVE:"):
            self.state["note"] = base64.b64decode(out[5:]).decode("utf-8").strip()
            self.feedback = "Da cap nhat ghi chu"
            self.store()
        else:
            self.feedback = "Huy ghi chu"

    def handle(self, key, right=False):
        kind, val = key
        S = self.state
        if kind == "err":
            self.toggle_code(val, right)
        elif kind == "level":
            S["level"] = "" if S["level"] == val else val
        elif kind == "joint":
            S["joints"].remove(val) if val in S["joints"] else S["joints"].append(val)
        elif kind in ("start", "end"):
            S[kind] = "" if right else str(self.t + 1)
        elif kind == "note":
            self.ask_note()
        elif kind == "next":
            return self.go(+1, finish=True)
        elif kind == "prev":
            return self.go(-1)
        if kind not in ("note",):
            self.store()

    def go(self, step, finish=False):
        if finish:
            if not self.state["codes"]:
                self.feedback = "Chon it nhat 1 loai loi (8 = khong loi ro)"
                return
            self.store(done=True)
        else:
            self.store()
        j = self.i + step
        if not 0 <= j < len(self.queue):
            self.feedback = "Da het danh sach" if step > 0 else "Dang o video dau"
            return
        self.open(j)

    def open(self, i):
        self.i = i
        try:
            self.v = Video(self.queue[i])
        except (OSError, ValueError, KeyError) as exc:
            print(f"Loi mo video {self.queue[i]['id']}: {exc}")
            raise
        self.t, self.paused, self.speed, self.feedback = 0, False, 2, ""
        self.load_state()

    def run(self):
        self.open(self.i)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1400, 720)
        cv2.setMouseCallback(self.window, self.on_mouse)
        try:
            try:
                while True:
                    cv2.imshow(self.window, self.render())
                    delay = 30 if self.paused else max(1, int(1000 / (15 * SPEEDS[self.speed])))
                    k = cv2.waitKeyEx(delay)
                    while self.clicks:
                        key, right = self.clicks.pop(0)
                        self.handle(key, right)
                    if k == -1:
                        if not self.paused:
                            self.t = (self.t + 1) % self.v.n
                    elif k in KEY_LEFT:
                        self.t, self.paused = max(0, self.t - 1), True
                    elif k in KEY_RIGHT:
                        self.t, self.paused = min(self.v.n - 1, self.t + 1), True
                    else:
                        c = chr(k & 0xFF).lower() if (k & 0xFF) < 128 else ""
                        if c == "q" or (k & 0xFF) == 27:
                            self.store()
                            break
                        elif c == " ":
                            self.paused = not self.paused
                        elif c == "a":
                            self.t, self.paused = max(0, self.t - 1), True
                        elif c == "d":
                            self.t, self.paused = min(self.v.n - 1, self.t + 1), True
                        elif c == "j":
                            self.t, self.paused = max(0, self.t - 10), True
                        elif c == "l":
                            self.t, self.paused = min(self.v.n - 1, self.t + 10), True
                        elif c == "r":
                            self.t = 0
                        elif c in "+=":
                            self.speed = min(len(SPEEDS) - 1, self.speed + 1)
                        elif c in "-_":
                            self.speed = max(0, self.speed - 1)
                        elif c and c in ERR_KEYS:
                            self.handle(("err", ERRORS[ERR_KEYS.index(c)][0]))
                        elif c == "g":
                            self.show["gt"] = not self.show["gt"]
                        elif c == "y":
                            self.show["yolo"] = not self.show["yolo"]
                        elif c == "e":
                            self.show["err"] = not self.show["err"]
                        elif c == "c":
                            self.show["names"] = not self.show["names"]
                        elif c == "b":
                            self.show["bbox"] = not self.show["bbox"]
                        elif c == "v":
                            self.show["side"] = not self.show["side"]
                        elif c == "[":
                            self.handle(("start", ""))
                        elif c == "]":
                            self.handle(("end", ""))
                        elif c == "t":
                            self.handle(("note", ""))
                        elif c == "n" or (k & 0xFF) == 13:
                            self.handle(("next", ""))
                        elif c == "p":
                            self.handle(("prev", ""))
                    if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                        self.store()
                        break
            except KeyboardInterrupt:  # Ctrl+C trong terminal: luu roi thoat em
                self.store()
                print("Nhan Ctrl+C - da luu va thoat.")
        finally:
            cv2.destroyAllWindows()
            done = sum(1 for r in self.records.values() if r.get("da_xong") == "1")
            print(f"Da luu {self.out_path} | da xong {done}/{len(self.queue)} video")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, default=QUEUE, help="CSV danh sach video (cot id, nhom)")
    ap.add_argument(
        "--out", type=Path, default=OUT, help="CSV ket qua gan nhan (tu tiep tuc neu da co)"
    )
    ap.add_argument("--nhom", help="Chi xem mot nhom, vd A, B, D_doi_chung")
    ap.add_argument("--start", help="Mo tu video id nay, vd 1532")
    args = ap.parse_args()
    queue = read_csv(args.queue)
    if args.nhom:
        queue = [r for r in queue if r.get("nhom") == args.nhom]
    if not queue:
        sys.exit("Danh sach video rong.")
    app = App(queue, args.out, load_predictions(), 0)
    if args.start:
        ids = [r["id"] for r in queue]
        if args.start not in ids:
            sys.exit(f"Khong co video {args.start} trong danh sach")
        app.i = ids.index(args.start)
    else:  # mo video dau tien chua xong
        app.i = next(
            (k for k, r in enumerate(queue) if app.records.get(r["id"], {}).get("da_xong") != "1"),
            0,
        )
    print(f"{len(queue)} video | ket qua luu tai {args.out}")
    app.run()


if __name__ == "__main__":
    main()
