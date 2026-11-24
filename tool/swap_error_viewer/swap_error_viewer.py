"""Review Penn validation videos where fixed L/R swap increased pose error.

Shows the same frame as GT, GT-assisted YOLO identity and GT-assisted YOLO swap.
All numbers reproduce the previously saved paired-joint analysis; GT matching
and swap are diagnostics, not deployable target selection or ground truth.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tool/person_selection_viewer"))
from person_selection_viewer import OUT, draw_box, load_video, read_csv, text  # noqa: E402

sys.path.insert(0, str(ROOT / "tool/dataset_viewer"))
from dataset_viewer import choose  # noqa: E402

ANALYSIS = ROOT / "outputs/skeleton_gt_comparison_validation_20261103"
SWAP = np.arange(12) ^ 1
NAMES = ("S", "S", "E", "E", "W", "W", "H", "H", "K", "K", "A", "A")
# Cyan = left according to code; magenta = right according to code.
COLORS = ((255, 255, 0), (255, 0, 255))
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
MODES = (
    "identity_better",
    "swap_better",
    "no_match",
    "different_selection",
    "low_gt_visible",
    "all",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs():
    provenance = json.loads((ANALYSIS / "provenance.json").read_text(encoding="utf-8"))
    if sha(OUT / "frame_matches_iou_0_3.csv") != provenance["source_frame_csv_sha256"]:
        raise ValueError("Bang chon frame khac voi phan tich 11 video.")
    if sha(OUT / "provenance.json") != provenance["source_provenance_sha256"]:
        raise ValueError("Nguon inference khac voi phan tich 11 video.")
    records = read_csv(ANALYSIS / "per_video.csv")
    videos = {r["id"]: r for r in read_csv(OUT / "videos_iou_0_3.csv")}
    frames = defaultdict(list)
    for r in read_csv(OUT / "frame_matches_iou_0_3.csv"):
        frames[r["id"]].append(r)
    predictions = defaultdict(list)
    for r in read_csv(OUT / "predictions_iou_0_3.csv"):
        predictions[r["id"]].append(r)
    if len(records) != 253 or len(videos) != 253 or {r["id"] for r in records} != set(videos):
        raise ValueError("Danh sach video khong khop artifact validation.")
    return records, videos, frames, predictions


def exceptions(records):
    """The exact selection used in the report: swap > identity on common joints."""
    result = []
    for r in records:
        a, b = r["gt_assisted_paired_error_px"], r["gt_assisted_swap_paired_error_px"]
        if a and b and float(b) > float(a):
            result.append(r)
    return sorted(
        result,
        key=lambda r: (
            float(r["gt_assisted_swap_paired_error_px"]) - float(r["gt_assisted_paired_error_px"])
        ),
        reverse=True,
    )


def frame_metrics(seq, data, rows):
    result = []
    for t, row in enumerate(rows):
        lo, hi = map(int, data["offsets"][t : t + 2])
        gt = np.asarray(seq.points[t, 1:13], dtype=np.float64)
        vis = np.asarray(seq.visible[t, 1:13], dtype=bool)
        bbox = data["gt_bbox"][t]
        bbox_ok = bool(np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1])
        gt_ok = vis & np.isfinite(gt).all(axis=1) & (gt > 0).all(axis=1) & bbox_ok
        identity = np.full((12, 2), np.nan, np.float64)
        max_xy = identity.copy()
        cf = np.zeros(12, np.float64)
        max_cf = cf.copy()
        selected, maximum = int(row["gt_index"]), int(row["max_index"])
        if selected >= 0:
            identity = np.asarray(data["xy"][lo + selected, 5:17], dtype=np.float64)
            cf = np.asarray(data["conf"][lo + selected, 5:17], dtype=np.float64)
        if maximum >= 0:
            max_xy = np.asarray(data["xy"][lo + maximum, 5:17], dtype=np.float64)
            max_cf = np.asarray(data["conf"][lo + maximum, 5:17], dtype=np.float64)

        def valid(xy, confidence):
            return (
                np.isfinite(xy).all(axis=1)
                & (xy > 0).all(axis=1)
                & np.isfinite(confidence)
                & (confidence > 0)
            )

        common = (
            gt_ok & valid(identity, cf) & valid(identity[SWAP], cf[SWAP]) & valid(max_xy, max_cf)
        )
        error_i = np.linalg.norm(identity - gt, axis=1)
        error_s = np.linalg.norm(identity[SWAP] - gt, axis=1)
        n = int(common.sum())
        result.append(
            {
                "gt": gt,
                "gt_visible": vis,
                "identity": identity,
                "swap": identity[SWAP],
                "conf": cf,
                "swap_conf": cf[SWAP],
                "common": common,
                "gt_usable": int(gt_ok.sum()),
                "common_count": n,
                "identity_errors": error_i,
                "swap_errors": error_s,
                "identity_error": float(error_i[common].mean()) if n else None,
                "swap_error": float(error_s[common].mean()) if n else None,
                "missing": selected < 0,
                "different_selection": selected >= 0 and selected != maximum,
                "low_gt_visible": int(gt_ok.sum()) < 6,
                "low_yolo_conf": int((cf < 0.5).sum()) if selected >= 0 else None,
                "n_detections": int(hi - lo),
            }
        )
    return result


def validate_metrics(metrics, video):
    count = sum(m["common_count"] for m in metrics)
    if count != int(video["paired_joints"]):
        raise ValueError("So khop chung khong khop phan tich da luu.")
    if count:
        for key, saved in (
            ("identity_error", "gt_assisted_paired_error_px"),
            ("swap_error", "gt_assisted_swap_paired_error_px"),
        ):
            avg = sum(m[key] * m["common_count"] for m in metrics if m["common_count"]) / count
            if not np.isclose(avg, float(video[saved]), rtol=0, atol=1e-4):
                raise ValueError("Sai so tinh lai khong khop phan tich da luu.")


def event(mode, metric, tolerance):
    a, b = metric["identity_error"], metric["swap_error"]
    if mode == "all":
        return True
    if mode == "no_match":
        return metric["missing"]
    if mode == "different_selection":
        return metric["different_selection"]
    if mode == "low_gt_visible":
        return metric["low_gt_visible"]
    if a is None:
        return False
    if mode == "identity_better":
        return b - a > tolerance
    if mode == "swap_better":
        return a - b > tolerance
    return False


def draw_pose(image, points, confidence=None, gt_rings=None):
    h, w = image.shape[:2]
    valid = np.isfinite(points).all(axis=1) & (points > 0).all(axis=1)
    valid &= (points[:, 0] < w) & (points[:, 1] < h)
    if confidence is not None:
        valid &= np.isfinite(confidence) & (confidence > 0)
    if gt_rings is not None:
        gt, visible = gt_rings
        for j in range(12):
            if visible[j] and np.isfinite(gt[j]).all() and 0 < gt[j, 0] < w and 0 < gt[j, 1] < h:
                cv2.circle(
                    image, tuple(np.rint(gt[j]).astype(int)), 7, COLORS[j % 2], 1, cv2.LINE_AA
                )
    for a, b in EDGES:
        if valid[a] and valid[b]:
            color = COLORS[a % 2] if a % 2 == b % 2 else (185, 185, 185)
            cv2.line(
                image,
                tuple(np.rint(points[a]).astype(int)),
                tuple(np.rint(points[b]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    for j in np.flatnonzero(valid):
        point = tuple(np.rint(points[j]).astype(int))
        color = COLORS[j % 2]
        cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
        label = ("L" if j % 2 == 0 else "R") + NAMES[j]
        cv2.putText(
            image,
            label,
            (point[0] + 3, point[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (point[0] + 3, point[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def draw_timeline(canvas, metrics, current, tolerance, y):
    n = len(metrics)
    left, right = 7, canvas.shape[1] - 8
    for x in range(left, right):
        t = min(n - 1, int((x - left) * n / max(1, right - left)))
        a, b = metrics[t]["identity_error"], metrics[t]["swap_error"]
        color = (
            (100, 100, 100)
            if a is None
            else (
                (40, 150, 255)
                if b - a > tolerance
                else ((60, 210, 60) if a - b > tolerance else (220, 220, 220))
            )
        )
        cv2.line(canvas, (x, y), (x, y + 14), color, 1)
    x = left + int((current + 0.5) * (right - left) / n)
    cv2.line(canvas, (x, y - 4), (x, y + 18), (255, 255, 255), 2)
    text(
        canvas,
        "CAM: GIU TOT HON | XANH: SWAP TOT HON | XAM: KHONG DU DU LIEU | TRANG: GAN BANG",
        left,
        y + 32,
        scale=0.35,
    )


def render(seq, data, rows, metrics, video, index, mode, paused, overlay, tolerance):
    row, m = rows[index], metrics[index]
    raw = seq.frame(index, False)
    h, w = raw.shape[:2]
    gt_img, identity_img, swap_img = raw.copy(), raw.copy(), raw.copy()
    gt_draw = m["gt"].copy()
    gt_draw[~m["gt_visible"]] = np.nan
    draw_pose(gt_img, gt_draw)
    lo, hi = map(int, data["offsets"][index : index + 2])
    boxes = data["boxes"][lo:hi]
    selected = int(row["gt_index"])
    for image in (gt_img, identity_img, swap_img):
        gtbox = data["gt_bbox"][index]
        if np.isfinite(gtbox).all() and gtbox[2] > gtbox[0] and gtbox[3] > gtbox[1]:
            draw_box(image, gtbox, (50, 210, 50), "GT bbox", 1)
        for j, box in enumerate(boxes):
            draw_box(
                image,
                box,
                (0, 215, 255) if j == selected else (120, 120, 120),
                "GT-chon" if j == selected else "khac",
                2 if j == selected else 1,
            )
    rings = (m["gt"], m["gt_visible"]) if overlay else None
    draw_pose(identity_img, m["identity"], m["conf"], rings)
    draw_pose(swap_img, m["swap"], m["swap_conf"], rings)
    top = 132
    bottom = 50
    canvas = np.zeros((top + h + bottom, w * 3, 3), np.uint8)
    canvas[top : top + h, :w], canvas[top : top + h, w : 2 * w], canvas[top : top + h, 2 * w :] = (
        gt_img,
        identity_img,
        swap_img,
    )
    a, b = m["identity_error"], m["swap_error"]
    status = (
        "KHONG DU KHOP CHUNG"
        if a is None
        else (
            "GIU SAI SO THAP HON"
            if b - a > tolerance
            else ("SWAP SAI SO THAP HON" if a - b > tolerance else "CHENH LECH NHO")
        )
    )
    text(
        canvas,
        f"{seq.item['id']} | {seq.item['action']} | frame {index + 1}/{seq.count} | {'PAUSE' if paused else 'PLAY'}",
        6,
        17,
    )
    text(
        canvas,
        f"{status} | GT-hop le {m['gt_usable']}/12 | khop chung {m['common_count']}/12 | tim: {mode}",
        6,
        35,
        (0, 215, 255),
        0.38,
    )
    text(
        canvas,
        f"VIDEO tren khop chung: GIU {float(video['gt_assisted_paired_error_px']):.2f}px | SWAP {float(video['gt_assisted_swap_paired_error_px']):.2f}px | {video['paired_joints']} khop",
        6,
        52,
        scale=0.38,
    )
    iou = f"{float(row['gt_iou']):.2f}" if row["gt_iou"] else "N/A"
    text(
        canvas,
        f"detections {m['n_detections']} | GT-chon IoU {iou} | GT-chon khac confidence {'CO' if m['different_selection'] else 'KHONG'} | YOLO conf<0.5: {m['low_yolo_conf'] if m['low_yolo_conf'] is not None else 'N/A'}/12",
        6,
        69,
        scale=0.36,
    )
    text(canvas, "GT (trai: xanh, phai: tim)", 6, 91, (50, 210, 50), 0.40)
    text(canvas, "YOLO GIU NGUYEN", w + 6, 91, (0, 215, 255), 0.40)
    text(canvas, "YOLO SWAP TRAI/PHAI", 2 * w + 6, 91, (0, 215, 255), 0.40)
    text(
        canvas,
        f"Frame: {a:.2f} px" if a is not None else "Frame: N/A (thieu)",
        w + 6,
        110,
        scale=0.40,
    )
    text(
        canvas,
        f"Frame: {b:.2f} px" if b is not None else "Frame: N/A (thieu)",
        2 * w + 6,
        110,
        scale=0.40,
    )
    text(canvas, "Vong rong: GT | Cham dac: YOLO | O: bat/tat vong GT", 6, 127, scale=0.35)
    draw_timeline(canvas, metrics, index, tolerance, top + h + 5)
    return canvas


def open_one(record, videos, frames, predictions, start, save, tolerance):
    sid = record["id"]
    seq, data, _ = load_video(sid, videos[sid], frames[sid], predictions[sid])
    try:
        metrics = frame_metrics(seq, data, frames[sid])
        validate_metrics(metrics, record)
        if not 1 <= start <= seq.count:
            raise ValueError("Frame nam ngoai video")
        counts = Counter(
            "no_match"
            if m["missing"]
            else (
                "no_common"
                if m["identity_error"] is None
                else (
                    "identity_better"
                    if event("identity_better", m, tolerance)
                    else ("swap_better" if event("swap_better", m, tolerance) else "close")
                )
            )
            for m in metrics
        )
        print(
            f"\n{sid} {record['action']}: GIU {float(record['gt_assisted_paired_error_px']):.2f}px; SWAP {float(record['gt_assisted_swap_paired_error_px']):.2f}px; {record['paired_joints']} khop chung"
        )
        print("Frame theo sai so (nguong loc %.1f px): %s" % (tolerance, dict(counts)))
        print(
            "L/R theo ten trong ma, khong khang dinh trai/phai giai phau; S/E/W/H/K/A = vai/khuyu/co tay/hong/goi/co chan."
        )
        print(
            "Space: phat/dung | A/D: frame | E: doi bo loc | N/P: su kien sau/truoc | O: vong GT | R: dau | Q: thoat"
        )
        if save:
            picture = render(
                seq, data, frames[sid], metrics, record, start - 1, "all", True, True, tolerance
            )
            if not cv2.imwrite(str(save), picture):
                raise OSError("Khong ghi duoc anh: " + str(save))
            print("Da luu", save)
            return
        index = start - 1
        paused = True
        overlay = True
        mode_index = 0
        window = "Kiem tra swap - 11 video Penn"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1600, 720)
        try:
            while True:
                mode = MODES[mode_index]
                cv2.imshow(
                    window,
                    render(
                        seq,
                        data,
                        frames[sid],
                        metrics,
                        record,
                        index,
                        mode,
                        paused,
                        overlay,
                        tolerance,
                    ),
                )
                key = cv2.waitKey(40 if paused else max(1, int(1000 / seq.fps))) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("a"):
                    index, paused = max(0, index - 1), True
                elif key == ord("d"):
                    index, paused = min(seq.count - 1, index + 1), True
                elif key == ord("r"):
                    index, paused = 0, True
                elif key == ord("o"):
                    overlay = not overlay
                elif key == ord("e"):
                    mode_index = (mode_index + 1) % len(MODES)
                    print("Bo loc:", MODES[mode_index], flush=True)
                elif key in (ord("n"), ord("p")):
                    step = 1 if key == ord("n") else -1
                    candidates = [
                        t
                        for t in range(index + step, seq.count if step == 1 else -1, step)
                        if event(mode, metrics[t], tolerance)
                    ]
                    if candidates:
                        index = candidates[0]
                    else:
                        print("Khong con frame thoa bo loc theo huong nay.", flush=True)
                    paused = True
                elif not paused:
                    index = (index + 1) % seq.count
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyWindow(window)
    finally:
        seq.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", help="ID validation; mac dinh hien menu 11 video")
    p.add_argument("--frame", type=int, default=1)
    p.add_argument("--list", action="store_true")
    p.add_argument(
        "--save-frame", type=Path, help="Luu mot frame thay vi mo cua so; can --sequence"
    )
    p.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Chenhlech px toi thieu de loc frame (khong anh huong 11 video)",
    )
    args = p.parse_args()
    if args.margin < 0:
        p.error("--margin phai khong am")
    if args.save_frame and not args.sequence:
        p.error("--save-frame can --sequence")
    records, videos, frames, predictions = read_inputs()
    cases = exceptions(records)
    if len(cases) != 11:
        raise ValueError(f"Du kien 11 video, artifact hien co {len(cases)}; kiem tra phan tich.")
    if args.list:
        for r in cases:
            print(
                f"{r['id']} {r['action']:16s} giu={float(r['gt_assisted_paired_error_px']):.2f}px swap={float(r['gt_assisted_swap_paired_error_px']):.2f}px khop_chung={r['paired_joints']}"
            )
        return
    if args.sequence:
        sid = args.sequence.zfill(4)
        options = [r for r in records if r["id"] == sid and r["gt_assisted_paired_error_px"]]
        if not options:
            p.error("ID khong co ket qua so sanh tren khop chung")
        open_one(options[0], videos, frames, predictions, args.frame, args.save_frame, args.margin)
        return
    try:
        while True:
            chosen = choose(
                "Chon video trong 11 truong hop",
                cases,
                lambda r: (
                    f"{r['id']} {r['action']} | giu={float(r['gt_assisted_paired_error_px']):.1f}px swap={float(r['gt_assisted_swap_paired_error_px']):.1f}px | {r['paired_joints']} khop chung"
                ),
            )
            open_one(chosen, videos, frames, predictions, args.frame, None, args.margin)
    except (KeyboardInterrupt, EOFError):
        print("\nDa thoat.")


if __name__ == "__main__":
    main()
