"""Inspect saved Penn validation detections and the two person-selection rules."""

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics_config"))
sys.path.insert(0, str(ROOT / "tool/dataset_viewer"))
from dataset_viewer import Sequence, choose  # noqa: E402

sys.path.insert(0, str(ROOT / "tool/pose_comparison_viewer"))
from pose_comparison_viewer import COCO_EDGES  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from data.action_dataset import ACTION_CLASSES, normalize_skeleton_sequence  # noqa: E402

OUT = ROOT / "outputs/person_selection_validation_20261102"
DATA = ROOT / "data/raw/Penn_Action"
MODES = ("problem", "changed", "no_detection", "no_match", "invalid_gt", "all")
# YOLO indices 5:17, ordered like the model's 12-joint input.
MODEL_EDGES = [
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


def make_model_input(data, rows):
    """Reconstruct exactly the GT-assisted input used by compare_person_selection.

    Detector zeros are preserved, as in the experiment; unmatched frames are
    NaN before normalization. No confidence threshold is applied here.
    """
    total = len(rows)
    coords = np.full((total, 12, 2), np.nan, dtype=np.float32)
    for t, row in enumerate(rows):
        selected = int(row["gt_index"])
        if selected >= 0:
            coords[t] = data["xy"][int(data["offsets"][t]) + selected, 5:17]
    normalized = normalize_skeleton_sequence(coords)
    positions = np.linspace(0, total - 1, 64)
    resampled = np.stack(
        [np.interp(positions, np.arange(total), normalized[:, j]) for j in range(24)], axis=-1
    ).astype(np.float32)
    return coords, normalized, positions, resampled


def contributing_tokens(index, positions):
    """Return 64-step input indices and source-frame interpolation weights."""
    result = []
    for token, position in enumerate(positions):
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        if lo == hi:
            weight = 1.0 if index == lo else 0.0
        elif index == lo:
            weight = hi - position
        elif index == hi:
            weight = position - lo
        else:
            weight = 0.0
        if weight > 1e-10:
            result.append((token, float(weight)))
    return result


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def category(row, gt):
    valid = np.isfinite(gt).all() and gt[2] > gt[0] and gt[3] > gt[1]
    if int(row["detections"]) == 0:
        return "no_detection"
    if not valid:
        return "invalid_gt"
    if int(row["gt_index"]) < 0:
        return "no_match"
    if int(row["max_index"]) != int(row["gt_index"]):
        return "changed"
    return "same"


def event_matches(mode, status):
    return mode == "all" or (mode == "problem" and status != "same") or mode == status


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_video(sid, video, frame_rows, predictions):
    cache = OUT / "detections" / f"{sid}.npz"
    label = DATA / "labels" / f"{sid}.mat"
    frame_dir = DATA / "frames" / sid
    with np.load(cache, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    if file_sha256(label) != str(data["annotation_sha256"].item()):
        raise ValueError("Hash annotation hien tai khac voi artifact thi nghiem.")
    item = {"id": sid, "action": video["action"], "source": frame_dir, "label": label}
    seq = Sequence("penn", item)
    total = int(video["frames"])
    offsets, boxes, xy, conf, gt = (data[k] for k in ("offsets", "boxes", "xy", "conf", "gt_bbox"))
    if (
        seq.count != total
        or len(frame_rows) != total
        or len(offsets) != total + 1
        or gt.shape != (total, 4)
    ):
        raise ValueError("So frame/annotation/cache khong khop.")
    if offsets[0] != 0 or offsets[-1] != len(boxes) or np.any(np.diff(offsets) < 0):
        raise ValueError("Offsets detection khong hop le.")
    if xy.shape != (len(boxes), 17, 2) or conf.shape != (len(boxes), 17):
        raise ValueError("Shape skeleton khong khop detection.")
    for t, row in enumerate(frame_rows):
        n = int(offsets[t + 1] - offsets[t])
        a, b = int(row["max_index"]), int(row["gt_index"])
        if (
            int(row["frame"]) != t + 1
            or int(row["detections"]) != n
            or not (-1 <= a < n and -1 <= b < n)
        ):
            raise ValueError(f"Frame {t + 1}: CSV va cache khong khop.")
        if n and a != int(np.argmax(boxes[offsets[t] : offsets[t + 1], 4])):
            raise ValueError(f"Frame {t + 1}: index confidence khong khop.")
    digest = hashlib.sha256()
    for frame in seq.frames:
        raw = frame.read_bytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    if digest.hexdigest() != str(data["image_bytes_sha256"].item()):
        raise ValueError(
            "Hash anh hien tai khac voi anh khi suy luan. Khong hien thi ket qua sai frame."
        )
    data["model_coords"], data["model_normalized"], data["model_positions"], data["model_input"] = (
        make_model_input(data, frame_rows)
    )
    statuses = [category(row, gt[t]) for t, row in enumerate(frame_rows)]
    return seq, data, statuses


def draw_box(image, box, color, text, width=2):
    x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, width)
    y = max(14, min(image.shape[0] - 4, y1 - 4))
    x = max(0, min(image.shape[1] - 1, x1))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def draw_detections(image, boxes, xy, scores, selected, gt, kpt_conf):
    height, width = image.shape[:2]
    if np.isfinite(gt).all() and gt[2] > gt[0] and gt[3] > gt[1]:
        draw_box(image, gt, (50, 210, 50), "GT bbox", 1)
    # Draw the chosen detection last so nearby boxes do not obscure it.
    order = [i for i in range(len(boxes)) if i != selected]
    if 0 <= selected < len(boxes):
        order.append(selected)
    for i in order:
        box, points, conf = boxes[i], xy[i], scores[i]
        selected_now = i == selected
        color = (0, 215, 255) if selected_now else (155, 155, 155)
        draw_box(
            image,
            box,
            color,
            f"{'CHON' if selected_now else 'khac'} {box[4]:.2f}",
            2 if selected_now else 1,
        )
        # Filtering here controls the drawing only, not the saved experiment's features.
        valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf >= kpt_conf)
        valid &= (
            (points[:, 0] > 0)
            & (points[:, 0] < width)
            & (points[:, 1] > 0)
            & (points[:, 1] < height)
        )
        for a, b in COCO_EDGES:
            if valid[a] and valid[b]:
                cv2.line(
                    image,
                    tuple(np.rint(points[a]).astype(int)),
                    tuple(np.rint(points[b]).astype(int)),
                    color,
                    2 if selected_now else 1,
                    cv2.LINE_AA,
                )
        for point in points[valid]:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
    return image


def text(image, value, x, y, color=(255, 255, 255), scale=0.45):
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_model_pose(image, points):
    """Show the selected 12 raw model joints without a display confidence cutoff."""
    h, w = image.shape[:2]
    good = np.isfinite(points).all(axis=1) & (points[:, 0] > 0) & (points[:, 1] > 0)
    good &= (points[:, 0] < w) & (points[:, 1] < h)
    color = (255, 255, 0)
    for a, b in MODEL_EDGES:
        if good[a] and good[b]:
            cv2.line(
                image,
                tuple(np.rint(points[a]).astype(int)),
                tuple(np.rint(points[b]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    for j in np.flatnonzero(good):
        point = tuple(np.rint(points[j]).astype(int))
        cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
    return image


def print_model_input(data, index):
    print(f"\nFrame {index + 1}: 12 toa do pixel GT-ho-tro (NaN = khong ghep duoc):")
    print(np.array2string(data["model_coords"][index], precision=3, suppress_small=True))
    print("24 gia tri sau chuan hoa, truoc lay mau 64 frame:")
    print(np.array2string(data["model_normalized"][index], precision=4, suppress_small=True))
    tokens = contributing_tokens(index, data["model_positions"])
    if not tokens:
        print("Frame nay khong dong gop truc tiep vao 64 buoc sau lay mau.")
    else:
        print(
            "Cac buoc 64-frame co frame nay dong gop (chi so 1-based, trong so; vector 24 gia tri):"
        )
        for token, weight in tokens:
            print(
                f"  Buoc {token + 1}, trong so {weight:.3f}: "
                f"{np.array2string(data['model_input'][token], precision=4, suppress_small=True)}"
            )


def render(seq, data, rows, statuses, index, mode, paused, kpt_conf):
    row = rows[index]
    lo, hi = int(data["offsets"][index]), int(data["offsets"][index + 1])
    boxes, xy, conf, gt = (
        data["boxes"][lo:hi],
        data["xy"][lo:hi],
        data["conf"][lo:hi],
        data["gt_bbox"][index],
    )
    raw = seq.frame(index, False)
    left = seq.frame(index, True)
    if np.isfinite(gt).all() and gt[2] > gt[0] and gt[3] > gt[1]:
        draw_box(left, gt, (50, 210, 50), "GT bbox", 1)
    center = draw_detections(raw.copy(), boxes, xy, conf, int(row["max_index"]), gt, kpt_conf)
    right = draw_detections(raw.copy(), boxes, xy, conf, int(row["gt_index"]), gt, kpt_conf)
    model = draw_model_pose(raw.copy(), data["model_coords"][index])
    h, w = raw.shape[:2]
    canvas = np.zeros((h + 116, w * 4, 3), dtype=np.uint8)
    (
        canvas[116:, :w],
        canvas[116:, w : 2 * w],
        canvas[116:, 2 * w : 3 * w],
        canvas[116:, 3 * w :],
    ) = left, center, right, model
    state = statuses[index]
    names = {
        "changed": "CHON KHAC NHAU",
        "same": "CUNG DETECTION",
        "no_detection": "YOLO KHONG PHAT HIEN",
        "no_match": "CO DETECTION, KHONG GHEP GT",
        "invalid_gt": "GT BBOX KHONG HOP LE",
    }
    text(
        canvas,
        f"{seq.item['id']} | {seq.item['action']} | frame {index + 1}/{seq.count} | {'PAUSE' if paused else 'PLAY'}",
        6,
        18,
    )
    text(canvas, f"{names[state]} | {len(boxes)} detection | tim: {mode}", 6, 39, (0, 215, 255))
    text(canvas, "GT: skeleton + bbox", 6, 72, (50, 210, 50))
    max_iou = row["max_iou"] or "N/A"
    gt_iou = row["gt_iou"] or "N/A"
    max_str = f"{float(max_iou):.2f}" if max_iou != "N/A" else max_iou
    gt_str = f"{float(gt_iou):.2f}" if gt_iou != "N/A" else gt_iou
    text(canvas, f"YOLO confidence | IoU GT: {max_str}", w + 6, 72)
    text(canvas, f"GT ho tro IoU >= 0.30 | IoU: {gt_str}", 2 * w + 6, 72)
    if int(row["gt_index"]) < 0:
        text(canvas, "GT-ASSIST: MISSING", 2 * w + 6, 90, (0, 100, 255))
    text(canvas, "DAU VAO MO HINH: 12 KHOP", 3 * w + 6, 72, (255, 255, 0))
    if int(row["gt_index"]) < 0:
        text(canvas, "KHONG GHEP GT: 24 SO 0 TRUOC LAY MAU", 3 * w + 6, 90, (0, 100, 255), 0.36)
    else:
        points = data["model_coords"][index]
        zero = int(np.sum(np.all(points == 0, axis=1)))
        text(
            canvas,
            f"12 khop nguon | {zero} khop (0,0) | khong loc confidence",
            3 * w + 6,
            90,
            (255, 255, 255),
            0.34,
        )
    ntokens = len(contributing_tokens(index, data["model_positions"]))
    text(
        canvas,
        f"Frame nay gop vao {ntokens} buoc / 64 | bam I xem so",
        3 * w + 6,
        108,
        (255, 255, 255),
        0.34,
    )
    return canvas


def play(sid, video, rows, predictions, start, kpt_conf):
    seq, data, statuses = load_video(sid, video, rows, predictions)
    count = Counter(statuses)
    print(f"\n{sid} | {video['action']} | {seq.count} frame | {dict(count)}")
    for row in predictions:
        print(
            f"Seed {row['seed']}: GT={row['action']}, confidence={ACTION_CLASSES[int(row['confidence_pred'])]}, GT-ho-tro={ACTION_CLASSES[int(row['gt_assisted_pred'])]}"
        )
    print(
        "Space: dung/phat | A/D: lui/tien | E: doi loai su kien | N/P: su kien tiep/truoc | I: du lieu dau vao | R: dau | Q: thoat"
    )
    index, paused, mode_index = start - 1, True, 0
    if not 0 <= index < seq.count:
        raise ValueError("Frame bat dau ngoai pham vi video.")
    window = "So sanh chon nguoi - Penn validation"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1800, 650)
        while True:
            mode = MODES[mode_index]
            cv2.imshow(window, render(seq, data, rows, statuses, index, mode, paused, kpt_conf))
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
            elif key == ord("i"):
                print_model_input(data, index)
            elif key == ord("e"):
                mode_index = (mode_index + 1) % len(MODES)
                print(f"Loai su kien: {MODES[mode_index]}", flush=True)
            elif key in (ord("n"), ord("p")):
                step = 1 if key == ord("n") else -1
                candidates = [
                    i
                    for i in range(index + step, seq.count if step == 1 else -1, step)
                    if event_matches(mode, statuses[i])
                ]
                if candidates:
                    index = candidates[0]
                else:
                    print(f"Khong con su kien {mode} theo huong nay.", flush=True)
                paused = True
            elif not paused:
                index = (index + 1) % seq.count
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        seq.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", help="ID validation, vi du 0018")
    parser.add_argument("--frame", type=int, default=1, help="Frame bat dau (1-based)")
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="Chi anh huong hien thi khop")
    parser.add_argument("--list", action="store_true", help="In danh sach video co su kien")
    parser.add_argument(
        "--save-frame", type=Path, help="Luu anh frame ra file thay vi mo cua so; can --sequence"
    )
    args = parser.parse_args()
    if not 0 <= args.kpt_conf <= 1:
        parser.error("--kpt-conf phai trong [0,1]")
    if args.save_frame and not args.sequence:
        parser.error("--save-frame can --sequence")
    videos = read_csv(OUT / "videos_iou_0_3.csv")
    frame_data = defaultdict(list)
    for row in read_csv(OUT / "frame_matches_iou_0_3.csv"):
        frame_data[row["id"]].append(row)
    prediction_data = defaultdict(list)
    for row in read_csv(OUT / "predictions_iou_0_3.csv"):
        prediction_data[row["id"]].append(row)
    for v in videos:
        v["no_detection"] = int(v["frames"]) - int(v["detected_frames"])
        v["no_match"] = int(v["detected_frames"]) - int(v["gt_matched_frames"])
    if args.list:
        for v in videos:
            if int(v["changed_frames"]) or v["no_detection"] or v["no_match"]:
                print(
                    f"{v['id']} {v['action']}: changed={v['changed_frames']} no_detection={v['no_detection']} no_match_including_invalid_gt={v['no_match']}"
                )
        return
    if args.sequence:
        options = [v for v in videos if v["id"] == args.sequence.zfill(4)]
        if not options:
            parser.error("Khong co video trong artifact validation.")
        if args.save_frame:
            sid = options[0]["id"]
            seq, data, statuses = load_video(sid, options[0], frame_data[sid], prediction_data[sid])
            try:
                if not 1 <= args.frame <= seq.count:
                    parser.error("Frame bat dau ngoai pham vi video.")
                result = render(
                    seq, data, frame_data[sid], statuses, args.frame - 1, "all", True, args.kpt_conf
                )
                if not cv2.imwrite(str(args.save_frame), result):
                    raise OSError(f"Khong ghi duoc {args.save_frame}")
                print(f"Da luu {args.save_frame}")
            finally:
                seq.close()
            return
        play(
            options[0]["id"],
            options[0],
            frame_data[options[0]["id"]],
            prediction_data[options[0]["id"]],
            args.frame,
            args.kpt_conf,
        )
        return
    try:
        while True:
            choices = ["Tat ca video", "Co chon khac", "YOLO khong phat hien", "Khong ghep duoc GT"]
            group = choose("Chon nhom", choices)
            filt = {
                "Tat ca video": lambda v: True,
                "Co chon khac": lambda v: int(v["changed_frames"]) > 0,
                "YOLO khong phat hien": lambda v: v["no_detection"] > 0,
                "Khong ghep duoc GT": lambda v: v["no_match"] > 0,
            }[group]
            options = [v for v in videos if filt(v)]
            if not options:
                print("Khong co video trong nhom nay.")
                continue
            v = choose(
                "Chon video",
                options,
                lambda x: (
                    f"{x['id']} {x['action']} | khac={x['changed_frames']} khong_detection={x['no_detection']} khong_ghep={x['no_match']}"
                ),
            )
            play(
                v["id"], v, frame_data[v["id"]], prediction_data[v["id"]], args.frame, args.kpt_conf
            )
    except (KeyboardInterrupt, EOFError):
        print("\nDa thoat.")


if __name__ == "__main__":
    main()
