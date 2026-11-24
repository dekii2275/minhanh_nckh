"""Show dataset ground-truth and YOLO pose side by side on the same video."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tool" / "dataset_viewer"))
from dataset_viewer import Sequence, catalog, choose  # noqa: E402

# COCO keypoint order used by the pretrained YOLO pose model.
COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
COCO_NAMES = (
    "nose",
    "L_eye",
    "R_eye",
    "L_ear",
    "R_ear",
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hip",
    "R_hip",
    "L_knee",
    "R_knee",
    "L_ankle",
    "R_ankle",
)


def predict(model, image, confidence, device):
    result = model.predict(source=image, imgsz=640, conf=confidence, device=device, verbose=False)[
        0
    ]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 17, 2)), np.empty((0, 17)), np.empty((0, 4)), np.empty(0)
    if result.keypoints is None:
        raise ValueError("Model khong tra ve keypoints. Hay chon model YOLO pose.")
    coords = result.keypoints.xy.cpu().numpy()
    scores = result.keypoints.conf
    scores = scores.cpu().numpy() if scores is not None else np.full(coords.shape[:2], np.nan)
    if coords.ndim != 3 or coords.shape[1:] != (17, 2):
        raise ValueError("Model phai su dung 17 khop COCO.")
    boxes = result.boxes.xyxy.cpu().numpy()
    box_scores = result.boxes.conf.cpu().numpy()
    if len(coords) != len(boxes):
        raise ValueError("So skeleton va bounding box khong khop.")
    return coords, scores, boxes, box_scores


def draw_yolo(image, coords, scores, boxes, box_scores, threshold, show_labels=True):
    height, width = image.shape[:2]
    selected = int(np.argmax(box_scores)) if len(box_scores) else None
    for i, (xy, conf, box, box_score) in enumerate(zip(coords, scores, boxes, box_scores)):
        color = (0, 215, 255) if i == selected else (180, 180, 180)
        thickness = 2 if i == selected else 1
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{'CHON' if i == selected else 'khac'} {box_score:.2f}"
        label_y = max(15, min(height - 4, y1 - 4))
        cv2.putText(
            image,
            label,
            (max(0, x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (max(0, x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        # A missing confidence channel is unknown, not a confidence of 1.
        valid = np.isfinite(xy).all(axis=1) & (~np.isfinite(conf) | (conf >= threshold))
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        valid &= (xy[:, 0] > 0) & (xy[:, 1] > 0)
        for a, b in COCO_EDGES:
            if valid[a] and valid[b]:
                cv2.line(
                    image,
                    tuple(np.rint(xy[a]).astype(int)),
                    tuple(np.rint(xy[b]).astype(int)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
        for joint in np.flatnonzero(valid):
            x, y = np.rint(xy[joint]).astype(int)
            cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
            if show_labels and i == selected:
                score = f"{conf[joint]:.2f}" if np.isfinite(conf[joint]) else "N/A"
                text = f"{COCO_NAMES[joint]} {score}"
                scale = 0.35
                (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                origin = (
                    max(0, min(width - text_width - 1, x + 4)),
                    max(10, min(height - 3, y - 4)),
                )
                cv2.putText(
                    image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
                )
    return image


def play(model, dataset, item, args):
    seq = Sequence(dataset, item)
    window = "GT vs YOLO-Pose"
    print(f"\n{dataset.upper()} | {item['action']} | {item['id']} | {seq.count} frames")
    print("Ben trai: GT | Ben phai: CHON vang = box confidence cao nhat; xam = nguoi khac")
    print("Space: dung/phat | A/D: lui/tien | C: nhan confidence | R: ve dau | Q/Esc: dong")
    index, paused, gt, yolo, labels = 0, False, True, True, True
    cached_index = -1
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            if cached_index != index:
                raw = seq.frame(index, False)
                gt_image = seq.frame(index, True)
                coords, scores, boxes, box_scores = predict(model, raw, args.conf, args.device)
                cached_index = index
            left = (gt_image if gt else raw).copy()
            right = raw.copy()
            if yolo:
                draw_yolo(right, coords, scores, boxes, box_scores, args.kpt_conf, labels)
            # Put captions above the frames so the original video is not obscured.
            height, width = raw.shape[:2]
            image = np.zeros((height + 54, width * 2, 3), dtype=np.uint8)
            image[54:, :width] = left
            image[54:, width:] = right
            status = f"{index + 1}/{seq.count} | {'PAUSE' if paused else 'PLAY'}"
            cv2.putText(
                image,
                f"GT | {status}",
                (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 225, 40),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Action: {item['action']}",
                (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"YOLO | {status}",
                (width + 5, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 100, 255),
                1,
                cv2.LINE_AA,
            )
            chosen = f"{float(np.max(box_scores)):.2f}" if len(box_scores) else "none"
            cv2.putText(
                image,
                f"{len(coords)} people | CHON: {chosen}",
                (width + 5, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, image)
            key = cv2.waitKey(50 if paused else max(1, int(1000 / seq.fps))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("g"):
                gt = not gt
            elif key == ord("y"):
                yolo = not yolo
            elif key == ord("c"):
                labels = not labels
            elif key == ord("a"):
                index, paused = max(0, index - 1), True
            elif key == ord("d"):
                index, paused = min(seq.count - 1, index + 1), True
            elif key == ord("r"):
                index = 0
            elif not paused:
                index = (index + 1) % seq.count
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        seq.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--dataset", choices=["penn", "jhmdb"])
    parser.add_argument("--action", help="Ten hanh dong trong dataset")
    parser.add_argument("--sequence", help="ID video trong dataset")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "yolo11n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Nguong phat hien nguoi")
    parser.add_argument(
        "--kpt-conf", type=float, default=0.0, help="Nguong hien thi khop YOLO (mac dinh 0)"
    )
    parser.add_argument("--device", default="0", help="GPU (mac dinh 0) hoac cpu")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.kpt_conf <= 1:
        parser.error("Nguong tin cay phai nam trong khoang 0-1")
    if not args.model.is_file():
        parser.error(f"Khong tim thay model: {args.model}")
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("Khong tim thay GPU CUDA. Kiem tra CUDA hoac dung --device cpu.")
    print(f"Dang tai model: {args.model} | Thiet bi: {args.device}", flush=True)
    model = YOLO(str(args.model), task="pose")
    print("Dang doc danh sach dataset...", flush=True)
    datasets = catalog(args.data_root)
    try:
        while True:
            available = [name for name, items in datasets.items() if items]
            if not available:
                print(f"Khong tim thay dataset tai {args.data_root}")
                return
            dataset = args.dataset or choose(
                "Chon dataset", available, lambda x: f"{x} ({len(datasets[x])} video)"
            )
            items = datasets[dataset]
            actions = sorted({item["action"] for item in items})
            if not actions:
                print(f"Khong co du lieu {dataset}")
                return
            action = args.action or choose("Chon hanh dong", actions)
            matches = [item for item in items if item["action"] == action]
            if not matches:
                print(f"Khong tim thay hanh dong: {action}")
                return
            if args.sequence:
                matches = [item for item in matches if item["id"] == args.sequence]
                if not matches:
                    print(f"Khong tim thay video: {args.sequence}")
                    return
                item = matches[0]
            else:
                item = choose("Chon video", matches, lambda x: x["id"])
            try:
                play(model, dataset, item, args)
            except (ValueError, OSError, cv2.error) as exc:
                print(f"Khong the xem video: {exc}")
            if args.sequence:
                return
    except (KeyboardInterrupt, EOFError):
        print("\nDa thoat.")


if __name__ == "__main__":
    main()
