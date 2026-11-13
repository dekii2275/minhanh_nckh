"""View dataset videos and their ground-truth action labels and skeletons."""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from inspect_dataset import PENN_SKELETON_EDGES, load_penn_annotation
J_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6),
           (3, 7), (7, 11), (4, 8), (8, 12), (5, 9), (9, 13), (6, 10), (10, 14)]


def catalog(root):
    result = {"penn": [], "jhmdb": []}
    penn = root / "Penn_Action"
    for label in sorted((penn / "labels").glob("*.mat")):
        folder = penn / "frames" / label.stem
        if not folder.is_dir():
            continue
        try:
            action = str(loadmat(label, variable_names=["action"])["action"].flat[0])
        except (KeyError, ValueError, OSError):
            continue
        result["penn"].append({"id": label.stem, "action": action,
                               "label": label, "source": folder})
    jroot = root / "jhmdb"
    for video in sorted((jroot / "ReCompress_Videos").glob("*/*.avi")):
        label = jroot / "joint_positions" / video.parent.name / video.stem / "joint_positions.mat"
        if label.is_file():
            result["jhmdb"].append({"id": video.stem, "action": video.parent.name,
                                    "label": label, "source": video})
    return result


@lru_cache(maxsize=8)
def annotation(dataset, path):
    if dataset == "penn":
        data = load_penn_annotation(path)
        return np.stack((data["x"], data["y"]), axis=-1), data["visibility"].astype(bool)
    pos = loadmat(path)["pos_img"]
    if pos.ndim != 3 or pos.shape[:2] != (2, 15):
        raise ValueError("J-HMDB pos_img must have shape (2, 15, frames)")
    points = pos.transpose(2, 1, 0)
    return points, np.isfinite(points).all(axis=-1) & (points > 0).all(axis=-1)


class Sequence:
    def __init__(self, dataset, item):
        self.dataset = dataset
        self.item = item
        self.points, self.visible = annotation(dataset, str(item["label"]))
        self.cap = None
        if dataset == "penn":
            self.frames = sorted(p for p in item["source"].iterdir()
                                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            source_count = len(self.frames)
            self.fps = 15.0  # Penn distributes images without playback timing.
        else:
            self.cap = cv2.VideoCapture(str(item["source"]))
            if not self.cap.isOpened():
                raise ValueError("Khong mo duoc video: " + str(item["source"]))
            source_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(self.fps) or self.fps <= 0:
                self.fps = 15.0
        self.count = min(source_count, len(self.points))
        if source_count != len(self.points):
            print(f"Canh bao: {source_count} frame video, {len(self.points)} frame nhan; "
                  f"chi xem {self.count} frame chung.")
        if not self.count:
            self.close()
            raise ValueError("Khong co frame de hien thi")

    def close(self):
        if self.cap is not None:
            self.cap.release()

    def frame(self, index, show_skeleton=True):
        if not 0 <= index < self.count:
            raise ValueError("Frame out of range")
        if self.dataset == "penn":
            # imdecode supports Windows paths containing non-ASCII characters.
            image = cv2.imdecode(np.fromfile(str(self.frames[index]), dtype=np.uint8), cv2.IMREAD_COLOR)
            edges = PENN_SKELETON_EDGES
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = self.cap.read()
            if not ok:
                image = None
            edges = J_EDGES
        if image is None:
            raise ValueError(f"Khong doc duoc frame {index + 1}")
        if show_skeleton:
            xy = self.points[index]
            height, width = image.shape[:2]
            valid = self.visible[index] & np.isfinite(xy).all(axis=1)
            valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
            for a, b in edges:
                if valid[a] and valid[b]:
                    cv2.line(image, tuple(np.rint(xy[a]).astype(int)), tuple(np.rint(xy[b]).astype(int)), (40, 225, 40), 2, cv2.LINE_AA)
            for point in xy[valid]:
                cv2.circle(image, tuple(np.rint(point).astype(int)), 4, (0, 255, 255), -1, cv2.LINE_AA)
        return image


def choose(prompt, options, display=str):
    for i, option in enumerate(options, 1):
        print(f"{i:4}. {display(option)}")
    while True:
        value = input(f"{prompt} (1-{len(options)}, q de thoat): ").strip()
        if value.lower() == 'q':
            raise KeyboardInterrupt
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1]
        print("Lua chon khong hop le.")


def play(dataset, item):
    seq = Sequence(dataset, item)
    window = "Dataset viewer - xem phim tat trong terminal"
    print(f"\n{dataset.upper()} | {item['action']} | {item['id']}")
    print(f"{seq.count} frames | {seq.fps:.1f} FPS" + (" (gia dinh cho Penn)" if dataset == 'penn' else ""))
    print("Phim: Space dung/phat | A/D lui/tien 1 frame | S bat/tat xuong | R ve dau | Q chon video khac")
    print("Xuong la annotation goc; mau xanh la canh, cham vang la khop.")
    index, paused, skeleton = 0, False, True
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            image = seq.frame(index, skeleton)
            status = f"{item['action']} | frame {index + 1}/{seq.count} | {'PAUSE' if paused else 'PLAY'} | skeleton {'ON' if skeleton else 'OFF'}"
            cv2.rectangle(image, (0, 0), (image.shape[1], 31), (0, 0, 0), -1)
            cv2.putText(image, status, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, image)
            key = cv2.waitKey(50 if paused else max(1, int(1000 / seq.fps))) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                skeleton = not skeleton
            elif key == ord('a'):
                index, paused = max(0, index - 1), True
            elif key == ord('d'):
                index, paused = min(seq.count - 1, index + 1), True
            elif key == ord('r'):
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
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--dataset', choices=['penn', 'jhmdb'])
    parser.add_argument('--action', help='Loc nhan hanh dong, vi du tennis_serve')
    parser.add_argument('--sequence', help='ID cua chuoi/video')
    args = parser.parse_args()
    print('Dang doc danh sach dataset...')
    datasets = catalog(args.data_root)
    try:
        while True:
            available = [name for name, items in datasets.items() if items]
            if not available:
                print(f'Khong tim thay video va nhan tai {args.data_root}')
                return
            dataset = args.dataset or choose('Chon dataset', available, lambda x: f'{x} ({len(datasets[x])} video)')
            items = datasets[dataset]
            if not items:
                print(f'Khong co du lieu {dataset} tai {args.data_root}')
                return
            actions = sorted({item['action'] for item in items})
            action = args.action or choose('Chon hanh dong', actions)
            matches = [item for item in items if item['action'] == action]
            if not matches:
                print(f'Khong tim thay hanh dong: {action}')
                return
            if args.sequence:
                matches = [item for item in matches if item['id'] == args.sequence]
                if not matches:
                    print(f'Khong tim thay video: {args.sequence}')
                    return
                item = matches[0]
            else:
                item = choose('Chon video', matches, lambda x: x['id'])
            try:
                play(dataset, item)
            except (ValueError, OSError, cv2.error) as exc:
                print(f'Khong the xem video: {exc}')
            if args.sequence:
                return
    except (KeyboardInterrupt, EOFError):
        print('\nDa thoat.')


if __name__ == '__main__':
    main()

