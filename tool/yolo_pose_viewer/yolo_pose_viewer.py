"""Run YOLO pose on dataset sequences or a local video and display predictions."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tool' / 'dataset_viewer'))
from dataset_viewer import catalog, choose


class VideoSource:
    def __init__(self, source):
        self.source = Path(source)
        self.cap = None
        if self.source.is_dir():
            self.frames = sorted(p for p in self.source.iterdir()
                                 if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            self.count, self.fps = len(self.frames), 15.0
        else:
            self.cap = cv2.VideoCapture(str(self.source))
            if not self.cap.isOpened():
                self.cap.release()
                raise ValueError(f'Khong mo duoc video: {self.source}')
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(self.fps) or self.fps <= 0:
                self.fps = 15.0
        if self.count <= 0:
            self.close()
            raise ValueError(f'Khong co frame doc duoc: {self.source}')

    def read(self, index):
        if not 0 <= index < self.count:
            raise ValueError('Frame ngoai pham vi')
        if self.cap is None:
            frame = cv2.imdecode(np.fromfile(str(self.frames[index]), dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok:
                frame = None
        if frame is None:
            raise ValueError(f'Khong doc duoc frame {index + 1}')
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


def predict(model, frame, confidence, device):
    options = {'source': frame, 'conf': confidence, 'verbose': False}
    if device is not None:
        options['device'] = device
    result = model.predict(**options)[0]
    if result.keypoints is None:
        raise ValueError('Model khong tra ve keypoints. Hay chon model YOLO pose.')
    return result.plot(), len(result.boxes)


def play(model, source, action, confidence, device):
    video = VideoSource(source)
    window = 'YOLO-Pose viewer'
    print(f'\nVideo: {source}\nNhan dataset: {action or "khong co"}; {video.count} frame, {video.fps:.1f} FPS')
    print('Space: dung/phat | A/D: lui/tien frame | S: bat/tat du doan | R: ve dau | Q/Esc: dong')
    index, paused, show = 0, False, True
    cached_index, raw, annotated, people = -1, None, None, 0
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            if index != cached_index:
                raw = video.read(index)
                annotated, people = predict(model, raw, confidence, device)
                cached_index = index
            image = (annotated if show else raw).copy()
            title = f"GT action: {action or 'N/A'} | {index + 1}/{video.count} | {'PAUSE' if paused else 'PLAY'}"
            info = f"YOLO pose: {'ON' if show else 'OFF'} | persons: {people}"
            cv2.rectangle(image, (0, 0), (image.shape[1], 43), (0, 0, 0), -1)
            cv2.putText(image, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, info, (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, image)
            key = cv2.waitKey(50 if paused else max(1, int(1000 / video.fps))) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                show = not show
            elif key == ord('a'):
                index, paused = max(0, index - 1), True
            elif key == ord('d'):
                index, paused = min(video.count - 1, index + 1), True
            elif key == ord('r'):
                index = 0
            elif not paused:
                index = (index + 1) % video.count
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        video.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--dataset', choices=['penn', 'jhmdb'])
    parser.add_argument('--action', help='Ten hanh dong trong dataset')
    parser.add_argument('--sequence', help='ID chuoi/video trong dataset')
    parser.add_argument('--video', type=Path, help='Duong dan video rieng')
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'yolo11n-pose.pt')
    parser.add_argument('--conf', type=float, default=0.25, help='Nguong tin cay phat hien nguoi (0-1)')
    parser.add_argument('--device', default='0', help='GPU (mac dinh: 0) hoac cpu')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('--conf phai nam trong khoang 0-1')
    if not args.model.is_file():
        parser.error(f'Khong tim thay model: {args.model}')
    if args.video and (args.dataset or args.action or args.sequence):
        parser.error('--video khong dung cung --dataset, --action, --sequence')
    if args.video and not args.video.is_file():
        parser.error(f'Khong tim thay video: {args.video}')
    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('Khong tim thay GPU CUDA. Kiem tra PyTorch/CUDA hoac chay voi --device cpu.')
    print(f'Thiet bi: {args.device}' +
          (f' ({torch.cuda.get_device_name(int(args.device))})' if args.device.isdigit() and int(args.device) < torch.cuda.device_count() else ''),
          flush=True)
    print(f'Dang tai model: {args.model}', flush=True)
    model = YOLO(str(args.model), task='pose')
    if args.video:
        play(model, args.video, None, args.conf, args.device)
        return
    print('Dang doc danh sach dataset...', flush=True)
    datasets = catalog(args.data_root)
    try:
        while True:
            available = [name for name, items in datasets.items() if items]
            if not available:
                print(f'Khong tim thay dataset tai {args.data_root}')
                return
            dataset = args.dataset or choose('Chon dataset', available, lambda x: f'{x} ({len(datasets[x])} video)')
            items = datasets[dataset]
            actions = sorted({item['action'] for item in items})
            if not actions:
                print(f'Khong co du lieu {dataset}')
                return
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
                play(model, item['source'], action, args.conf, args.device)
            except (ValueError, OSError, cv2.error) as exc:
                print(f'Khong the xem video: {exc}')
            if args.sequence:
                return
    except (KeyboardInterrupt, EOFError):
        print('\nDa thoat.')


if __name__ == '__main__':
    main()
