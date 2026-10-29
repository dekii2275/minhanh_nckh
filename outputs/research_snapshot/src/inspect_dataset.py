"""
Module: inspect_dataset.py
Purpose: Inspect Penn Action annotations (.mat), verify coordinate systems,
         and visualize Ground Truth directly onto RGB frames to validate alignment.
"""

import glob
import os

import cv2
import numpy as np
import scipy.io as sio

# Penn Action 13 joint labels (1-indexed in MATLAB, 0-indexed in Python)
PENN_ACTION_JOINTS = [
    "head",  # 0 (MATLAB 1)
    "left_shoulder",  # 1 (MATLAB 2)
    "right_shoulder",  # 2 (MATLAB 3)
    "left_elbow",  # 3 (MATLAB 4)
    "right_elbow",  # 4 (MATLAB 5)
    "left_wrist",  # 5 (MATLAB 6)
    "right_wrist",  # 6 (MATLAB 7)
    "left_hip",  # 7 (MATLAB 8)
    "right_hip",  # 8 (MATLAB 9)
    "left_knee",  # 9 (MATLAB 10)
    "right_knee",  # 10 (MATLAB 11)
    "left_ankle",  # 11 (MATLAB 12)
    "right_ankle",  # 12 (MATLAB 13)
]

# Anatomical skeleton connections for Penn Action
PENN_SKELETON_EDGES = [
    (0, 1),
    (0, 2),  # head to shoulders
    (1, 2),  # shoulder to shoulder
    (1, 3),
    (3, 5),  # left arm
    (2, 4),
    (4, 6),  # right arm
    (1, 7),
    (2, 8),  # shoulders to hips
    (7, 8),  # hip to hip
    (7, 9),
    (9, 11),  # left leg
    (8, 10),
    (10, 12),  # right leg
]


def load_penn_annotation(mat_path):
    """
    Load and parse a Penn Action .mat annotation file.
    Returns a dict with parsed, standardized fields.
    """
    data = sio.loadmat(mat_path)

    # Clean string fields
    action = str(data["action"][0]) if "action" in data else "unknown"
    pose = str(data["pose"][0]) if "pose" in data else "unknown"
    nframes = int(data["nframes"][0, 0]) if "nframes" in data else 0

    x = np.array(data["x"], dtype=np.float32)
    y = np.array(data["y"], dtype=np.float32)
    visibility = np.array(data["visibility"], dtype=np.int32)
    bbox = np.array(data["bbox"], dtype=np.float32) if "bbox" in data else None
    dimensions = (
        np.array(data["dimensions"].squeeze(), dtype=np.int32) if "dimensions" in data else None
    )

    # Check orientation: we expect shape (nframes, 13)
    if x.shape[1] != 13 and x.shape[0] == 13:
        x = x.T
        y = y.T
        visibility = visibility.T

    return {
        "mat_path": mat_path,
        "action": action,
        "pose": pose,
        "nframes": nframes,
        "x": x,
        "y": y,
        "visibility": visibility,
        "bbox": bbox,
        "dimensions": dimensions,
        "raw_keys": [k for k in data.keys() if not k.startswith("__")],
    }


def print_mat_inspection(anno):
    """Print comprehensive diagnostic information about a .mat file."""
    print("=" * 60)
    print(f"File: {anno['mat_path']}")
    print(f"Raw Keys: {anno['raw_keys']}")
    print(f"Action: '{anno['action']}', Pose: '{anno['pose']}', Total Frames: {anno['nframes']}")
    print(
        f"x shape: {anno['x'].shape}, dtype: {anno['x'].dtype}, range: [{anno['x'].min():.1f}, {anno['x'].max():.1f}]"
    )
    print(
        f"y shape: {anno['y'].shape}, dtype: {anno['y'].dtype}, range: [{anno['y'].min():.1f}, {anno['y'].max():.1f}]"
    )
    print(
        f"visibility shape: {anno['visibility'].shape}, sum visible: {anno['visibility'].sum()}/{anno['visibility'].size}"
    )
    if anno["dimensions"] is not None:
        print(f"Dimensions: {anno['dimensions']} (Height x Width x Frames)")
    if anno["bbox"] is not None:
        print(f"bbox shape: {anno['bbox'].shape}")
    print("=" * 60)


def draw_ground_truth_on_frame(img_bgr, x_coords, y_coords, vis_flags, frame_info_text=""):
    """
    Draw Penn Action GT keypoints (green) and skeleton lines on the frame.
    """
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]

    # Draw skeleton connections first
    for j1, j2 in PENN_SKELETON_EDGES:
        if vis_flags[j1] == 1 and vis_flags[j2] == 1:
            x1, y1 = int(round(x_coords[j1])), int(round(y_coords[j1]))
            x2, y2 = int(round(x_coords[j2])), int(round(y_coords[j2]))
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                # Skeleton line in green (BGR: 0, 220, 0)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2, cv2.LINE_AA)

    # Draw keypoint dots & labels
    for j_idx in range(len(PENN_ACTION_JOINTS)):
        if vis_flags[j_idx] == 1:
            kx, ky = int(round(x_coords[j_idx])), int(round(y_coords[j_idx]))
            if 0 <= kx < w and 0 <= ky < h:
                # Green circle for Ground Truth
                cv2.circle(canvas, (kx, ky), 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, (kx, ky), 7, (0, 100, 0), 1, cv2.LINE_AA)

    if frame_info_text:
        cv2.putText(
            canvas,
            frame_info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return canvas


def inspect_and_verify_gt(
    penn_dir="data/raw/Penn_Action", output_dir="outputs/figures/gt_inspection"
):
    """
    Run GT verification across sample sequences and export verification images.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(penn_dir, "labels")
    frames_dir = os.path.join(penn_dir, "frames")

    if not os.path.exists(labels_dir):
        print(f"Labels directory not found at {labels_dir}")
        return False

    mat_files = sorted(glob.glob(os.path.join(labels_dir, "*.mat")))
    print(f"Found {len(mat_files)} annotation files in {labels_dir}")

    inspected_count = 0
    # Test across first 5 files
    for mat_path in mat_files[:5]:
        seq_id = os.path.splitext(os.path.basename(mat_path))[0]
        seq_frame_dir = os.path.join(frames_dir, seq_id)
        if not os.path.exists(seq_frame_dir):
            continue

        anno = load_penn_annotation(mat_path)
        print_mat_inspection(anno)

        # Pick 2 frames per sequence to inspect
        test_frame_indices = [0, anno["nframes"] // 2]
        for f_idx in test_frame_indices:
            frame_filename = f"{f_idx + 1:06d}.jpg"
            frame_path = os.path.join(seq_frame_dir, frame_filename)
            if not os.path.exists(frame_path):
                print(f"Frame not found: {frame_path}")
                continue

            img = cv2.imread(frame_path)
            if img is None:
                continue

            info = f"Seq: {seq_id} | Act: {anno['action']} | Frame: {f_idx + 1}/{anno['nframes']}"
            vis_img = draw_ground_truth_on_frame(
                img,
                anno["x"][f_idx],
                anno["y"][f_idx],
                anno["visibility"][f_idx],
                frame_info_text=info,
            )

            out_img_path = os.path.join(output_dir, f"gt_verify_{seq_id}_f{f_idx + 1:04d}.jpg")
            cv2.imwrite(out_img_path, vis_img)
            print(f"Saved GT verification image: {out_img_path}")
            inspected_count += 1

    print(f"Successfully inspected and exported {inspected_count} verification frames.")
    return inspected_count >= 5


if __name__ == "__main__":
    inspect_and_verify_gt()
