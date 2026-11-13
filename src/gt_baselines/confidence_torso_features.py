"""Prediction-only confidence channels and stable clip-wise torso normalization.

No GT keypoints, labels, bbox scale, learned parameters or validation statistics.
Root interpolation and clip median use future frames; this is offline processing.
"""

import numpy as np

THRESHOLD = 0.30
MIN_SCALE_PX = 5.0
DEFAULT_SCALE_PX = 50.0
TORSO = (0, 1, 6, 7)


def observed_joints(xy):
    xy = np.asarray(xy)
    if xy.ndim != 3 or xy.shape[1:] != (12, 2) or len(xy) < 1:
        raise ValueError("Expected nonempty [T,12,2]")
    return np.isfinite(xy).all(2) & (xy > 0).all(2)


def confidence_channels(xy, conf, steps=64):
    """12 interpolated confidence + 12 binary reliable masks at resampled steps."""
    observed = observed_joints(xy)
    conf = np.asarray(conf, dtype=np.float32)
    if conf.shape != observed.shape:
        raise ValueError("Confidence shape mismatch")
    clean = np.where(observed & np.isfinite(conf), np.clip(conf, 0.0, 1.0), 0.0)
    reliable = observed & np.isfinite(conf) & (conf >= THRESHOLD)
    grid = np.linspace(0, len(xy) - 1, steps)
    confidence = np.stack(
        [np.interp(grid, np.arange(len(xy)), clean[:, j]) for j in range(12)], axis=1
    )
    lo, hi = np.floor(grid).astype(int), np.ceil(grid).astype(int)
    mask = reliable[lo] & reliable[hi]
    return np.concatenate((confidence, mask.astype(np.float32)), axis=1).astype(np.float32)


def interpolate_root(root, anchors):
    result = np.zeros_like(root, dtype=np.float64)
    if anchors.any():
        grid = np.arange(len(root))
        for k in range(2):
            result[:, k] = np.interp(grid, grid[anchors], root[anchors, k])
    return result


def normalize_torso(xy, conf):
    """Use reliable hips as root anchors and a median torso scale per clip.

    All observed coordinates are retained, including low-confidence joints. Only
    the root/scale estimation uses confidence, so pose coverage does not change.
    When paired hips never occur, use reliable torso centroid, else all reliable
    joints. Interpolate only the root, never missing joint coordinates.
    """
    xy = np.asarray(xy, dtype=np.float64)
    observed = observed_joints(xy)
    conf = np.asarray(conf, dtype=np.float64)
    if conf.shape != observed.shape:
        raise ValueError("Confidence shape mismatch")
    reliable = observed & np.isfinite(conf) & (conf >= THRESHOLD)
    hips = reliable[:, 6] & reliable[:, 7]
    raw_root = np.full((len(xy), 2), np.nan)
    if hips.any():
        raw_root[hips] = (xy[hips, 6] + xy[hips, 7]) / 2.0
        root = interpolate_root(raw_root, hips)
        root_source = "paired_hips"
        root_anchors = int(hips.sum())
    else:
        anchors = np.zeros(len(xy), bool)
        for t in range(len(xy)):
            good = np.array(TORSO)[reliable[t, TORSO]]
            if not len(good):
                good = np.flatnonzero(reliable[t])
            if len(good):
                raw_root[t] = xy[t, good].mean(0)
                anchors[t] = True
        root = interpolate_root(raw_root, anchors)
        root_source = "reliable_centroid" if anchors.any() else "zero_no_reliable_joints"
        root_anchors = int(anchors.sum())

    torso_valid = reliable[:, TORSO].all(1)
    lengths = np.linalg.norm((xy[:, 0] + xy[:, 1]) / 2.0 - (xy[:, 6] + xy[:, 7]) / 2.0, axis=1)
    candidates = lengths[torso_valid & np.isfinite(lengths) & (lengths > MIN_SCALE_PX)]
    scale_source = "median_torso"
    if not len(candidates):
        shoulder_valid = reliable[:, 0] & reliable[:, 1]
        widths = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1)
        candidates = widths[shoulder_valid & np.isfinite(widths) & (widths > MIN_SCALE_PX)]
        scale_source = "median_shoulders"
    if not len(candidates):
        distances = np.linalg.norm(xy - root[:, None], axis=2)
        radii = np.max(np.where(reliable, distances, 0.0), axis=1)
        candidates = radii[np.isfinite(radii) & (radii > MIN_SCALE_PX)]
        scale_source = "median_reliable_radius"
    if len(candidates):
        scale = float(np.median(candidates))
    else:
        scale = DEFAULT_SCALE_PX
        scale_source = "fixed_fallback"
    normalized = np.where(observed[..., None], (xy - root[:, None]) / scale, 0.0)
    result = normalized.reshape(len(xy), 24).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite torso normalization")
    return result, {
        "frames": len(xy),
        "root_source": root_source,
        "root_anchor_frames": root_anchors,
        "scale_source": scale_source,
        "scale_px": scale,
        "scale_observations": int(len(candidates)),
        "observed_joint_frames": int(observed.sum()),
        "reliable_joint_frames": int(reliable.sum()),
    }


def resample_features(features, steps=64):
    features = np.asarray(features)
    grid = np.linspace(0, len(features) - 1, steps)
    return np.stack(
        [
            np.interp(grid, np.arange(len(features)), features[:, j])
            for j in range(features.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def make_variants(baseline24, xy, conf):
    baseline24 = np.asarray(baseline24, dtype=np.float32)
    if baseline24.shape != (64, 24) or not np.isfinite(baseline24).all():
        raise ValueError("Expected baseline [64,24] finite")
    cm = confidence_channels(xy, conf)
    torso, stats = normalize_torso(xy, conf)
    torso = resample_features(torso)
    zeros = np.zeros((64, 24), np.float32)
    return {
        "baseline": baseline24.copy(),
        "confidence_mask": np.concatenate((baseline24, cm), axis=1),
        "torso": torso,
        "torso_confidence_mask": np.concatenate((torso, cm), axis=1),
        "capacity_control": np.concatenate((baseline24, zeros), axis=1),
        "torso_capacity_control": np.concatenate((torso, zeros), axis=1),
    }, stats
