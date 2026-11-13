"""Prediction-only normalization ablations; independent paper-informed presets.

The Kassir ICCA 2025 original full text was not accessible. Do not call these
an exact reproduction. Thresholds, clipping and missing-data rules are ours.
"""
import numpy as np

import confidence_torso_features as current
import evaluate_pose_corrections as correction

METHODS = ('strict_midhip_shoulder', 'median_shoulder', 'smooth_midhip', 'interpolate3')
CLIP = 5.


def hip_root(xy, conf):
    """Exactly the reliable-hip/interpolated-root policy of the 94.86% setup."""
    reliable = current.observed_joints(xy) & np.isfinite(conf) & (conf >= current.THRESHOLD)
    hips = reliable[:, 6] & reliable[:, 7]
    root = np.full((len(xy), 2), np.nan, dtype=np.float64)
    if hips.any():
        root[hips] = (xy[hips, 6]+xy[hips, 7])/2.
        return current.interpolate_root(root, hips)
    anchors = np.zeros(len(xy), bool)
    for t in range(len(xy)):
        good = np.array(current.TORSO)[reliable[t, current.TORSO]]
        if not len(good):
            good = np.flatnonzero(reliable[t])
        if len(good):
            root[t] = xy[t, good].mean(0); anchors[t] = True
    return current.interpolate_root(root, anchors)


def centered_mean5(values):
    return np.stack([values[max(0, t-2):min(len(values), t+3)].mean(0) for t in range(len(values))])


def normalize(xy, root, scale):
    observed = current.observed_joints(xy)
    return np.where(observed[..., None], (xy-root[:, None])/scale, 0.).reshape(len(xy), 24).astype(np.float32)


def variants(xy, conf):
    xy = np.asarray(xy, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    observed = current.observed_joints(xy)
    if conf.shape != observed.shape:
        raise ValueError('Confidence shape mismatch')
    reliable = observed & np.isfinite(conf) & (conf >= current.THRESHOLD)
    base_native, stats = current.normalize_torso(xy, conf)
    root = hip_root(xy, conf)
    # Check the replicated root against the existing normalizer, without changing it.
    np.testing.assert_allclose(normalize(xy, root, stats['scale_px']), base_native, rtol=1e-6, atol=1e-6)
    cm = current.confidence_channels(xy, conf)
    outputs = {'reference': np.concatenate((current.resample_features(base_native), cm), axis=1)}

    # Strict framewise midhip/shoulder: project-chosen reliability/min-width/clip.
    widths = np.linalg.norm(xy[:, 0]-xy[:, 1], axis=1)
    frame_valid = reliable[:, current.TORSO].all(1) & np.isfinite(widths) & (widths > current.MIN_SCALE_PX)
    strict = np.zeros((len(xy), 12, 2), np.float64)
    strict[frame_valid] = (xy[frame_valid] - ((xy[frame_valid, 6]+xy[frame_valid, 7])/2.)[:, None])/widths[frame_valid, None, None]
    strict = np.where(observed[..., None], strict, 0.)
    strict = np.clip(strict, -CLIP, CLIP).reshape(len(xy), 24)
    strict_xy = xy.copy(); strict_conf = conf.copy()
    strict_xy[~frame_valid] = np.nan; strict_conf[~frame_valid] = 0.
    strict_cm = current.confidence_channels(strict_xy, strict_conf)
    outputs['strict_midhip_shoulder'] = np.concatenate((current.resample_features(strict), strict_cm), axis=1)

    # Change only clip scale; keep root, coordinate coverage and confidence/mask.
    shoulder_good = reliable[:, 0] & reliable[:, 1] & np.isfinite(widths) & (widths > current.MIN_SCALE_PX)
    shoulder_scale = float(np.median(widths[shoulder_good])) if shoulder_good.any() else stats['scale_px']
    outputs['median_shoulder'] = np.concatenate((current.resample_features(normalize(xy, root, shoulder_scale)), cm), axis=1)

    # Change only root trajectory, retaining all joints and the original uncertainty.
    smooth_root = centered_mean5(root)
    outputs['smooth_midhip'] = np.concatenate((current.resample_features(normalize(xy, smooth_root, stats['scale_px'])), cm), axis=1)

    # Imputed coordinates do not acquire a fake observed/confident mask.
    interp, ic, filled = correction.interpolate_short(xy.astype(np.float32), conf.astype(np.float32))
    interpolated, _ = current.normalize_torso(interp, ic)
    outputs['interpolate3'] = np.concatenate((current.resample_features(interpolated), cm), axis=1)
    for method, x in outputs.items():
        if x.shape != (64, 48) or not np.isfinite(x).all():
            raise ValueError('Invalid features '+method)
        if not observed.any() and np.any(x != 0):
            raise ValueError('Invented features in absent video')
    return outputs, {'strict_valid_frames': int(frame_valid.sum()), 'frames': len(xy),
                     'median_shoulder_scale_px': shoulder_scale, 'torso_scale_px': stats['scale_px'],
                     'shoulder_scale_observations': int(shoulder_good.sum()),
                     'filled_joint_frames': int(filled.sum())}
