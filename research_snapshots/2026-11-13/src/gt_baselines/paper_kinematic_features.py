"""Independent 12-joint adaptation of Ma et al. (2026), sections 3.6/3.8.

These are per-frame component features, not four aggregate profile percentiles.
Keep the project's coordinate normalization and 64-step representation fixed.
Angular channels are in radians/pi; finite differences are per resampled step,
not physical speed in seconds. Invalid joints never generate derivative spikes.
"""
import numpy as np

ANGLE_TRIPLETS = ((2, 0, 6), (3, 1, 7), (0, 2, 4), (1, 3, 5),
                  (0, 6, 8), (1, 7, 9), (6, 8, 10), (7, 9, 11))
ANGLE_NAMES = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
               'left_hip', 'right_hip', 'left_knee', 'right_knee')
TORSO = (0, 1, 6, 7)
DIMENSIONS = {'coordinates': 24, 'angles': 32, 'velocity': 48,
              'full': 59, 'capacity_control': 59}


def resampled_validity(xy, steps=64):
    """A resampled joint is valid iff all contributing source points are valid."""
    xy = np.asarray(xy)
    if xy.ndim != 3 or xy.shape[1:] != (12, 2) or len(xy) < 1:
        raise ValueError('Expected nonempty [T,12,2]')
    valid = np.isfinite(xy).all(2) & (xy > 0).all(2)
    grid = np.linspace(0, len(xy)-1, steps)
    lo = np.floor(grid).astype(int)
    hi = np.ceil(grid).astype(int)
    return valid[lo] & valid[hi]


def components(pos24, valid):
    pos24 = np.asarray(pos24, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if pos24.ndim != 2 or pos24.shape[1] != 24 or valid.shape != (len(pos24), 12):
        raise ValueError('Coordinate/mask shape mismatch')
    if not np.isfinite(pos24).all():
        raise ValueError('Nonfinite normalized coordinates')
    p = pos24.reshape(-1, 12, 2)
    angles = np.zeros((len(p), 8), np.float32)
    for k, (a, j, b) in enumerate(ANGLE_TRIPLETS):
        u, v = p[:, a]-p[:, j], p[:, b]-p[:, j]
        denom = np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1)
        good = valid[:, a] & valid[:, j] & valid[:, b] & (denom > 1e-8)
        cosine = np.sum(u[good]*v[good], axis=1)/denom[good]
        angles[good, k] = np.arccos(np.clip(cosine, -1., 1.))/np.pi
    velocity = np.zeros_like(p)
    if len(p) > 1:
        velocity[1:] = np.where((valid[1:] & valid[:-1])[..., None], p[1:]-p[:-1], 0.)
    com_valid = valid[:, TORSO].all(1)
    com = np.where(com_valid[:, None], p[:, TORSO].mean(1), 0.).astype(np.float32)
    speed = np.zeros((len(p), 1), np.float32)
    if len(p) > 1:
        speed[1:, 0] = np.where(com_valid[1:] & com_valid[:-1], np.linalg.norm(com[1:]-com[:-1], axis=1), 0.)
    return angles, velocity.reshape(-1, 24), com, speed


def enrich(pos24, valid, mode):
    if mode not in DIMENSIONS:
        raise ValueError(mode)
    pos24 = np.asarray(pos24, dtype=np.float32)
    if mode == 'coordinates':
        return pos24.copy()
    if mode == 'capacity_control':
        return np.concatenate((pos24, np.zeros((len(pos24), 35), np.float32)), axis=1)
    angles, velocity, com, speed = components(pos24, valid)
    extra = {'angles': (angles,), 'velocity': (velocity,),
             'full': (angles, velocity, com, speed)}[mode]
    result = np.concatenate((pos24, *extra), axis=1).astype(np.float32)
    if result.shape != (len(pos24), DIMENSIONS[mode]) or not np.isfinite(result).all():
        raise ValueError('Invalid kinematic features')
    return result
