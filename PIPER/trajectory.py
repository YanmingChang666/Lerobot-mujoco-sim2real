"""Timed multistroke trajectories with full-pose IK at the pen tip."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from control.hanzi_loader import text_to_strokes
from control.latin_strokes import latin_to_strokes
from .env import DT

PEN_ROTATION = np.diag([-1., 1., -1.])  # pen local +Z points down


def solve_ik(env, point, seed, target_rotation=PEN_ROTATION):
    def error(q):
        env.data.qpos[env.qadr] = q
        state = env.state()
        rot = env.data.site_xmat[env.site].reshape(3, 3)
        orientation = Rotation.from_matrix(target_rotation @ rot.T).as_rotvec()
        return np.r_[state[:3] - point, 0.15 * orientation]

    bounds = (env.limits[:, 0] + 1e-5, env.limits[:, 1] - 1e-5)
    result = least_squares(error, np.clip(seed, *bounds), bounds=bounds,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=100)
    residual = error(result.x)
    if np.linalg.norm(residual[:3]) > 0.0002 or np.linalg.norm(residual[3:]) / .15 > .005:
        raise ValueError(f"Unreachable pen pose {point}: position error={np.linalg.norm(residual[:3]):.5f}m")
    return result.x


def cartesian_path(text, size, center, height, lift, speed, data_dir, rotation_deg=0.):
    """Text -> timed pen-tip points (N,3) and pen-down mask, before any IK."""
    if not text.strip() or min(size, lift, speed) <= 0:
        raise ValueError("Text must be nonempty; size, lift and speed must be positive")
    if not np.isfinite(rotation_deg):
        raise ValueError("Rotation angle must be finite")
    angle = np.radians(rotation_deg)
    rotation_xy = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
    if text.isascii():
        strokes = latin_to_strokes(text, size_m=size)
        offset = 0.
    else:
        if any(c.isascii() and c.isalpha() for c in text):
            raise ValueError("Mixed Chinese/English layout is not supported; write them separately")
        strokes = text_to_strokes(text, size_m=size, data_dir=str(data_dir))
        offset = (len(text) - 1) * size * 1.2 / 2
    points, down = [], []

    def append_segment(target, writing):
        target = np.asarray(target)
        if not points:
            points.append(target)
            down.append(False)
            return
        start = points[-1]
        # Quintic time scaling: zero velocity and acceleration at corners.
        # Its peak derivative is 1.875, so duration accounts for peak speed.
        count = max(2, int(np.ceil(1.875 * np.linalg.norm(target - start) / (speed * DT))))
        for t in np.linspace(0, 1, count + 1)[1:]:
            s = 10*t**3 - 15*t**4 + 6*t**5
            points.append(start + s * (target - start))
            down.append(writing)

    for stroke in strokes:
        xy = (np.asarray(stroke) - [offset, 0]) @ rotation_xy.T + np.array(center)
        append_segment([*xy[0], height + lift], False)
        append_segment([*xy[0], height], False)
        for p in xy[1:]:
            append_segment([*p, height], True)
        append_segment([*xy[-1], height + lift], False)
    return np.array(points), np.array(down, dtype=bool)


def pen_rotation(pen_yaw_deg=0.):
    if not np.isfinite(pen_yaw_deg):
        raise ValueError("Rotation angle must be finite")
    return Rotation.from_euler("z", pen_yaw_deg, degrees=True).as_matrix() @ PEN_ROTATION


def track_path(env, points, q0, ik_solver, target_rotation, max_speed=.4):
    """IK every point starting from the first-point solution q0; check joint speed."""
    q = q0
    joints = [q]
    for p in points[1:]:
        q = ik_solver(env, p, q, target_rotation=target_rotation)
        joints.append(q)
    joints = np.array(joints)
    peak_speed = np.max(np.abs(np.diff(joints, axis=0) / DT))
    if peak_speed > max_speed:
        raise ValueError(f"Reference requires {peak_speed:.3f}rad/s; reduce --speed (limit {max_speed})")
    return joints


def plan(env, text, size, center, height, lift, speed, data_dir, seeds=None, ik_solver=solve_ik, rotation_deg=0., pen_yaw_deg=0.):
    points, down = cartesian_path(text, size, center, height, lift, speed, data_dir, rotation_deg)
    target_rotation = pen_rotation(pen_yaw_deg)
    # Solve once with deterministic alternate seeds if the initial pose is poor.
    seeds = seeds if seeds is not None else [[0, 1.3, -1.5, 0, .5, 0], [0, 2, -1, 0, -.8, 0],
             [0, .5, -2, 0, .8, 0]]
    for seed in seeds:
        try:
            q = ik_solver(env, points[0], np.array(seed), target_rotation=target_rotation)
            break
        except ValueError:
            continue
    else:
        raise ValueError("Initial pen pose unreachable; adjust center/height/size")
    joints = track_path(env, points, q, ik_solver, target_rotation)
    return points, joints, down
