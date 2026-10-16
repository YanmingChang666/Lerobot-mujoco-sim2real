"""Timed multistroke trajectories with full-pose IK at the pen tip."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from control.hanzi_loader import text_to_strokes
from .env import DT

PEN_ROTATION = np.diag([-1., 1., -1.])  # pen local +Z points down


def solve_ik(env, point, seed):
    def error(q):
        env.data.qpos[env.qadr] = q
        state = env.state()
        rot = env.data.site_xmat[env.site].reshape(3, 3)
        orientation = Rotation.from_matrix(PEN_ROTATION @ rot.T).as_rotvec()
        return np.r_[state[:3] - point, 0.15 * orientation]

    bounds = (env.limits[:, 0] + 1e-5, env.limits[:, 1] - 1e-5)
    result = least_squares(error, np.clip(seed, *bounds), bounds=bounds,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=100)
    residual = error(result.x)
    if np.linalg.norm(residual[:3]) > 0.0002 or np.linalg.norm(residual[3:]) / .15 > .005:
        raise ValueError(f"Unreachable pen pose {point}: position error={np.linalg.norm(residual[:3]):.5f}m")
    return result.x


def plan(env, text, size, center, height, lift, speed, data_dir):
    if not text.strip() or min(size, lift, speed) <= 0:
        raise ValueError("Text must be nonempty; size, lift and speed must be positive")
    strokes = text_to_strokes(text, size_m=size, data_dir=str(data_dir))
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

    offset = (len(text) - 1) * size * 1.2 / 2
    for stroke in strokes:
        xy = np.asarray(stroke) + np.array(center) - [offset, 0]
        append_segment([*xy[0], height + lift], False)
        append_segment([*xy[0], height], False)
        for p in xy[1:]:
            append_segment([*p, height], True)
        append_segment([*xy[-1], height + lift], False)
    # Solve once with deterministic alternate seeds if the initial pose is poor.
    seeds = [[0, 1.3, -1.5, 0, .5, 0], [0, 2, -1, 0, -.8, 0],
             [0, .5, -2, 0, .8, 0]]
    for seed in seeds:
        try:
            q = solve_ik(env, points[0], np.array(seed))
            break
        except ValueError:
            continue
    else:
        raise ValueError("Initial pen pose unreachable; adjust center/height/size")
    joints = [q]
    for p in points[1:]:
        q = solve_ik(env, p, q)
        joints.append(q)
    joints = np.array(joints)
    peak_speed = np.max(np.abs(np.diff(joints, axis=0) / DT))
    if peak_speed > .4:
        raise ValueError(f"Reference requires {peak_speed:.3f}rad/s; reduce --speed (limit 0.4)")
    return np.array(points), joints, np.array(down, dtype=bool)
