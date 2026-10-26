"""Pen-tip IK for any joint count, plus automatic IK seed search.

Three modes (chosen from the joint count when ik.mode is "auto"):
  full_pose           6+ joints: pen tip position and full orientation (tip down, fixed yaw)
  pen_axis            5 joints:  position + pen axis vertical; rotation about the pen is free
  position_soft_tilt  <=4 joints: position only; pen tilt is minimised softly, not guaranteed

The solvers use MuJoCo's analytic site Jacobian and kinematics only (no dynamics),
so they stay fast for whole-body humanoid models.
"""
from functools import partial

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from PIPER.env import DT
from PIPER.trajectory import PEN_ROTATION, cartesian_path, pen_rotation, track_path

POSITION_TOL = .0002   # m
ANGLE_TOL = .005       # rad
MAX_REFERENCE_SPEED = .4  # rad/s, same limit as the existing robots
DEFAULT_CONTINUITY = {"full_pose": .001, "pen_axis": .001, "position_soft_tilt": .01}
ORIENTATION_WEIGHT = .15


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class _Kinematics:
    """Pen-tip pose and Jacobian for the writing joints of one env (cached per env)."""

    def __init__(self, env):
        self.env = env
        self.jacp = np.zeros((3, env.model.nv))
        self.jacr = np.zeros((3, env.model.nv))
        self.q = None

    def update(self, q, jacobian=False):
        env = self.env
        if self.q is None or not np.array_equal(q, self.q):
            env.data.qpos[env.qadr] = q
            mujoco.mj_kinematics(env.model, env.data)
            self.q = np.array(q, dtype=float)
            self.has_jacobian = False
        position = env.data.site_xpos[env.site].copy()
        rotation = env.data.site_xmat[env.site].reshape(3, 3).copy()
        if not jacobian:
            return position, rotation
        if not self.has_jacobian:
            mujoco.mj_comPos(env.model, env.data)
            mujoco.mj_jacSite(env.model, env.data, self.jacp, self.jacr, env.site)
            self.has_jacobian = True
        return position, rotation, self.jacp[:, env.vadr], self.jacr[:, env.vadr]


def _kinematics(env):
    if getattr(env, "_writing_kinematics", None) is None:
        env._writing_kinematics = _Kinematics(env)
    return env._writing_kinematics


def _solve(env, point, seed, target_rotation, continuity, axes, position_weight,
           orientation_weight, max_nfev):
    """Least squares on [w_p*(p - point), w_o*(R[:,k] - R*[:,k]) for k in axes, c*(q - seed)]."""
    kin = _kinematics(env)
    kin.q = None  # env.data may have been changed (reset/step) since the last solve
    seed = np.asarray(seed, dtype=float)
    point = np.asarray(point, dtype=float)
    n = len(seed)

    def residual(q):
        position, rotation = kin.update(q)
        parts = [position_weight * (position - point)]
        parts += [orientation_weight * (rotation[:, k] - target_rotation[:, k]) for k in axes]
        return np.concatenate(parts + [continuity * (q - seed)])

    def jacobian(q):
        _, rotation, jp, jr = kin.update(q, jacobian=True)
        rows = [position_weight * jp]
        rows += [-orientation_weight * _skew(rotation[:, k]) @ jr for k in axes]
        return np.vstack(rows + [continuity * np.eye(n)])

    low, high = env.limits[:, 0] + 1e-5, env.limits[:, 1] - 1e-5
    result = least_squares(residual, np.clip(seed, low, high), jac=jacobian, bounds=(low, high),
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=max_nfev)
    position, rotation = kin.update(result.x)
    return result.x, np.linalg.norm(position - point), rotation


def solve_full_pose(env, point, seed, target_rotation=PEN_ROTATION, continuity=.001):
    q, error, rotation = _solve(env, point, seed, target_rotation, continuity, (0, 2), 1.,
                                ORIENTATION_WEIGHT, 100)
    angle = np.linalg.norm(Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec())
    if error > POSITION_TOL or angle > ANGLE_TOL:
        raise ValueError(f"Unreachable writing pose {np.round(point, 4)}: "
                         f"position error={error:.5f}m, orientation error={np.degrees(angle):.2f}deg")
    return q


def solve_pen_axis(env, point, seed, target_rotation=PEN_ROTATION, continuity=.001):
    q, error, rotation = _solve(env, point, seed, target_rotation, continuity, (2,), 1.,
                                ORIENTATION_WEIGHT, 100)
    angle = np.arccos(np.clip(rotation[:, 2] @ target_rotation[:, 2], -1, 1))
    if error > POSITION_TOL or angle > ANGLE_TOL:
        raise ValueError(f"Unreachable vertical-pen position {np.round(point, 4)}: "
                         f"position error={error:.5f}m, pen tilt={np.degrees(angle):.2f}deg")
    return q


def solve_position_soft_tilt(env, point, seed, target_rotation=PEN_ROTATION, continuity=.01):
    q, error, _ = _solve(env, point, seed, target_rotation, continuity, (2,), 10., 1e-4, 150)
    if error > POSITION_TOL:
        raise ValueError(f"Unreachable pen position {np.round(point, 4)}: position error={error:.5f}m")
    return q


SOLVERS = {"full_pose": solve_full_pose, "pen_axis": solve_pen_axis,
           "position_soft_tilt": solve_position_soft_tilt}


def make_solver(mode, continuity=None):
    weight = DEFAULT_CONTINUITY[mode] if continuity is None else continuity
    return partial(SOLVERS[mode], continuity=weight)


def _loosened(solver):
    """Same solver with a much weaker pull toward the previous pose, for large coarse steps."""
    if isinstance(solver, partial) and "continuity" in solver.keywords:
        return partial(solver.func, **{**solver.keywords, "continuity": solver.keywords["continuity"] * .01})
    return solver


def _coarse_check(env, points, q0, solver, target_rotation, samples=60):
    """Chain IK over a subsample of the path. Returns None if feasible, else the reason.

    Catches unreachable regions and IK branch flips (a joint jump faster than the
    speed limit allows) for a fraction of the cost of solving every point.
    """
    solver = _loosened(solver)
    step = max(1, len(points) // samples)
    indices = np.unique(np.r_[np.arange(0, len(points), step), len(points) - 1])
    q = q0
    for previous, index in zip(indices[:-1], indices[1:]):
        try:
            q_next = solver(env, points[index], q, target_rotation=target_rotation)
        except ValueError as error:
            return str(error)
        allowed = 1.5 * MAX_REFERENCE_SPEED * (index - previous) * DT
        if np.max(np.abs(q_next - q)) > allowed:
            return f"IK branch flip near {np.round(points[index], 4)}"
        q = q_next
    return None


def search_start(env, points, solver, target_rotation, seeds=(), random_seeds=300, home=None,
                 max_full_attempts=6, rng_seed=0, log=print):
    """Return (q0, joints) for the first start pose whose whole path is feasible.

    Order: configured seeds as written in the YAML, then deterministic random seeds
    within joint limits (home itself first), sorted by distance from home (a closer start means a
    shorter, slower approach). Deterministic, so --mode plan and later runs agree.
    """
    home = np.zeros(len(env.qadr)) if home is None else np.asarray(home, dtype=float)
    reasons = []

    def attempt(q0):
        """-> (joints or None, whether the expensive full-path solve was run)."""
        reason = _coarse_check(env, points, q0, solver, target_rotation)
        if reason is not None:
            reasons.append(reason)
            return None, False
        try:
            return track_path(env, points, q0, solver, target_rotation, max_speed=MAX_REFERENCE_SPEED), True
        except ValueError as error:
            reasons.append(str(error))
            return None, True

    for seed in seeds:
        try:
            q0 = solver(env, points[0], np.asarray(seed, dtype=float), target_rotation=target_rotation)
        except ValueError as error:
            reasons.append(f"configured seed: {error}")
            continue
        joints, _ = attempt(q0)
        if joints is not None:
            return q0, joints
    if random_seeds <= 0:
        raise ValueError(f"No configured IK seed works ({reasons[-1] if reasons else 'no seeds'}); "
                         "adjust center/height/size or set ik.random_seeds > 0")
    log(("Configured IK seeds failed; s" if seeds else "S") +
        f"earching up to {random_seeds} random IK start poses...", flush=True)
    rng = np.random.default_rng(rng_seed)
    low, high = env.limits[:, 0] + 1e-5, env.limits[:, 1] - 1e-5
    starts = []
    # Home first: redundant joints (e.g. rotation about the pen) then stay near home.
    for seed in np.vstack([np.clip(home, low, high), rng.uniform(low, high, size=(random_seeds, len(low)))]):
        try:
            q0 = solver(env, points[0], seed, target_rotation=target_rotation)
        except ValueError:
            continue
        if not any(np.allclose(q0, other, atol=.02) for other in starts):
            starts.append(q0)
    if not starts:
        raise ValueError("The first pen pose is unreachable from every sampled start; move the paper "
                         "(--center/--height), shrink the text, or change --pen-yaw-deg")
    starts.sort(key=lambda q: np.linalg.norm(q - home))  # closest overall posture first
    full_attempts = 0
    for q0 in starts[:40]:
        joints, ran_full_path = attempt(q0)
        if joints is not None:
            log(f"IK start found among {len(starts)} distinct first-point solutions; "
                f"max joint distance from home {np.degrees(np.max(np.abs(q0 - home))):.1f}deg.\n"
                f"  Put it first in ik.seeds of the robot YAML to skip the search:\n"
                f"  - {np.round(q0, 6).tolist()}", flush=True)
            return q0, joints
        # Only starts that passed the cheap coarse check use up the full-path budget.
        if ran_full_path:
            full_attempts += 1
            if full_attempts >= max_full_attempts:
                break
    common = max(set(reasons), key=reasons.count) if reasons else "unknown"
    raise ValueError(f"No start pose gives a feasible whole path (most common reason: {common}); "
                     "move the paper, shrink the text, reduce --speed or change --pen-yaw-deg")


def spec_planner(spec):
    mode = spec.resolved_ik_mode
    solver = make_solver(mode, spec.continuity_weight)

    def plan(env, text, size, center, height, lift, speed, data_dir, rotation_deg=0., pen_yaw_deg=0.):
        if mode != "full_pose" and pen_yaw_deg:
            raise ValueError(f"{spec.name} uses IK mode '{mode}' ({spec.dof} joints): pen yaw is not "
                             "controllable, so --pen-yaw-deg must be 0. Use --rotation-deg to rotate text.")
        points, down = cartesian_path(text, size, center, height, lift, speed, data_dir, rotation_deg)
        target_rotation = pen_rotation(pen_yaw_deg)
        _, joints = search_start(env, points, solver, target_rotation, seeds=spec.seeds,
                                 random_seeds=spec.random_seeds, home=np.radians(spec.home))
        return points, joints, down

    plan.ik_mode = mode
    return plan
