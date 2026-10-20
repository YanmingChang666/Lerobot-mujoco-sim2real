"""Full-pose IK with a local continuity penalty for the redundant seventh axis."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIPER.trajectory import plan as timed_plan, PEN_ROTATION

SEED = [.92554644, -.41913206, -1.33945726, -1.77813039,
        -.69048797, -.98614513, .53061671]


def solve_ik(env, point, seed, target_rotation=PEN_ROTATION):
    seed = np.asarray(seed)
    def error(q):
        env.data.qpos[env.qadr] = q
        state = env.state()
        rotation = env.data.site_xmat[env.site].reshape(3, 3)
        attitude = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
        return np.r_[state[:3]-point, .15*attitude, .001*(q-seed)]
    bounds = (env.limits[:, 0]+1e-5, env.limits[:, 1]-1e-5)
    result = least_squares(error, np.clip(seed, *bounds), bounds=bounds,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=150)
    residual = error(result.x)
    if np.linalg.norm(residual[:3]) > .0002 or np.linalg.norm(residual[3:6])/.15 > .005:
        raise ValueError(f"Unreachable writing pose: {point}; residual={residual[:6]}")
    return result.x


def plan(env, text, size, center, height, lift, speed, data_dir, rotation_deg=0., pen_yaw_deg=0.):
    return timed_plan(env, text, size, center, height, lift, speed, data_dir,
                      seeds=[SEED], ik_solver=solve_ik, rotation_deg=rotation_deg, pen_yaw_deg=pen_yaw_deg)
