"""Four-axis position IK with soft pen-direction and continuity objectives.

A four-joint arm cannot generally meet arbitrary 6D poses. Tool yaw is not
commandable; the actual tilt is measured by the common writing runner.
"""
import numpy as np
from scipy.optimize import least_squares
from PIPER.trajectory import plan as timed_plan, PEN_ROTATION

SEEDS = [[-.1,1.19,.44,-1.54],[-.8,.6,.5,-1.2],[-1.,.3,-.5,-1.5],[.5,.5,1.,-1.]]


def solve_ik(env, point, seed, target_rotation=PEN_ROTATION):
    seed=np.asarray(seed)
    def residual(q):
        env.data.qpos[env.qadr]=q
        state=env.state()
        axis=env.data.site_xmat[env.site].reshape(3,3)[:,2]
        return np.r_[10*(state[:3]-point), .0001*(axis-target_rotation[:,2]), .01*(q-seed)]
    bounds=(env.limits[:,0]+1e-5,env.limits[:,1]-1e-5)
    result=least_squares(residual,np.clip(seed,*bounds),bounds=bounds,
                         xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=200)
    error=residual(result.x)
    if np.linalg.norm(error[:3])/10 > .0002:
        raise ValueError(f'Unreachable four-axis pen position: {point}')
    return result.x


def plan(env,text,size,center,height,lift,speed,data_dir,rotation_deg=0.,pen_yaw_deg=0.):
    if pen_yaw_deg != 0:
        raise ValueError('x02cTpose has four active joints and a fixed wrist; --pen-yaw-deg is unsupported. Use --rotation-deg for text.')
    return timed_plan(env,text,size,center,height,lift,speed,data_dir,
                      seeds=SEEDS,ik_solver=solve_ik,rotation_deg=rotation_deg)
