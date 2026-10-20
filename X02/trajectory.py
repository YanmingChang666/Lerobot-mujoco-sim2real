"""X02 right-arm pose IK with redundancy continuity and timed multistroke paths."""
from PIPER.trajectory import plan as timed_plan, PEN_ROTATION
from WheelArmR.trajectory import solve_ik

SEED = [.0603105, .37643317, -.47847961, 1.00896766,
        .58529650, .17548442, -.92446556]


CENTER_SEED = [.83140025, -.07364815, -1.77465745, .67842312,
               .83919808, -.36878368, -.82963078]

def plan(env, text, size, center, height, lift, speed, data_dir, rotation_deg=0., pen_yaw_deg=0.):
    seeds = [SEED] if pen_yaw_deg == 0 else [CENTER_SEED, SEED]
    return timed_plan(env, text, size, center, height, lift, speed, data_dir,
                      seeds=seeds, ik_solver=solve_ik, rotation_deg=rotation_deg, pen_yaw_deg=pen_yaw_deg)
