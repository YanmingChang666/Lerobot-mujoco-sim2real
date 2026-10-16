"""
用跟 wheelarm_diagnose.py 完全相同的记录方式, 跑一遍 SO-101 已验证的 12_22/DBKN
模型 (README 里给出 error_dis~0.13 的那个), 作为"已知良好"的对照组: 如果它的
跟踪误差曲线在最初几十步也和 WheelArmR 一样单调爬升到 0.1+, 说明这只是正常的
起步过渡, 我们的问题在别处; 如果它一直贴着参考轨迹几乎不涨, 说明 WheelArmR
那边确实存在真正的发散, 不是正常现象。

用法:
    python soarm_diagnose_reference.py --env SOARM101 --x_dim 8 --u_dim 5 \
        --model DBKN --suffix 12_22
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from args import Args
from control.MPC_Controler import MPCController_KESO
from control.TrajectoryGenerator import CartesianTrajectoryGenerator
from models.init_model import init_model
from SOARM101.SOARM101_Env import SOARM101Env

EE_SITE_NAME = "gripperframe"
OUT_DIR = "control/ControlResults/diagnose"


def run_diagnosis(args, max_steps: int | None = 60):
    traj_generator = CartesianTrajectoryGenerator(
        model_path=args.xml_path,
        ee_site_name=EE_SITE_NAME,
        num_joints=args.u_dim,
        idx=0,  # 复用 README 里验证过的桌面平面 Fig8
        time_horizon=60,
        time_steps_per_sec=10,
    )
    cartesian_points, joint_angle_traj, _ = traj_generator.generate(traj_name="Fig8", target_orientation=None)
    state_all_ref = np.hstack([cartesian_points, joint_angle_traj])

    model = init_model(args)
    model.double()
    model.load_state_dict(torch.load(args.output_dir + "/best_model.pt", map_location="cpu"))
    mpc = MPCController_KESO(model, args)
    print("state_full =", mpc.state_full)

    env = SOARM101Env(xml_path=args.xml_path, render_mode=False)
    num_joints = args.u_dim
    H = mpc.H

    env.data.qpos[:num_joints] = joint_angle_traj[mpc.startid]
    mujoco.mj_forward(env.model, env.data)
    state0 = state_all_ref[0] if mpc.startid == 0 else state_all_ref[: mpc.startid + 1]
    state_tensor = torch.DoubleTensor(state0).unsqueeze(0)

    total_frames = len(joint_angle_traj)
    n_steps = total_frames - H if max_steps is None else min(max_steps, total_frames - H)

    log = dict(ref_pos=[], actual_pos=[], qpos=[], u_raw=[], u_clipped=[])

    traj_index = mpc.startid
    for _ in range(n_steps):
        ref_traj_segment = state_all_ref[traj_index + 1 : traj_index + H + 1]
        if mpc.state_full:
            lifted_ref = np.zeros((H, mpc.Nkoopman))
            for i in range(len(ref_traj_segment)):
                lifted_ref[i] = mpc.Psi_o(torch.DoubleTensor(ref_traj_segment[i]).unsqueeze(0)).flatten()
            ref_flat = lifted_ref.flatten().reshape(-1, 1)
        else:
            ref_flat = ref_traj_segment.flatten().reshape(-1, 1)

        z0 = mpc.Psi_o(state_tensor)
        if mpc.args.MPC_type == "mpc":
            p = np.vstack([ref_flat, z0.reshape(-1, 1)])
        else:
            p = np.vstack([ref_flat, z0.reshape(-1, 1), mpc.u_prev.reshape(-1, 1)])

        u_raw, a = mpc.get_control(p)
        mpc.u_prev = a.copy() if mpc.args.MPC_type == "delta_mpc" else mpc.u_prev

        env.data.ctrl[:num_joints] = a[:num_joints]
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)
        actual_next = env._get_state()

        log["ref_pos"].append(cartesian_points[traj_index + 1].copy())
        log["actual_pos"].append(actual_next[:3].copy())
        log["qpos"].append(env.data.qpos[:num_joints].copy())
        log["u_raw"].append(u_raw.copy())
        log["u_clipped"].append(a.copy())

        state_tensor = torch.DoubleTensor(actual_next).unsqueeze(0)
        traj_index += 1

    for k in log:
        log[k] = np.array(log[k])

    os.makedirs(OUT_DIR, exist_ok=True)
    pos_err = np.linalg.norm(log["actual_pos"] - log["ref_pos"], axis=1)
    print("前30步跟踪误差:", np.round(pos_err, 4))

    plt.figure(figsize=(9, 4))
    plt.plot(pos_err)
    plt.title("SO-101 (12_22/DBKN, 已验证参考组) 末端跟踪误差 (m)")
    plt.xlabel("step")
    plt.ylabel("m")
    plt.grid(True)
    out_path = os.path.join(OUT_DIR, "SOARM101_reference_diagnosis.png")
    plt.savefig(out_path, dpi=130)
    print(f"已保存: {out_path}")
    return log


if __name__ == "__main__":
    args = Args()
    run_diagnosis(args)
