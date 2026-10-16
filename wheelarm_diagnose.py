"""
wheelarm_sim2sim.py 的无头(headless)诊断版本：不开 MuJoCo 窗口、不等交互输入，
跑同一套 MPC 闭环逻辑，但每一步都记录更细的诊断量，用来回答"发散到底是因为
Koopman 模型预测错了，还是 MPC/控制环路本身有问题"：

- ref_pos / actual_pos: 参考位置 vs 仿真实际位置 (跟踪误差)
- u_raw / u_clipped: MPC 求解器原始输出 vs 实际施加的裁剪后控制量 (是否在打饱和)
- qpos: 7 个关节实际角度 (是否顶到 model.jnt_range 限位)
- model_pred_next / actual_next: Koopman 模型对下一步状态的预测 vs 仿真步进后的
  真实下一步状态 (这是关键: 如果这两者本身就差很远, 说明是模型预测不准;
  如果模型预测和实际很接近但整体还是跟踪不上参考轨迹, 说明是 MPC/参考轨迹本身
  的问题, 不是模型的问题)

用法:
    python wheelarm_diagnose.py --env WheelArmR --x_dim 10 --u_dim 7 \
        --model IBKN --suffix wheelarm_front_v2 --use_stable True
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无头环境, 只存文件不弹窗
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from args import Args
from control.MPC_Controler import MPCController_KESO
from control.TrajectoryGenerator import CartesianTrajectoryGenerator
from models.init_model import init_model
from WheelArmR.WheelArmR_Env import WheelArmREnv

EE_SITE_NAME = "eeframe"
JOINT_NAMES = [f"Link{i}_R_joint" for i in range(1, 8)]
TRAJ_NAME = "Fig8"
FRONT_PLANE_CENTER = (0.35, -0.25, 0.25)
FRONT_PLANE_SCALE = 0.4
FRONT_PLANE_SEED = np.array([0.0, 0.0, 0.0, -0.6, 0.0, 0.0, 0.0])

OUT_DIR = "control/ControlResults/diagnose"


def run_diagnosis(args, max_steps: int | None = None):
    traj_generator = CartesianTrajectoryGenerator(
        model_path=args.xml_path,
        ee_site_name=EE_SITE_NAME,
        num_joints=args.u_dim,
        joint_names=JOINT_NAMES,
        idx=1,
        time_horizon=60,
        time_steps_per_sec=10,
    )
    traj_generator.traj_scale = FRONT_PLANE_SCALE
    cartesian_points, joint_angle_traj, _ = traj_generator.generate(
        traj_name=TRAJ_NAME,
        target_orientation=None,
        center=FRONT_PLANE_CENTER,
        initial_qpos=FRONT_PLANE_SEED,
    )
    state_all_ref = np.hstack([cartesian_points, joint_angle_traj])  # (N, 10)

    model = init_model(args)
    model.double()
    model.load_state_dict(torch.load(args.output_dir + "/best_model.pt", map_location="cpu"))
    mpc = MPCController_KESO(model, args)

    env = WheelArmREnv(xml_path=args.xml_path, render_mode=False)
    num_joints = args.u_dim
    H = mpc.H

    env.data.qpos[:num_joints] = joint_angle_traj[mpc.startid]
    mujoco.mj_forward(env.model, env.data)
    state0 = state_all_ref[0] if mpc.startid == 0 else state_all_ref[: mpc.startid + 1]
    state_tensor = torch.DoubleTensor(state0).unsqueeze(0)

    total_frames = len(joint_angle_traj)
    n_steps = total_frames - H if max_steps is None else min(max_steps, total_frames - H)

    log = dict(
        ref_pos=[],
        actual_pos=[],
        qpos=[],
        u_raw=[],
        u_clipped=[],
        model_pred_next=[],
        actual_next=[],
    )

    traj_index = mpc.startid
    for _ in range(n_steps):
        ref_traj_segment = state_all_ref[traj_index + 1 : traj_index + H + 1]
        if mpc.state_full:
            # setup_mpc/setup_delta_mpc 的 state_full 分支要求 ref 是 (H, Nkoopman)
            # 的"lifted"参考轨迹 (每个点都过一遍编码器)。
            lifted_ref = np.zeros((H, mpc.Nkoopman))
            for i in range(len(ref_traj_segment)):
                lifted_ref[i] = mpc.Psi_o(torch.DoubleTensor(ref_traj_segment[i]).unsqueeze(0)).flatten()
            ref_flat = lifted_ref.flatten().reshape(-1, 1)
        else:
            # state_full=False 分支只在物理状态的前 x_dim 维(位置+关节角)算误差,
            # ref 直接就是原始参考状态, 不需要过编码器。
            ref_flat = ref_traj_segment.flatten().reshape(-1, 1)

        z0 = mpc.Psi_o(state_tensor)
        if mpc.args.MPC_type == "mpc":
            p = np.vstack([ref_flat, z0.reshape(-1, 1)])
        else:
            p = np.vstack([ref_flat, z0.reshape(-1, 1), mpc.u_prev.reshape(-1, 1)])

        u_raw, a = mpc.get_control(p)
        mpc.u_prev = a.copy() if mpc.args.MPC_type == "delta_mpc" else mpc.u_prev

        # 模型自己对"施加 a 之后下一步状态"的预测 (在 lifted 空间里推一步再解码)。
        # DBKN/IBKN 都是双线性模型, B_total = Bd + sum_j z0[j] * H_hat_j
        # (跟 MPC_Controler.py::linearize_B 是同一个公式, 只是这里用 numpy 而不是
        # CasADi 符号, 因为这里只是拿一个具体的 z0 算一次数值, 不用建优化问题)。
        z0_flat = z0.flatten()
        B_total = mpc.Bd.copy()
        if hasattr(mpc, "H_hat_list"):
            for j in range(mpc.Nkoopman):
                B_total = B_total + z0_flat[j] * mpc.H_hat_list[j]
        z_pred_next = mpc.Ad @ z0_flat + B_total @ a
        if hasattr(mpc, "C"):
            model_pred_next = mpc.C @ z_pred_next
        else:
            # IBKN (use_decoder=True 的可逆网络) 没有固定的 lC, 解码是一个真正学出来
            # 的网络 (net.x_decoder), 不是矩阵乘法, 只能过一遍 forward。
            z_pred_next_t = torch.DoubleTensor(z_pred_next).unsqueeze(0)
            model_pred_next = model.x_decoder(z_pred_next_t).detach().numpy().flatten()

        # 真正施加到仿真里
        env.data.ctrl[:num_joints] = a[:num_joints]
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)
        actual_next = env._get_state()

        log["ref_pos"].append(cartesian_points[traj_index + 1].copy())
        log["actual_pos"].append(actual_next[:3].copy())
        log["qpos"].append(env.data.qpos[:num_joints].copy())
        log["u_raw"].append(u_raw.copy())
        log["u_clipped"].append(a.copy())
        log["model_pred_next"].append(model_pred_next.copy() if model_pred_next is not None else np.zeros(10))
        log["actual_next"].append(actual_next.copy())

        state_tensor = torch.DoubleTensor(actual_next).unsqueeze(0)
        traj_index += 1

    for k in log:
        log[k] = np.array(log[k])

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.model}_{args.suffix}"
    np.savez(os.path.join(OUT_DIR, f"{tag}.npz"), **log, jnt_range=env.model.jnt_range.copy())

    _plot_diagnosis(log, env.model.jnt_range, tag)
    return log


def _plot_diagnosis(log, jnt_range, tag):
    t = np.arange(len(log["ref_pos"]))
    pos_err = np.linalg.norm(log["actual_pos"] - log["ref_pos"], axis=1)
    model_err = np.linalg.norm(log["model_pred_next"] - log["actual_next"], axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(11, 14))

    axes[0].plot(t, pos_err)
    axes[0].set_title("末端位置跟踪误差 |actual - ref| (m)")
    axes[0].set_ylabel("m")
    axes[0].grid(True)

    axes[1].plot(t, np.abs(log["u_raw"]).max(axis=1), label="u_raw (裁剪前, MPC原始解)")
    axes[1].plot(t, np.abs(log["u_clipped"]).max(axis=1), label="u_clipped (实际施加)")
    axes[1].set_title("控制量幅值 (逐步取7个关节里的最大绝对值)")
    axes[1].set_ylabel("rad/s")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, model_err)
    axes[2].set_title("模型单步预测误差 |model_pred_next - actual_next| (10维状态整体L2范数)")
    axes[2].set_ylabel("误差范数")
    axes[2].grid(True)

    for j in range(log["qpos"].shape[1]):
        axes[3].plot(t, log["qpos"][:, j], label=f"joint{j+1}")
    for j in range(log["qpos"].shape[1]):
        lo, hi = jnt_range[j]
        axes[3].axhline(lo, color="gray", linestyle=":", linewidth=0.5)
        axes[3].axhline(hi, color="gray", linestyle=":", linewidth=0.5)
    axes[3].set_title("7个关节角度随时间变化 (虚线=该关节的真实物理限位)")
    axes[3].set_ylabel("rad")
    axes[3].set_xlabel("step")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{tag}_diagnosis.png")
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存: {out_path}")
    print(f"前10步跟踪误差: {pos_err[:10]}")
    print(f"前10步模型单步预测误差: {model_err[:10]}")
    print(f"第0步 u_raw (裁剪前): {log['u_raw'][0]}")
    print(f"第0步 model_pred_next vs actual_next: {log['model_pred_next'][0]} vs {log['actual_next'][0]}")


if __name__ == "__main__":
    args = Args()
    run_diagnosis(args)
