"""
wheel_robot 右臂 (Link1_R..Link7_R, 7自由度) 的 sim2sim 轨迹跟踪验证脚本。

纯 MuJoCo 闭环仿真 (Koopman 建模 + MPC), 不含 ZMQ/真机同步逻辑 —— 这条臂目前没有真机
驱动, 硬保留那部分只会留下用不到的死代码。想跑真机时可以参照 DBKN_KESO.py 的
joint_offsets/ZMQCommunicator 部分单独补上。

用法:
    python wheelarm_sim2sim.py --env WheelArmR --x_dim 10 --u_dim 7 \
        --model DBKN --suffix wheelarm_v0

Link5_R/Link6_R/Link7_R 三个手腕关节的轴线交汇于 eeframe 这一点, 对末端*位置*没有影响
(只影响姿态), 所以下面把 target_orientation 设为 None (姿态不约束), 完全靠
Link1_R~Link4_R 四个关节做位置跟踪, 这是这条臂的运动学结构决定的, 不是 bug。

机器人左右臂分别挂在 Link1_L(y=+0.121)/Link1_R(y=-0.121), 说明 Y 是左右方向, 因此
"正前方"是 +X。零位附近转 Link1_R_joint(Y轴, 类似肩摆)对末端位置几乎没影响(退化),
真正把臂甩到正前方的是肘部 Link4_R_joint(Z轴)先弯开。所以 IK 用 FRONT_PLANE_SEED
(肘部预弯 -0.6 rad)起步求解, 而不是默认的全零位——从零位起步沿轨迹逐点走, dm_control
的 IK 不感知关节限位, 会收敛到数学有解但物理上超限的分支 (已用 model.jnt_range 逐点
验证过)。FRONT_PLANE_CENTER/SCALE 对应 Y-Z 平面(idx=1, X 固定=正前方深度)。
"""

import time

import mujoco
import numpy as np
import torch

from args import Args
from control.config import Config
from control.MPC_Controler import MPCController_KESO
from control.TrajectoryGenerator import CartesianTrajectoryGenerator
from models.init_model import init_model
from WheelArmR.WheelArmR_Env import WheelArmREnv

EE_SITE_NAME = "eeframe"
JOINT_NAMES = [f"Link{i}_R_joint" for i in range(1, 8)]
TRAJ_NAME = "Fig8"  # Fig8 / Rectangle / Heart 都验证过, 在这组参数下全部落在关节限位内
# Y-Z 平面 (idx=1), X=0.35 即"正前方"深度; Y=-0.25/Z=0.25 落在肘部预弯后的可达范围内。
FRONT_PLANE_CENTER = (0.35, -0.25, 0.25)
FRONT_PLANE_SCALE = 0.4
# IK 起始猜测: 肘部 (Link4_R) 先预弯 -0.6 rad, 否则从全零位起步会收敛到 Link2_R/Link4_R
# 超出真实限位的分支 (逐点用 model.jnt_range 验证过, dm_control 自己不会报错/不会拒绝)。
FRONT_PLANE_SEED = np.array([0.0, 0.0, 0.0, -0.6, 0.0, 0.0, 0.0])


class Test:
    """在 MuJoCo 里播放 MPC 闭环轨迹跟踪，结束后保存跟踪误差，不做真机通信。"""

    def __init__(self, path, controller, cartesian_points, joint_angle_traj, num_joints, draw_num, config):
        self.env = WheelArmREnv(path, render_mode=True)
        self.mpc_controller = controller
        self.H = controller.H
        self.config = config

        self.actual_traj = []
        self.u_list = []
        self.cartesian_points = cartesian_points  # (N, 3)
        self.joint_angle_traj = joint_angle_traj  # (N, 7)
        self.state_all_ref = np.hstack([cartesian_points, joint_angle_traj])  # (N, 10)
        self.num_joints = num_joints

        self.traj_index = controller.startid
        self.total_frames = len(joint_angle_traj)
        self.draw_step = max(1, len(cartesian_points) // draw_num)
        print(f"轨迹总长: {self.total_frames}, 绘图采样步长: {self.draw_step}")

        self.home_qpos = None
        key_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            self.home_qpos = self.env.model.key_qpos[key_id].copy()
            self.home_ctrl = self.env.model.key_ctrl[key_id].copy()
        else:
            print("警告: XML中未找到名为 'home' 的 <keyframe>")

        self.return_traj = None
        self.return_index = 0
        self.return_duration = 2.0

    def runBefore(self):
        if self.total_frames > 0:
            self.env.data.qpos[: self.num_joints] = self.joint_angle_traj[self.traj_index]
            mujoco.mj_forward(self.env.model, self.env.data)
            self.state0 = self.state_all_ref[0] if self.traj_index == 0 else self.state_all_ref[: self.traj_index + 1]
            self.state_tensor = torch.DoubleTensor(self.state0).unsqueeze(0)

        print("仿真即将开始...")
        # 画出参考轨迹 (红色小球, 半透明)
        for pt in self.cartesian_points[:: self.draw_step]:
            if self.env.viewer.user_scn.ngeom >= self.env.viewer.user_scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                self.env.viewer.user_scn.geoms[self.env.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.004, 0, 0],
                pos=pt,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.3],
            )
            self.env.viewer.user_scn.ngeom += 1
        if self.env.render_mode:
            self.env.render()
        input("按 Enter 开始...")

    def runFunc(self):
        step_start = time.time()

        if self.traj_index < self.total_frames - self.H:
            self.runMPC()
            mujoco.mj_forward(self.env.model, self.env.data)
            # 画出实际末端位置 (绿色小球, 不透明)
            current_pos = self.actual_traj[self.traj_index - self.mpc_controller.startid][:3]
            if self.env.viewer.user_scn.ngeom <= self.env.viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    self.env.viewer.user_scn.geoms[self.env.viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.004, 0, 0],
                    pos=current_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 0, 1],
                )
                self.env.viewer.user_scn.ngeom += 1
            self.traj_index += 1
        elif self.return_traj is None and self.home_qpos is not None:
            self.config.save_result(
                np.array(self.u_list),
                np.array(self.actual_traj),
                self.cartesian_points[
                    self.mpc_controller.startid + 1 : self.mpc_controller.startid + 1 + len(self.actual_traj)
                ],
                self.joint_angle_traj[
                    self.mpc_controller.startid + 1 : self.mpc_controller.startid + 1 + len(self.actual_traj)
                ],
            )
            print("主轨迹播放完毕，生成回归 Home 的路径...")
            time.sleep(0.5)
            start_qpos = self.joint_angle_traj[-1]
            end_qpos = self.home_qpos[: self.num_joints]
            steps = int(self.return_duration / 0.1)
            self.return_traj = np.linspace(start_qpos, end_qpos, steps)
            self.return_index = 0
        elif self.return_traj is not None and self.return_index < len(self.return_traj):
            self.env.data.qpos[: self.num_joints] = self.return_traj[self.return_index]
            mujoco.mj_forward(self.env.model, self.env.data)
            self.return_index += 1
            self.env.step(self.home_ctrl)
        else:
            self.env.data.qpos[: self.num_joints] = self.home_qpos[: self.num_joints]
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env.step(self.home_ctrl)

        time_until_next_step = self.env.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def runMPC(self):
        ref_traj_segment = self.state_all_ref[self.traj_index + 1 : self.traj_index + self.H + 1]
        lifted_ref = np.zeros((self.H, self.mpc_controller.Nkoopman))
        for i in range(len(ref_traj_segment)):
            lifted_ref[i] = self.mpc_controller.Psi_o(torch.DoubleTensor(ref_traj_segment[i]).unsqueeze(0)).flatten()
        ref_flat = lifted_ref.flatten().reshape(-1, 1)

        z0 = self.mpc_controller.Psi_o(self.state_tensor)
        if self.mpc_controller.args.MPC_type == "mpc":
            p = np.vstack([ref_flat, z0.reshape(-1, 1)])
        else:
            p = np.vstack([ref_flat, z0.reshape(-1, 1), self.mpc_controller.u_prev.reshape(-1, 1)])

        u, a = self.mpc_controller.get_control(p)
        self.mpc_controller.u_prev = u

        s_next = self.runstep(a)
        self.actual_traj.append(s_next)
        self.u_list.append(u)
        if self.mpc_controller.startid > 0:
            self.state0 = np.concatenate((self.state0[1:, :], s_next.reshape(1, -1)), axis=0)
            s_next = self.state0.copy()
        self.state_tensor = torch.DoubleTensor(s_next).unsqueeze(0)

    def runstep(self, action):
        target_velocity = action[: self.env.udim]
        self.env.data.ctrl[: self.env.udim] = target_velocity
        for _ in range(self.env.frame_skip):
            mujoco.mj_step(self.env.model, self.env.data)
        next_state = self.env._get_state()
        if self.env.render_mode:
            self.env.render()
        return next_state

    def is_running(self):
        return self.env.viewer.is_running()

    def run_loop(self):
        self.runBefore()
        while self.is_running():
            self.runFunc()


if __name__ == "__main__":
    args = Args()
    args.device = "cpu"
    if args.env != "WheelArmR":
        raise SystemExit("请加上 --env WheelArmR --x_dim 10 --u_dim 7 运行本脚本")

    MODEL_XML_PATH = args.xml_path
    NUM_JOINTS = args.u_dim

    traj_generator = CartesianTrajectoryGenerator(
        model_path=MODEL_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS,
        joint_names=JOINT_NAMES,
        idx=1,
        time_horizon=60,
        time_steps_per_sec=10,
    )
    traj_generator.traj_scale = FRONT_PLANE_SCALE
    cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
        traj_name=TRAJ_NAME,
        target_orientation=None,
        center=FRONT_PLANE_CENTER,
        initial_qpos=FRONT_PLANE_SEED,
    )

    config = Config(
        suffix=args.suffix,
        env_name=args.env,
        method=args.model,
        use_KEM=False,
        use_nosie=False,
        use_payload=0,
        use_eso=False,
        traj_name=TRAJ_NAME,
    )

    model = init_model(args)
    model.double()
    load_model_path = args.output_dir + "/best_model.pt"
    model.load_state_dict(torch.load(load_model_path, map_location=torch.device("cpu")))
    mpc_controller = MPCController_KESO(model, args)

    try:
        test = Test(
            path=MODEL_XML_PATH,
            controller=mpc_controller,
            cartesian_points=cartesian_points,
            joint_angle_traj=joint_angle_traj,
            num_joints=NUM_JOINTS,
            draw_num=300,
            config=config,
        )
        test.run_loop()
    except KeyboardInterrupt:
        print("仿真程序被用户中断")
