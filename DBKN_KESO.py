"""
"DBKN + KESO" 闭环仿真主实验脚本：训练好的 DBKN(双线性 Koopman)模型
+ MPC + KESO(卡尔曼滤波 + 扩张状态观测器)，跑一次完整的
"生成参考轨迹 -> MuJoCo 闭环仿真跟踪 -> (可选)通过 ZMQ 同步给真机" 流程。

数据/控制流：
    TrajectoryGenerator 生成参考轨迹(笛卡尔点 + 关节角)
        -> Test.runMPC(): 取未来 H 步参考、构造 CasADi 参数、调用 MPC 求解器
        -> Test.runstep(): 把 MPC 算出的控制量喂给 MuJoCo 仿真步进一步，
           同时按需注入噪声/负载扰动，并用 KF/ESO 修正状态估计
        -> 每步都把当前仿真关节角通过 ZMQ 广播给 so101_real.py 驱动真机

本文件直接运行(__main__)时是"only-sim"验证实验，ZMQ 只是把仿真结果广播出去，
不强制要求真机在线；README 里"部署到真实机械臂"一节讲的就是同时开着这个
脚本和 so101_real.py 联动的用法。write_demo.py 写汉字用的也是这个文件里
的 Test 类。

============================================================
【在写字流程中的角色】
============================================================
write_demo.py 只 import 本文件的三样东西:
    Test           闭环仿真播放器 (本文件最重要的类)
    joint_offsets  仿真零位 -> 真机零位的标定偏移 (度)
    sim_to_real    角度换算函数
本文件自己的 __main__ 是"画单笔图形 (Heart/Fig8...)"的实验入口, 写字时不会执行。

Test 每一帧 (约 50Hz, 0.02s) 的执行顺序 (runFunc):
    ┌ 主轨迹阶段  traj_index < 总点数 - H
    │     runMPC(): 取未来 H 步参考 -> Koopman 提升 -> MPC 求解 -> runstep() 步进 MuJoCo
    ├ 阶段2       主轨迹结束 (只执行一次): 保存结果, 生成"回 Home"的关节插值轨迹
    ├ 阶段3       逐帧播放回 Home 轨迹
    └ 阶段4       保持 Home 姿态
    每帧结尾 (无论哪个阶段): 仿真关节角(度) - joint_offsets -> ZMQ 发给真机, 并补睡眠凑满 0.02s

状态向量约定 (8 维): [末端 x, y, z (3 维笛卡尔位置), 5 个关节角 (弧度)]
    参考状态 state_all_ref = [笛卡尔参考点 | 关节角参考], 与 SOARM101Env._get_state() 的顺序一致。

读写字轨迹时要注意的两点:
    - 主轨迹阶段在 traj_index < N - H 时就结束了: 最后 H 个参考点只作为 MPC 的预测窗口,
      不会被"当作当前步"真正走到 (MPCController 里 H=10, 即最后 10 个点, 约 0.2 秒)。
    - 回 Home 阶段是直接在关节空间插值并直接写 qpos, 不经过 MPC/Koopman。
"""

# math    : 弧度 -> 角度 (math.degrees)
# time    : 帧率控制 (time.sleep / time.time)
# mujoco  : 直接操作 MuJoCo 模型与数据 (qpos、xfrc_applied、mj_step ...)
# numpy   : 数组运算
# torch   : Koopman 网络输入输出 (DoubleTensor)
import math
import time

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import mujoco
import numpy as np
import torch

# Args                         : 命令行/默认参数
# Config                       : 实验配置 + 结果保存
# MPCController_KESO           : Koopman + MPC (+ UKF/KESO) 控制器 (数学细节在 control/MPC_Controler.py)
# CartesianTrajectoryGenerator : 轨迹生成 + IK (写字用的就是它的 generate_multi_stroke)
# init_model                   : 根据 args 构建网络
# SOARM101Env                  : MuJoCo 机械臂环境 (Gymnasium 风格, 提供 step / _get_state / render ...)
# ZMQCommunicator              : ZMQ 发布端
from args import Args
from control.config import Config
from control.MPC_Controler import MPCController_KESO
from control.TrajectoryGenerator import (
    CartesianTrajectoryGenerator,
    CartesianTrajectoryGenerator_pinocchio,
)
from models.init_model import init_model
from SOARM101.SOARM101_Env import SOARM101Env
from utility.ZMQ import ZMQCommunicator

# --- 修改后的主仿真类 ---
# 仿真关节零位和真机标定零位之间的差值(度)，真机指令角度 = 仿真角度 - offset。
# 每次重新 lerobot-calibrate 真机后都要重新走一遍 README 里的标定流程算出这组值，
# 详见 README 的"标定 joint_offsets"一节。
# 6 个值的顺序: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], 单位 度。
# 换算关系: 真机目标角 = 仿真角 - offset (见 sim_to_real)。
# 当前 5 个关节都是 0, 表示完全信任 lerobot-calibrate 产生的标定零点。
joint_offsets = [
    0,  # shoulder_pan: 2026-09-22 重新 lerobot-calibrate 后, 信任标定零点, 残差判断为人工摆姿态误差而非真实偏移
    0,  # shoulder_lift
    0,  # elbow_flex
    0,  # wrist_flex
    0,  # wrist_roll
    0,  # gripper: MPC 不控制这个关节
]


# 逐关节做 sim - offset, 返回 list。输入输出都是"度"。
# write_demo.py 里 sync_real_to_sim_start 做的是反向换算: sim = real + offset。
def sim_to_real(q_sim_deg, offsets):
    """MuJoCo 角度 (度) -> 真实机器人指令角度 (度)"""
    return [sim - off for sim, off in zip(q_sim_deg, offsets)]


# Test: 闭环仿真播放器。构造时创建 MuJoCo 环境; run_loop() = runBefore() 一次 + 反复 runFunc()。
class Test:
    """
    闭环仿真播放器：驱动 MuJoCo 可视化窗口，每一帧调用一次 MPC 算出控制量、
    步进仿真、并把结果通过 ZMQ 广播出去。整个类本质上是一个"仿真主循环 +
    MPC 控制器 + 状态估计器 + 真机通信"的胶水代码，具体的 Koopman/MPC/KESO
    数学都在 control/MPC_Controler.py 里。
    """

    def __init__(
        self,
        path,
        Controller,
        communicator,
        cartesian_points,
        joint_angle_traj,
        num_joints,
        draw_num,
        config,
    ):
        """
        初始化参数
        :param path: XML 模型路径
        :param communicator: 通信器实例
        :param cartesian_points: 笛卡尔空间轨迹点 (N, 3)
        :param joint_angle_traj: 关节空间轨迹 (N, num_joints)
        :param num_joints: 机器人的关节数量 (例如 5, 6, 7)
        """

        # ---- 基本对象 ----
        self.path = path
        self.communicator = communicator
        self.config = config
        # 闭环仿真环境; render_mode=True 会弹出 MuJoCo 可视化窗口 (需要图形桌面)。
        self.env = SOARM101Env(path, render_mode=True)
        # 控制相关
        # ---- 控制器 ----
        self.mpc_controller = Controller
        # H = MPC 预测时域 (向前看的步数), 取自控制器: MPCController 里 self.H = 10, 与训练时 pre_length=10 的开环窗口一致。
        self.H = Controller.H
        # 保存轨迹数据
        # ---- 记录区 ----
        # actual_traj: 每步仿真后的真实状态 (8 维)
        # u_list: 每步 MPC 输出的控制量; 轨迹播完后一起交给 config.save_result 保存。
        self.actual_traj = []
        self.u_list = []
        # ---- 参考轨迹 (来自 generate_multi_stroke 的 xyz 与关节角) ----
        self.cartesian_points = cartesian_points  # (N, 3)
        self.joint_angle_traj = joint_angle_traj  # (N, 5)
        # 参考状态 = [笛卡尔位置, 关节角] 拼接，和 SOARM101Env._get_state() 的
        # 状态定义顺序一致，方便直接用它构造 MPC 的参考轨迹/初始状态。
        self.state_all_ref = np.hstack([cartesian_points, joint_angle_traj])
        # 受控关节数 (5)。
        self.num_joints = num_joints

        # 轨迹播放进度计数器
        # startid 对时序类网络(Koopformer/LSTM)非零，因为这类模型需要先攒够
        # seq_len-1 步历史才能开始编码，从这个索引开始播放才有完整历史窗口；
        # 线性/双线性网络(DBKN 等)startid=0，从第一个点直接开始。
        # traj_index: 当前参考轨迹的播放位置 (第几个点)。DBKN 这类非时序模型 startid=0, 从第一个点开始。
        self.traj_index = Controller.startid
        # 参考轨迹总点数 N (写字时 = 落笔点 + 抬笔/平移点)。
        self.total_frames = len(joint_angle_traj)

        # 预计算采样步长 (防止点太多卡顿)
        # 保证屏幕上最多显示 300-500 个红点
        # 红点显示的抽稀步长: 每隔 draw_step 个点画一个, 使屏幕上最多约 draw_num 个点, 避免渲染卡顿。
        self.draw_step = max(1, len(cartesian_points) // draw_num)
        print(f"轨迹总长: {self.total_frames}, 绘图采样步长: {self.draw_step}")
        # --- 新增：读取 Home Keyframe ---
        # ---- Home 姿态: 轨迹播完后回到这里; 来自 XML 里名为 home 的 <keyframe> ----
        self.home_qpos = None
        try:
            # 1. 获取名为 "home" 的关键帧 ID
            # 按名字查 keyframe 的 id; key_qpos / key_ctrl 分别是该关键帧的关节位置和执行器控制量。
            key_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_KEY, "home")

            # 2. 如果 ID 有效 (>=0)，则读取对应的 qpos 数据
            if key_id >= 0:
                # model.key_qpos 是一个 (nkey, nq) 的数组
                self.home_qpos = self.env.model.key_qpos[key_id].copy()
                self.home_ctrl = self.env.model.key_ctrl[key_id].copy()
                print(f"已加载 'home' 姿态: {self.home_qpos}")
            else:
                print("警告: XML中未找到名为 'home' 的 <keyframe>")
        except Exception as e:
            print(f"读取 Keyframe 出错: {e}")

        # --- 2. 回归轨迹相关的变量 ---
        # ---- 回 Home 的状态变量 ----
        # return_traj: 要播放的关节插值轨迹(None 表示还没生成); return_index: 播放进度; return_duration: 期望时长(秒)。
        self.return_traj = None  # 存储生成的回归路径
        self.return_index = 0  # 回归播放进度
        self.return_duration = 2.0  # 回归过程耗时 2秒

    def runBefore(self):
        """
        在仿真循环开始前执行一次：把仿真复位到轨迹起点、画出参考轨迹的
        静态红点、把起始姿态发给真机，并等待用户按 Enter 确认后再正式开始
        （给操作者一个"肉眼确认仿真和真机对齐"的机会，尤其是首次上真机时）。
        """
        # 将机器人复位到轨迹的起始位置
        # print(self.traj_index)
        # ---- 把仿真复位到轨迹起点 (直接写 qpos, 属于"瞬移"; 真机端的对齐已由 write_demo 的 sync_real_to_sim_start 完成) ----
        if self.total_frames > 0:
            self.env.data.qpos[: self.num_joints] = self.joint_angle_traj[self.traj_index]
            mujoco.mj_forward(self.env.model, self.env.data)
            # state0 = MPC 的当前状态输入: startid=0 时是单个 8 维状态; 时序网络(startid>0)时是前 startid+1 步的历史窗口。
            if self.traj_index == 0:
                self.state0 = self.state_all_ref[0]
            else:
                self.state0 = self.state_all_ref[: self.traj_index + 1]
            # 加 batch 维 -> (1, 8) 或 (1, 历史步数, 8), 网络需要的输入形状。
            self.state_tensor = torch.DoubleTensor(self.state0).unsqueeze(0)

        print("仿真即将开始...")
        # --- 1. 绘制静态轨迹 (红色小球) ---
        # 注意：必须每帧都画，因为 viewer.sync() 会清空 user_scn
        # 把参考轨迹画成半透明红点 (每隔 draw_step 个画一个): 写字时就是字的形状 + 抬笔转移的空中路径。
        for pt in self.cartesian_points[:: self.draw_step]:
            # 安全检查：如果 geometry 满了就不画了
            if self.env.viewer.user_scn.ngeom >= self.env.viewer.user_scn.maxgeom:
                break

            mujoco.mjv_initGeom(
                self.env.viewer.user_scn.geoms[self.env.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.002, 0, 0],  # 2mm 红球
                pos=pt,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.3],  # 半透明红
            )
            self.env.viewer.user_scn.ngeom += 1

        if self.env.render_mode:
            self.env.render()
        # 取前 6 个关节 (5 个受控 + gripper) 当前值 (弧度)。
        sim_joint_rad = self.env.data.qpos[:6].copy()
        # 将弧度转换为角度
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]
        # 换算成真机指令角: 真机角 = 仿真角 - offset。
        q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)
        # 先把起始姿态发给真机一次, 便于对照检查。
        self.communicator.send_data(q_real_target_deg)
        # 阻塞等待用户确认: 肉眼确认仿真与真机姿态一致、周围无障碍后再开始; 开始后随时可 Ctrl+C。
        input("按 Enter 开始...")

    def runFunc(self):
        """
        每一帧都会被调用的函数：按当前播放阶段(主轨迹跟踪/生成回归路径/
        播放回归路径/保持 Home)分发到对应逻辑，并在每帧结束时把当前仿真
        关节角通过 ZMQ 广播给真机，同时做简单的帧率控制(维持约 50Hz)。
        """
        # --- 代码：重力补偿 ---
        # `data.qfrc_bias` 存储了由重力、科里奥利力等产生的偏置力矩。
        # 对于静止或慢速运动的机器人，它主要就是重力力矩
        # 记录帧起点时间, 帧末用来补睡眠、凑满 0.02s。
        step_start = time.time()

        # self.env.data.qfrc_applied[:] =  self.env.data.qfrc_bias[:]
        # --- 2. 机器人运动控制 ---
        # 如果轨迹还没播完
        # 提前 H 步结束主轨迹阶段，是因为 runMPC 要用 [traj_index+1, traj_index+H+1]
        # 这一段作为参考窗口，如果不留出这段余量，播放到轨迹末尾时参考窗口会越界。
        # ==== 阶段1: 主轨迹跟踪 ====
        # 条件: 还剩至少 H 个参考点可作为预测窗口。每次进入: 一次 MPC + 一次仿真步进 + 画点 + traj_index+1。
        # 因此最后 H 个点不会被当作"当前目标"走到。
        if self.traj_index < self.total_frames - self.H:
            # 核心一步: 取参考窗口 -> 求解 MPC -> 步进仿真 (见 runMPC / runstep)。
            self.runMPC()
            # 前向动力学计算
            mujoco.mj_forward(self.env.model, self.env.data)
            # 绘制当前目标点 (绿色大球)
            # current_pos = self.cartesian_points[self.traj_index]
            # 取刚走完这一步后的末端实际位置 (actual_traj 每项的前 3 维 = 笛卡尔 x, y, z), 用绿点标出, 与红色参考点对照可直观看跟踪误差。
            current_pos = self.actual_traj[self.traj_index - self.mpc_controller.startid][:3]
            if self.env.viewer.user_scn.ngeom <= self.env.viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    self.env.viewer.user_scn.geoms[self.env.viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.002, 0, 0],  # 1cm 绿球
                    pos=current_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 0, 1],  # 不透明绿
                )
                self.env.viewer.user_scn.ngeom += 1

            # 进度 +1
            # 播放进度 +1: 下一帧参考窗口整体后移一格。
            self.traj_index += 1

            # 如果觉得播放太快，可以在这里加一点点延时，但通常不建议在 GUI 线程 sleep 太久
            # time.sleep(0.002)

        # --- 阶段 2: 主轨迹刚结束，生成回归路径 (只执行一次) ---
        # ==== 阶段2: 主轨迹刚结束 (return_traj 仍为 None, 所以只会执行一次) ====
        elif self.return_traj is None and self.home_qpos is not None:
            # 主轨迹刚播完，先把这次实验的完整跟踪结果落盘 (config.save_result)，
            # 供 control/config.py 里的 ConfigGenerator 之后批量汇总画图用。
            # 保存本次实验: 控制量 u_list、实际轨迹 actual_traj、对应的笛卡尔参考、对应的关节角参考。
            # 参考轨迹按 [startid+1 : startid+1+len(actual_traj)] 截取, 与实际轨迹逐点对齐
            # (因为第 k 步控制之后得到的状态对应参考的第 k+1 个点)。
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
            print(len(self.actual_traj))
            time.sleep(0.5)
            # 回 Home 的起点取参考轨迹最后一个关节角 (不是仿真里的实际角度; 两者差距很小时无所谓)。
            start_qpos = self.joint_angle_traj[-1]  # 当前位置
            end_qpos = self.home_qpos[: self.num_joints]  # 目标位置

            # 计算需要多少帧 (假设 timestep=0.002, 持续 2秒 => 1000帧)
            # 插值帧数 = 2.0 / 0.1 = 20 帧。而 runFunc 每帧约 0.02s, 所以实际回 Home 只需约 0.4 秒,
            # 不是注释里写的"2 秒 / 1000 帧" (那是旧值遗留)。
            # 若想让收尾动作更慢更安全, 把 0.1 改成 0.02 (-> 100 帧 ≈ 2 秒)。
            steps = int(self.return_duration / 0.1)

            # 生成插值轨迹 (Shape: [steps, num_joints])
            # 使用 linspace 生成平滑过渡
            # 如果想要非线性平滑(ease-in-out)，可以使用 cos 函数处理 steps
            # 注意：回归 Home 走的是关节空间直接线性插值，不再经过 MPC/Koopman，
            # 因为这一段只是"收尾动作"，不需要跟踪精度，直接插值更简单可靠。
            # 关节空间逐关节线性插值 (steps, num_joints): 每个关节同步匀速地从终点姿态走到 Home。
            self.return_traj = np.linspace(start_qpos, end_qpos, steps)

            self.return_index = 0  # 准备开始播放回归

        # --- 阶段 3: 播放回归轨迹 ---
        # ==== 阶段3: 逐帧播放回 Home 轨迹, 每帧取一行, 直接写入 qpos (运动学瞬移, 不是动力学跟踪) ====
        elif self.return_traj is not None and self.return_index < len(self.return_traj):
            print("回归 Home...")
            # 设置回归过程中的关节角度
            self.env.data.qpos[: self.num_joints] = self.return_traj[self.return_index]
            mujoco.mj_forward(self.env.model, self.env.data)
            # 这里不再绘制绿球，因为已经在“回家”路上了
            self.return_index += 1
            s_next, _, _, _, _ = self.env.step(self.home_ctrl)  # (8,)
        # ==== 阶段4: 全部结束, 持续把姿态保持在 Home (没有 home 关键帧则停在轨迹终点) ====
        # --- 阶段 4: 全部结束，保持 Home 姿态 ---
        else:
            if self.home_qpos is not None:
                self.env.data.qpos[: self.num_joints] = self.home_qpos[: self.num_joints]
            else:
                # 如果没有 home，就停在轨迹终点
                self.env.data.qpos[: self.num_joints] = self.joint_angle_traj[-1]

            mujoco.mj_forward(self.env.model, self.env.data)
            s_next, _, _, _, _ = self.env.step(self.home_ctrl)  # (8,)

        # --- 3. 通信器逻辑 (保留你的原始逻辑) ---
        # 不管当前处于哪个阶段(主轨迹/回归/Home)，每帧都把仿真关节角转成度数、
        # 减去标定偏移量，通过 ZMQ 广播出去，真机端 so101_real.py 只是被动
        # "追"这个目标角度(单向开环，不读真机反馈，见 README 安全须知)。
        # ---- 每帧结尾的通信 (四个阶段共用): 仿真关节角 -> 度 -> 减标定偏移 -> ZMQ 发给真机 ----
        sim_joint_rad = self.env.data.qpos[:6].copy()
        # 将弧度转换为角度
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]
        # 真机目标角 = 仿真角 - joint_offsets。
        q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)
        # 发布到 ZMQ :5555, so101_real.py 订阅后驱动真机; 真机端另有单次 10° 的位移限幅。回读真机状态不在这里发生 (开环)。
        self.communicator.send_data(q_real_target_deg)

        # 简单的帧率控制：目标每帧耗时 0.02s(50Hz)，如果本帧计算/渲染更快就
        # 补齐睡眠时间，避免仿真播放速度和真实物理时间不成比例。
        # 本帧已用时间 = now - step_start; 不足 0.02s 的部分补睡眠 -> 约 50Hz。
        time_until_next_step = 0.02 - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # time.sleep(0.01)  # 控制发送频

    def runMPC(self):
        """
        单步 MPC 控制：取未来 H 步参考轨迹、把当前状态编码进 Koopman 观测
        空间、组装 CasADi 求解器需要的参数 p，求解得到控制量后步进仿真，
        并维护时序类网络需要的历史窗口 state0。
        """
        # --- 2. MPC 控制计算 ---
        # 参考窗口: 取下标 [traj_index+1, traj_index+H] 共 H 个未来参考状态 (每个 8 维)。
        # "+1" 是因为现在状态对应第 traj_index 个点, MPC 要让下一步及之后逐步靠近后面的点。
        # 写字时这些参考点里可能混有抬笔/平移段 (z 不同), 控制器并不区分, 都当作普通参考。
        ref_traj_segment = self.state_all_ref[self.traj_index + 1 : self.traj_index + self.H + 1]
        # 两种参考表示: 误差在 Koopman 提升空间里算(state_full=True), 或直接在原始状态空间里算。
        if self.mpc_controller.state_full:
            # state_full=True(可逆网络等) 时参考轨迹要先逐点编码到完整的
            # Koopman 观测空间，因为 MPC 的代价函数是在观测空间里定义误差的。
            lifted_ref = np.zeros((self.H, self.mpc_controller.Nkoopman))
            for i in range(len(ref_traj_segment)):
                lifted_ref[i] = self.mpc_controller.Psi_o(
                    torch.DoubleTensor(ref_traj_segment[i]).unsqueeze(0)
                ).flatten()
            ref_flat = lifted_ref.flatten().reshape(-1, 1)
        else:
            # 否则直接用原始状态(笛卡尔位置+关节角)作为参考，代价函数只在
            # 物理量意义明确的前几维上定义误差。
            ref_flat = ref_traj_segment.flatten().reshape(-1, 1)

        # 把当前状态编码到 Koopman 观测空间 (提升): 在这个空间里系统动力学近似线性/双线性, MPC 才好解。
        z0 = self.mpc_controller.Psi_o(self.state_tensor)  # (32,1)
        # 保存提升后的状态, 给 runstep 里的卡尔曼滤波/ESO 用 (写字演示里这两个都关闭)。
        self.z_last = z0
        # 不同 MPC_type 对应的 CasADi solver 需要的参数向量 p 的组成不同
        # (delta_mpc 多一个 u_prev，用于把决策变量的"增量"累加成绝对控制量)。
        # 组装 CasADi 求解器的参数向量 p: [参考窗口 | 当前提升状态]; delta_mpc 额外附带上一时刻控制量 u_prev (决策变量是控制增量)。
        if self.mpc_controller.args.MPC_type == "mpc":
            p = np.vstack([ref_flat, z0.reshape(-1, 1)])
        if self.mpc_controller.args.MPC_type == "delta_mpc":
            p = np.vstack([ref_flat, z0.reshape(-1, 1), self.mpc_controller.u_prev.reshape(-1, 1)])

        # b. 计算控制输入
        # 求解 MPC (control/MPC_Controler.py 的 get_control)。返回 (u, a):
        #     u = 优化解的第 1 步 + ESO 补偿 u_eso + u_prev
        #         (delta_mpc 下优化变量是增量, 加上 u_prev 才是绝对控制量; 未启用 ESO 时 u_eso 恒为 0)
        #     a = clip(u, -2, 2), 才是真正写进仿真的指令
        # 两者只差一次限幅, ESO 补偿在两者里都有。
        # 注意: 求解时 NLP 里并没有 u 的约束, 限幅是求解之后才做的 —— 优化器不知道 ±2 的上限。
        u, a = self.mpc_controller.get_control(p)
        # 无条件把"未限幅的 u"存为下一步的 u_prev (注意是 u 不是 a)。
        # - delta_mpc(默认): 正是想要的行为, 下一步的增量累加在上一步控制量上。
        #   若某步发生了限幅, 这里存的仍是限幅前的值 (get_control 里刚存的限幅后的 a 被覆盖掉了)。
        # - mpc: get_control 里同样会把 u_prev 加到结果上, 绝对控制量会因此被逐步累加, 这与
        #   get_control 注释里"mpc 模式下 u_prev 恒为 0"的设想不一致。使用 --MPC_type mpc 前请先核对。
        self.mpc_controller.u_prev = u  # 保存当前控制以备下次使用

        # --- 3. 应用控制并步进仿真 ---
        # 把指令喂给 MuJoCo 步进一次, 返回新的 8 维状态 [x, y, z, 5 个关节角]。
        s_next = self.runstep(a, self.z_last)  # (8,)
        # 存入实际轨迹/控制量记录, 用于结束时保存。
        self.actual_traj.append(s_next)
        self.u_list.append(u)
        # 处理时间序列：时序类网络需要维护一个滑动窗口(丢最老的一步、
        # 加入最新一步)，非时序网络(startid=0)直接用最新状态即可。
        # 时序网络需要历史窗口: 滑动窗口 (丢掉最老一步, 追加最新一步)。DBKN 的 startid=0, 此分支不执行。
        if self.mpc_controller.startid > 0:
            self.state0 = np.concatenate((self.state0[1:, :], s_next.reshape(1, -1)), axis=0)  # (5, 8)
            s_next = self.state0.copy()

        # 更新下一次 MPC 的输入状态 (加 batch 维)。
        self.state_tensor = torch.DoubleTensor(s_next).unsqueeze(0)  # (1,8) / (1, 5, 8)

    def runstep(self, action, z_last):
        """
        把 MPC 算出的控制量真正喂给 MuJoCo 仿真步进一步，并按 config 里的
        开关注入负载力/观测噪声，再用 KF/ESO 修正一次状态估计（如果开启）。
        这是 README 里"抗负载"/"抗噪声"实验的具体实现位置。
        """
        # config.use_nosie
        # 本函数把 MPC 指令真正作用到 MuJoCo。写字演示中 use_payload=0、use_nosie=False、use_KEM=False、use_eso=False,
        # 所以实际只执行: 写 ctrl -> 清零末端外力 -> mj_step -> 取状态 -> 渲染。其余分支用于抗扰动实验。
        target_velocity = action[: self.env.udim]
        # 把前 udim(=5) 维指令写入执行器的 ctrl。SOARM101Env 的注释写明 u 是"关节目标速度",
        # 并假设 XML 里配的是 velocity 执行器 (XML 本身我没读到, 无法确认)。
        # 量级对照: 训练数据里 u 是 [-0.5, 0.5] 上的均匀随机量; 而这里的限幅是 [-2, 2],
        # MPC 允许输出比训练时见过的更大的指令。
        self.env.data.ctrl[: self.env.udim] = target_velocity
        # --- 新增逻辑：施加负载力 ---
        # ---- 负载扰动实验: 在末端施加向下的力模拟夹持重物 (写字演示 use_payload=0, 走 else 分支把外力清零) ----
        if self.config.use_payload:
            # 假设负载 1kg，重力加速度 9.81
            payload_mass = self.config.use_payload
            g = 9.81
            force = payload_mass * g

            # 获取末端 ID
            ee_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "gripperframe")

            # xfrc_applied 格式: [fx, fy, fz, tx, ty, tz] (力 + 力矩)
            # 施加一个向下的力 (Z轴负方向)：模拟末端夹持了一个物体的重力，
            # 这是 Koopman 模型训练时完全没见过的"未建模扰动"，专门用来
            # 检验 KESO 的扰动补偿能力。
            self.env.data.xfrc_applied[ee_id] = [0, 0, -force, 0, 0, 0]
        else:
            # 如果没有负载，记得清零，因为 data 里的数据会残留
            ee_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "gripperframe")
            self.env.data.xfrc_applied[ee_id] = [0, 0, 0, 0, 0, 0]
        # 执行物理模拟
        # 一个控制步包含 frame_skip 个 MuJoCo 物理子步 (控制周期 0.02s, 物理步长更小), 这样控制频率 50Hz 而物理仿真更精细。
        for _ in range(self.env.frame_skip):
            mujoco.mj_step(self.env.model, self.env.data)
        state_full = None
        # ---- 噪声实验: 给关节位置加均匀噪声 (写字演示关闭) ----
        if self.config.use_nosie:
            # 定义扰动的最大强度（例如：±0.01 弧度）
            angle_disturbance_magnitude = 0.02
            # 选项 A: 扰动关节位置 (qpos)
            # qpos 尺寸为 (nq,)，即广义坐标位置维度
            # 生成一个随机扰动角度 (均匀分布在 [-magnitude, +magnitude])
            # 直接扰动 qpos 而不是扰动"观测值"，模拟的是"编码器/传感器读数
            # 本身带噪声"这种更贴近真实情况的噪声注入方式。
            angle_disturb = np.random.uniform(
                low=-angle_disturbance_magnitude,
                high=angle_disturbance_magnitude,
                size=self.env.udim,  # 扰动作用于所有关节位置
            )
            # 将扰动加到当前的关节角度上
            self.env.data.qpos[self.env.joint_ids] += angle_disturb
            # 关键步骤：在修改 qpos 后，必须调用 mj_forward 来更新所有内部变量
            # 否则 MuJoCo 的状态（如速度、雅可比矩阵等）将不一致。
            mujoco.mj_forward(self.env.model, self.env.data)
        # ---- UKF/卡尔曼滤波: 融合模型预测与带噪观测, 把修正后的关节角写回仿真 (写字演示关闭) ----
        if self.config.use_KEM:
            # 用卡尔曼滤波修正状态估计，融合"模型预测"和"(带噪声的)观测"，
            # 修正后的关节角直接写回仿真状态，相当于闭环里插入了一次"感知
            # 层面的降噪"。
            self.env.data.qpos[self.env.joint_ids], state_full = self.mpc_controller.get_updated_state_KF(
                self.env._get_state(), z_last, action
            )
            mujoco.mj_forward(self.env.model, self.env.data)
        # ---- KESO 扩张状态观测器: 估计并补偿未建模扰动 (写字演示关闭) ----
        if self.config.use_eso:
            if state_full is None:
                state_full = self.mpc_controller.Psi_o(torch.DoubleTensor(self.env._get_state()).unsqueeze(0))
            # 用扩张状态观测器估计未建模扰动(比如上面注入的负载力)，
            # 更新出的状态和随后累积的 u_eso 补偿量会在下一次 runMPC 里
            # 通过 get_control() 叠加进真正执行的控制量。
            self.env.data.qpos[self.env.joint_ids] = self.mpc_controller.get_updated_state_ESO(
                state_full, self.state_all_ref[self.traj_index + 1]
            )
            mujoco.mj_forward(self.env.model, self.env.data)

        # 4. 获取模拟后的新状态
        # 读取步进后的状态, 8 维 [x, y, z, 5 个关节角]。
        next_state = self.env._get_state()
        # 5. 渲染 (如果需要)
        # 刷新可视化窗口。
        if self.env.render_mode:
            self.env.render()
        return next_state

    # 窗口仍然打开时返回 True; 关闭 MuJoCo 窗口即结束 run_loop 的 while 循环。
    def is_running(self):
        return self.env.viewer.is_running()

    # 总入口: 先做一次开始前准备 (复位、画红点、等待 Enter), 再每帧调用 runFunc 直到窗口关闭。
    def run_loop(self):
        self.runBefore()
        while self.is_running():
            self.runFunc()


if __name__ == "__main__":
    # --- 0. 基本配置 ---
    # 示例参数，请替换为您自己的模型信息
    args = Args()
    # MPC 是逐帧在线求解的小规模 QP，用 CPU 跑 IPOPT 通常比 GPU 更合适
    # (数据搬运到 GPU 的开销反而更大)，和训练阶段默认用 cuda 是两回事。
    args.device = "cpu"
    MODEL_XML_PATH = args.xml_path
    EE_SITE_NAME = "gripperframe"  # 你的XML里定义的夹爪中心的 <site>
    NUM_JOINTS = args.u_dim  # 你的机器人关节数量
    use_pinocchio = False
    idx = 0  # 0 = X-Y 平面(贴桌面, wide_range 模型已覆盖此范围), 1 = Y-Z 平面(竖直)

    # 注意：这里的轨迹形状/噪声/负载/KF/ESO 开关是硬编码在 config_dict 里的，
    # 不读取 args 里的同名命令行参数(见 README 里的踩坑提示)，想跑不同实验
    # 配置需要直接改这个字典，或者用 control/config.py 的 ConfigGenerator_DBKN
    # 批量生成配置组合。
    config_dict = {
        "suffix": args.suffix,
        "env_name": args.env,
        "method": args.model,
        "use_KEM": False,
        "use_nosie": False,
        "use_payload": 0,  # 0,1,2
        "use_eso": False,
        "traj_name": "Heart",  # Rectangle, Fig8, FigStar, Heart, Helix, Lissajous
    }
    config = Config(**config_dict)

    if not use_pinocchio:
        # 创建生成器实例
        traj_generator = CartesianTrajectoryGenerator(
            model_path=MODEL_XML_PATH,
            ee_site_name=EE_SITE_NAME,
            num_joints=NUM_JOINTS,
            idx=idx,
            time_horizon=60,
            time_steps_per_sec=10,
        )
        # 定义末端执行器在整个轨迹中要保持的姿态 (例如，垂直向下)
        target_quat = None  # 绕X轴旋转90度
        # target_quat = np.array([0, 0, 1, 0])
        # 调用generate方法，反解出关节角度
        cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
            traj_name=config_dict["traj_name"],
            target_orientation=target_quat,
        )
    else:
        ARM_XML_PATH = args.arm_xml_path
        traj_generator = CartesianTrajectoryGenerator_pinocchio(
            arm_model_path=ARM_XML_PATH,
            ee_site_name=EE_SITE_NAME,
            num_joints=NUM_JOINTS,
            idx=idx,
            time_horizon=60,
            time_steps_per_sec=10,
        )
        # 角度（度）
        angle_degrees = 90 if idx == 0 else 0  # 0  90
        # 转换为弧度
        angle_radians = math.radians(angle_degrees)
        # 计算 cos 和 sin 值
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        # 构建绕 X 轴旋转的矩阵
        target_orientation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        # 调用generate方法，它会完成笛卡尔轨迹生成和IK求解两项工作
        cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
            traj_name=config_dict["traj_name"],
            target_orientation_matrix=target_orientation,
        )

    # ZMQ 发布者：不管真机是否在线都会尝试发送，真机端 so101_real.py 订阅
    # 这个地址；纯 sim2sim 验证时可以不启动真机进程，发送只是"石沉大海"。
    zmq_communicator = ZMQCommunicator("tcp://127.0.0.1:5555")
    model = init_model(args)
    # MPC/CasADi 走双精度计算，模型也要切到 double 保持精度一致，
    # 否则 float32 权重和 float64 的 CasADi 符号运算之间会有类型不匹配问题。
    model.double()
    load_model_path = args.output_dir + "/best_model.pt"
    model.load_state_dict(torch.load(load_model_path, map_location=torch.device("cpu")))
    MPC_Controller = MPCController_KESO(model, args)
    # MPC_Controller.verify_eso_stability()

    try:
        # 实例化播放器
        test = Test(
            path=MODEL_XML_PATH,
            Controller=MPC_Controller,
            communicator=zmq_communicator,  # 传入你的通信器
            cartesian_points=cartesian_points,
            joint_angle_traj=joint_angle_traj,
            num_joints=NUM_JOINTS,
            draw_num=300,
            config=config,
        )
        # 启动
        test.run_loop()

    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 清理通信资源
        zmq_communicator.cleanup()