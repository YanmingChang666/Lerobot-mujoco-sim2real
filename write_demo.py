"""
写汉字演示: 用 hanzi-writer-data 的笔画数据，抬笔 -> 平移 -> 落笔 交替书写。

复用 DBKN_KESO.py 里现成的 MuJoCo 仿真 + MPC 播放器 (Test 类)，
只是把参考轨迹换成 CartesianTrajectoryGenerator.generate_multi_stroke() 生成的
"多笔画"轨迹，而不是单笔连续画的 Fig8/Heart 等形状。

正式开始书写轨迹前，会先读取真机当前姿态 (通过 so101_real.py 在 :5556 广播的状态)，
平滑插值移动到轨迹起点，而不是让真机在指令流开始后自己"盲追" —— 方便肉眼确认真机和
仿真已经对齐了再继续。

改 CHAR 就能换字；想写一整串字用 hanzi_loader.text_to_strokes(text) 替换下面的
hanzi_to_strokes 调用即可（多字符需要覆盖的范围会成倍增大，建议先单字验证）。
"""

import json
import time

import numpy as np
import torch
import zmq

from args import Args
from control.config import Config
from control.hanzi_loader import hanzi_to_strokes
from control.MPC_Controler import MPCController_KESO
from control.TrajectoryGenerator import CartesianTrajectoryGenerator
from DBKN_KESO import Test, joint_offsets, sim_to_real  # 复用现成的 MuJoCo 仿真 + MPC 播放器 + 标定偏移
from models.init_model import init_model
from utility.ZMQ import ZMQCommunicator

CHAR = "二"  # 想写别的字改这里，比如 "书"/"人"/"永"

# wide_range 这轮训练时用的关节角覆盖范围 (SOARM101_Env.py 里的 perturbation_limits)，
# 用来在真正开仿真前粗查一下这个字需要的关节角度会不会超出模型学过的范围。
_TRAINED_JOINT_LOW = np.array([-0.45, -1.35, -1.25, -0.75, -0.3])
_TRAINED_JOINT_HIGH = np.array([0.45, 1.45, 0.85, 0.85, 0.3])


def check_reachability(joint_angle_traj: np.ndarray):
    """粗查这条轨迹用到的关节角度是否超出训练覆盖范围, 只是打印警告, 不会中止。"""
    jmin, jmax = joint_angle_traj.min(axis=0), joint_angle_traj.max(axis=0)
    over_low = jmin < _TRAINED_JOINT_LOW
    over_high = jmax > _TRAINED_JOINT_HIGH
    if over_low.any() or over_high.any():
        print("警告: 以下关节角度超出了 wide_range 训练覆盖的范围, 跟踪精度可能会变差:")
        for i in range(len(jmin)):
            if over_low[i] or over_high[i]:
                print(
                    f"  关节{i}: 本次需要 [{jmin[i]:.3f}, {jmax[i]:.3f}], "
                    f"训练覆盖 [{_TRAINED_JOINT_LOW[i]:.3f}, {_TRAINED_JOINT_HIGH[i]:.3f}]"
                )
    else:
        print("关节角度范围检查通过, 都在 wide_range 训练覆盖范围内。")


def sync_real_to_sim_start(
    target_sim_deg: np.ndarray,
    zmq_communicator: ZMQCommunicator,
    duration: float = 5.0,
    hz: float = 50.0,
    state_port: int = 5556,
    state_timeout_ms: int = 2000,
):
    """
    读取真机当前姿态 (订阅 so101_real.py 在 :state_port 广播的状态)，平滑插值移动到
    target_sim_deg (仿真角度, 度, 长度6) 对应的真机目标姿态，而不是让真机"盲追"。

    需要 so101_real.py 已经在运行 (否则读不到状态，会报错退出)。
    """
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{state_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, state_timeout_ms)

    print(f"正在读取真机当前姿态 (tcp://127.0.0.1:{state_port})...")
    try:
        msg = sub.recv_string()
    except zmq.Again as e:
        sub.close()
        ctx.term()
        raise RuntimeError(f"{state_timeout_ms / 1000:.0f}秒内没有收到真机状态, 请确认 so101_real.py 正在运行") from e
    finally:
        sub.close()
        ctx.term()

    current_real_deg = np.array(json.loads(msg))
    # sim_to_real: real = sim - offset  =>  反过来 sim = real + offset
    current_sim_deg = current_real_deg + np.array(joint_offsets)
    print("真机当前姿态 (换算成仿真角度):", current_sim_deg.round(2))
    print("轨迹起点目标 (仿真角度):        ", np.array(target_sim_deg).round(2))

    n_steps = max(1, int(duration * hz))
    for i in range(n_steps + 1):
        t = i / n_steps
        interp_sim_deg = current_sim_deg + t * (np.array(target_sim_deg) - current_sim_deg)
        interp_real_deg = sim_to_real(interp_sim_deg, joint_offsets)
        zmq_communicator.send_data(list(interp_real_deg))
        time.sleep(1.0 / hz)
    print("已平滑移动到轨迹起点，真机和仿真现在应该对齐了。")


if __name__ == "__main__":
    args = Args()
    args.device = "cpu"
    MODEL_XML_PATH = args.xml_path
    EE_SITE_NAME = "gripperframe"
    NUM_JOINTS = args.u_dim

    config_dict = {
        "suffix": args.suffix,
        "env_name": args.env,
        "method": args.model,
        "use_KEM": False,
        "use_nosie": False,
        "use_payload": 0,
        "use_eso": False,
        "traj_name": f"Write_{CHAR}",
    }
    config = Config(**config_dict)

    strokes = hanzi_to_strokes(CHAR, size_m=0.12)
    print(f"'{CHAR}' 共 {len(strokes)} 笔画")

    traj_generator = CartesianTrajectoryGenerator(
        model_path=MODEL_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS,
        idx=0,
        time_horizon=60,
        time_steps_per_sec=10,
    )
    # 第一遍: wrist_roll 自由求解，只是为了拿到轨迹起点"当前"的自然 wrist_roll 值
    _, preview_joint_traj, _ = traj_generator.generate_multi_stroke(
        strokes, center=(0.35, 0.0), write_height=0.025, lift_height=0.12
    )
    fixed_wrist_roll_rad = preview_joint_traj[0, 4] - np.radians(90)
    print(
        f"wrist_roll 起点自然值 {np.degrees(preview_joint_traj[0, 4]):.1f}°，"
        f"锁定在 -90° = {np.degrees(fixed_wrist_roll_rad):.1f}° 处，全程保持"
    )

    # 第二遍: wrist_roll 锁死在上面算出的值，只用另外4个关节去够位置
    cartesian_points, joint_angle_traj, pen_down_mask = traj_generator.generate_multi_stroke(
        strokes,
        center=(0.35, 0.0),
        write_height=0.025,  # 贴桌面书写高度
        lift_height=0.12,  # 抬笔转移高度
        fixed_wrist_roll_rad=fixed_wrist_roll_rad,
    )
    print(f"落笔书写点数: {pen_down_mask.sum()}, 抬笔/平移点数: {(~pen_down_mask).sum()}")
    check_reachability(joint_angle_traj)

    zmq_communicator = ZMQCommunicator("tcp://127.0.0.1:5555")

    try:
        # 正式开始书写前, 先平滑同步真机姿态到轨迹起点 (需要 so101_real.py 已在运行)
        target_sim_deg = np.degrees(np.append(joint_angle_traj[0], 0.0))  # 补一个 gripper=0
        sync_real_to_sim_start(target_sim_deg, zmq_communicator, duration=5.0)

        model = init_model(args)
        model.double()
        load_model_path = args.output_dir + "/best_model.pt"
        model.load_state_dict(torch.load(load_model_path, map_location=torch.device("cpu")))
        MPC_Controller = MPCController_KESO(model, args)

        test = Test(
            path=MODEL_XML_PATH,
            Controller=MPC_Controller,
            communicator=zmq_communicator,
            cartesian_points=cartesian_points,
            joint_angle_traj=joint_angle_traj,
            num_joints=NUM_JOINTS,
            draw_num=300,
            config=config,
        )
        test.run_loop()
    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        zmq_communicator.cleanup()
