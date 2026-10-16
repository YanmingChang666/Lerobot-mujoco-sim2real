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

============================================================
【详细流程】(与 __main__ 里的步骤一一对应)
============================================================
  步骤1 取字      hanzi_to_strokes(CHAR, size_m=0.12)
                  汉字 -> 笔画中线折线, 单位米, 以字中心为原点 (见 control/hanzi_loader.py)
  步骤2 第1遍轨迹 generate_multi_stroke(..., wrist_roll 自由)
                  笔画 -> 笛卡尔点序列 (落笔 / 抬笔 / 平移 / 落笔) -> 逐点 IK -> 关节角
                  目的只有一个: 拿到轨迹起点处"自然"的 wrist_roll 值
  步骤3 锁腕      fixed_wrist_roll_rad = 起点自然 wrist_roll - 90°
  步骤4 第2遍轨迹 generate_multi_stroke(..., fixed_wrist_roll_rad=...)
                  wrist_roll 不再参与 IK, 只用另外 4 个关节够 XYZ -> 最终 (xyz, 关节角, 落笔掩码)
  步骤5 查范围    check_reachability(): 关节角是否超出 wide_range 模型的训练覆盖范围 (只警告)
  步骤6 对齐真机  sync_real_to_sim_start(): 读真机当前角度, 5 秒平滑插值到轨迹起点
  步骤7 闭环跟踪  Test.run_loop() (来自 DBKN_KESO.py):
                  每帧: 取未来 H 步参考 -> Koopman 提升 -> MPC 求解 -> MuJoCo 步进
                        -> 仿真关节角(度) 减去 joint_offsets -> ZMQ :5555 发给真机
  步骤8 收尾      主轨迹播完 -> 保存结果 -> 关节空间线性插值回 Home 姿态

【几个容易误解的点】
  1. 仿真里没有墨迹/笔: "写字"就是让末端在 write_height 高度沿笔画走, 红点只是参考轨迹的可视化。
  2. pen_down_mask 只在本脚本里用来统计点数, 没有传给 Test:
     对控制器来说, 抬笔/落笔只是 Z 坐标不同的普通参考点。
  3. 这里 use_KEM / use_eso / use_nosie / use_payload 全部关闭:
     实际跑的是"DBKN Koopman 模型 + 纯 MPC"跟踪, 没有 UKF/KESO 补偿, 也没有噪声/负载扰动。
  4. 控制是开环单向的: MPC 只用仿真自己的状态, 真机只是被动追目标角度, 不回读真机传感器
     (步骤6 读真机状态只发生在开始前的一次, 用来对齐)。
  5. 抬笔高度 0.12m、落笔高度 0.025m 是这个脚本里调用时传的参数,
     与 generate_multi_stroke 函数自身的默认值 (0.055 / 0.12) 不同, 以这里为准。
"""

# ---- 标准库 / 第三方库 ----
# json  : 解析 so101_real.py 通过 ZMQ 广播的 JSON 字符串 (真机当前关节角)
# time  : 控制插值发送的节奏 (sleep)
# numpy : 数组运算 (轨迹、关节角范围比较)
# torch : 加载训练好的 DBKN 模型权重
# zmq   : 订阅真机状态 (本文件里只用到 SUB 端; 发布端封装在 utility/ZMQ.py)
import json
import time

import numpy as np
import torch
import zmq

# ---- 项目内部模块 ----
# Args                         : 命令行/默认参数 (模型类型、suffix、XML 路径、关节数 u_dim、输出目录 ...)
# Config                       : 一次实验的配置, 负责结果保存路径 (traj_name 会出现在保存的文件名里)
# hanzi_to_strokes             : 汉字 -> 笔画折线
# MPCController_KESO           : Koopman 模型 + MPC (+ 可选 UKF/KESO) 的控制器
# CartesianTrajectoryGenerator : 笛卡尔轨迹生成 + dm_control 逆运动学
# Test / joint_offsets / sim_to_real : 复用 DBKN_KESO.py 里的闭环仿真播放器、标定偏移和角度换算
# init_model                   : 按 args 构建网络结构 (权重稍后再加载)
# ZMQCommunicator              : ZMQ 发布端封装, send_data() 把目标关节角发给真机
from args import Args
from control.config import Config
from control.hanzi_loader import hanzi_to_strokes
from control.MPC_Controler import MPCController_KESO
from control.TrajectoryGenerator import CartesianTrajectoryGenerator
from DBKN_KESO import Test, joint_offsets, sim_to_real  # 复用现成的 MuJoCo 仿真 + MPC 播放器 + 标定偏移
from models.init_model import init_model
from utility.ZMQ import ZMQCommunicator

# 要写的字 (单个汉字)。想写别的字改这里。
# 同时它也会进入 traj_name ("Write_二"), 从而影响实验结果的保存目录名。
CHAR = "二"  # 想写别的字改这里，比如 "书"/"人"/"永"

# wide_range 这轮训练时用的关节角覆盖范围 (SOARM101_Env.py 里的 perturbation_limits)，
# 用来在真正开仿真前粗查一下这个字需要的关节角度会不会超出模型学过的范围。
# 5 个值的顺序 = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll], 单位弧度。
# 这是训练数据采集时 SOARM101_Env.perturbation_limits 的上下界:
# Koopman 模型只在这个范围内"见过"数据, 超出范围预测/控制会变差。
# (gripper 不由 MPC 控制, 所以没有第 6 项。)
_TRAINED_JOINT_LOW = np.array([-0.45, -1.35, -1.25, -0.75, -0.3])
_TRAINED_JOINT_HIGH = np.array([0.45, 1.45, 0.85, 0.85, 0.3])


def check_reachability(joint_angle_traj: np.ndarray):
    """粗查这条轨迹用到的关节角度是否超出训练覆盖范围, 只是打印警告, 不会中止。"""
    # joint_angle_traj 形状 (N, 5)。按列取最小/最大值 = 整条轨迹每个关节用到的角度范围。
    jmin, jmax = joint_angle_traj.min(axis=0), joint_angle_traj.max(axis=0)
    # 逐关节判断: 是否有时刻低于训练下界 / 高于训练上界 (布尔数组, 长度 5)。
    over_low = jmin < _TRAINED_JOINT_LOW
    over_high = jmax > _TRAINED_JOINT_HIGH
    # 只要有任何关节越界就逐个打印; 只是提示, 不会中止程序 —— 是否继续由使用者判断。
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
    # 为什么每次调用都新建一个 SUB socket、读完就关:
    #     这个函数只需要"开始前读一次真机当前姿态", 不需要常驻订阅, 所以用完立即释放, 避免占用 :5556。
    ctx = zmq.Context()
    # SUB = 订阅端: 接收 so101_real.py 用 PUB 广播的真机当前关节角。
    sub = ctx.socket(zmq.SUB)
    # 连接到本机 so101_real.py 的状态广播端口 (默认 5556)。
    sub.connect(f"tcp://127.0.0.1:{state_port}")
    # 订阅前缀为空字符串 = 订阅所有消息。
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    # 接收超时 (毫秒)。超时后 recv_string() 抛 zmq.Again, 而不是无限阻塞 —— 真机进程没开时能及时报错。
    sub.setsockopt(zmq.RCVTIMEO, state_timeout_ms)

    print(f"正在读取真机当前姿态 (tcp://127.0.0.1:{state_port})...")
    try:
        # 阻塞等待下一条状态消息 (so101_real.py 持续广播, 一般很快收到)。
        msg = sub.recv_string()
    # 超时没收到消息: 先释放资源再抛出带提示的 RuntimeError (from e 保留原始异常链)。外层 finally 也会关闭, 重复 close 是无害的。
    except zmq.Again as e:
        sub.close()
        ctx.term()
        raise RuntimeError(f"{state_timeout_ms / 1000:.0f}秒内没有收到真机状态, 请确认 so101_real.py 正在运行") from e
    finally:
        sub.close()
        ctx.term()

    # 消息内容是 JSON 数组 = 真机 6 个关节当前角度 (度, 真机标定坐标系): [pan, lift, elbow, wrist_flex, wrist_roll, gripper]。
    current_real_deg = np.array(json.loads(msg))
    # 真机角度和仿真角度差一个标定偏移, 所以要换算到仿真坐标系再与轨迹起点比较/插值。
    # sim_to_real: real = sim - offset  =>  反过来 sim = real + offset
    current_sim_deg = current_real_deg + np.array(joint_offsets)
    print("真机当前姿态 (换算成仿真角度):", current_sim_deg.round(2))
    print("轨迹起点目标 (仿真角度):        ", np.array(target_sim_deg).round(2))

    # 插值步数 = 总时长 * 发送频率 (默认 5s * 50Hz = 250 步); max(1, ...) 防止 duration 过小时除零。
    n_steps = max(1, int(duration * hz))
    # 多循环 1 次 (+1) 是为了让最后一步 t=1 刚好落在目标姿态上。
    for i in range(n_steps + 1):
        # 归一化进度 t: 0 -> 1。
        t = i / n_steps
        # 在仿真角度空间做线性插值 (每个关节同步匀速运动, 同时到达目标)。
        interp_sim_deg = current_sim_deg + t * (np.array(target_sim_deg) - current_sim_deg)
        # 换算回真机坐标系: real = sim - offset。
        interp_real_deg = sim_to_real(interp_sim_deg, joint_offsets)
        # 发给 so101_real.py (ZMQ :5555)。真机端还有 max_relative_target=10° 的单次位移限幅作为最后一道保护。
        zmq_communicator.send_data(list(interp_real_deg))
        # 按 hz 控制发送节奏, 使 5 秒的插值真的耗时约 5 秒。
        time.sleep(1.0 / hz)
    print("已平滑移动到轨迹起点，真机和仿真现在应该对齐了。")


if __name__ == "__main__":
    # 步骤0: 读取参数。命令行用法: python write_demo.py --model DBKN --suffix wide_range
    args = Args()
    # MPC 每帧在线求解小规模问题, 用 CPU 反而比 GPU 快 (省去数据搬运)。
    args.device = "cpu"
    # MuJoCo 机械臂模型 XML: 既用于 IK 的离线求解, 也用于闭环仿真环境。
    MODEL_XML_PATH = args.xml_path
    # 末端执行器参考点 (XML 里的 <site>)。"笔尖"位置就是这个点; 若真实装了笔, 要把笔尖伸出量折算进 write_height。
    EE_SITE_NAME = "gripperframe"
    # 受控关节数 = 5 (不含 gripper)。
    NUM_JOINTS = args.u_dim

    # 实验配置。本演示只做最基础的"Koopman + MPC"跟踪, 所以所有增强/扰动开关都关闭。
    # (与 DBKN_KESO.py 的 __main__ 一样, 这些值是写死在字典里的, 不读命令行里的同名参数。)
    config_dict = {
        "suffix": args.suffix,  # 训练批次标签, 决定加载 results/SOARM101/<suffix>/<model>/best_model.pt
        "env_name": args.env,  # 环境名, 参与结果路径
        "method": args.model,  # 模型类型, 如 DBKN
        "use_KEM": False,  # 是否用 UKF/卡尔曼滤波做状态估计 (写字演示关闭)
        "use_nosie": False,  # 是否向关节位置注入噪声 (变量名 nosie 为源码原拼写, 勿改; 写字演示关闭)
        "use_payload": 0,  # 末端负载 (kg 数值, 0 = 无; 写字演示关闭)
        "use_eso": False,  # 是否启用 KESO 扰动补偿 (写字演示关闭)
        "traj_name": f"Write_{CHAR}",  # 仅用于命名保存结果的文件夹; 实际轨迹由下面的 generate_multi_stroke 生成
    }
    # Config 对象会在轨迹播完后负责 save_result() (保存控制量、实际轨迹、参考轨迹)。
    config = Config(**config_dict)

    # 步骤1: 汉字 -> 笔画。
    #     返回 strokes = [[(x, y), ...], ...], 单位米, 原点在字中心; size_m=0.12 即字大约 12cm x 12cm。
    #     此时还只是 2D 折线, 没有 z, 没有抬笔/落笔动作, 也不知道要写在桌面哪里。
    strokes = hanzi_to_strokes(CHAR, size_m=0.12)
    # 笔画数 = 外层列表长度。笔画越多, 抬笔/平移段越多, 轨迹越长。
    print(f"'{CHAR}' 共 {len(strokes)} 笔画")

    # 步骤2 准备: 创建轨迹生成器。内部会加载一份独立的 MuJoCo 模型专门做逆运动学 (与闭环仿真里的环境是两个实例)。
    traj_generator = CartesianTrajectoryGenerator(
        model_path=MODEL_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS,
        idx=0,  # 0 = X-Y 水平面(贴桌面); 1 = Y-Z 竖直面。写字用桌面平面。
        time_horizon=60,  # 对 generate_multi_stroke 实际无效: 点数由笔画决定, 时间向量会被重新生成
        time_steps_per_sec=10,  # 同上, 这两个参数只对 generate() 画单笔图形时决定点数
    )
    # 第一遍: wrist_roll 自由求解，只是为了拿到轨迹起点"当前"的自然 wrist_roll 值
    # 第1遍的返回值只取中间的关节角轨迹:
    #     xyz 点云和 pen_down_mask 这一遍不用, 用 _ 丢弃。
    # 参数含义:
    #     center=(0.35, 0.0)  字中心放在世界系 X=0.35m、Y=0m 处 (机械臂正前方桌面上)
    #     write_height=0.025  落笔时末端 Z=2.5cm (贴近桌面, 假设桌面在 z≈0)
    #     lift_height=0.12    抬笔转移时末端 Z=12cm
    # 此时没有传 fixed_wrist_roll_rad, wrist_roll 由 IK 自由决定。
    _, preview_joint_traj, _ = traj_generator.generate_multi_stroke(
        strokes, center=(0.35, 0.0), write_height=0.025, lift_height=0.12
    )
    # 步骤3: preview_joint_traj[0, 4] = 起点处 wrist_roll (弧度), 下标 4 即第 5 个关节。
    #     在它的基础上再转 -90°, 作为全程固定值 —— 让夹爪(笔)的朝向相对"自然姿态"转 90°。
    #     (-90° 是作者按手感选的; 想改书写时的握笔方向, 改这里的 90 即可)
    fixed_wrist_roll_rad = preview_joint_traj[0, 4] - np.radians(90)
    print(
        f"wrist_roll 起点自然值 {np.degrees(preview_joint_traj[0, 4]):.1f}°，"
        f"锁定在 -90° = {np.degrees(fixed_wrist_roll_rad):.1f}° 处，全程保持"
    )

    # 步骤4: 第2遍才是真正使用的轨迹。两遍的 strokes/center/write_height/lift_height 必须完全一致,
    # 否则"起点自然 wrist_roll"就和最终轨迹不匹配了。
    # 第二遍: wrist_roll 锁死在上面算出的值，只用另外4个关节去够位置
    # 三个返回值:
    #     cartesian_points : (N, 3) 笛卡尔参考点 (世界系, 米) —— 红点可视化、MPC 的笛卡尔参考
    #     joint_angle_traj : (N, 5) 对应的关节角 (弧度) —— 参考关节角, 也用来复位仿真起点
    #     pen_down_mask    : (N,) 布尔数组, True = 落笔书写点, False = 抬笔/平移/落笔过渡点
    cartesian_points, joint_angle_traj, pen_down_mask = traj_generator.generate_multi_stroke(
        strokes,
        center=(0.35, 0.0),
        write_height=0.025,  # 贴桌面书写高度
        lift_height=0.12,  # 抬笔转移高度
        fixed_wrist_roll_rad=fixed_wrist_roll_rad,
    )
    # 统计信息: 书写点越多越慢; 抬笔/平移点占比高 = 笔画多或笔画间距离远。注意 pen_down_mask 之后并没有传给 Test。
    print(f"落笔书写点数: {pen_down_mask.sum()}, 抬笔/平移点数: {(~pen_down_mask).sum()}")
    # 步骤5: 提前提示可达性风险。越界时精度可能变差, 但不影响轨迹生成。
    check_reachability(joint_angle_traj)

    # 创建 ZMQ 发布端 (PUB, 绑定 :5555)。后面所有"发给真机的目标角度"都走它:
    #     对齐阶段 (sync_real_to_sim_start) 用一次, 闭环仿真阶段 (Test) 每帧用一次。
    zmq_communicator = ZMQCommunicator("tcp://127.0.0.1:5555")

    try:
        # 步骤6 前置: 此时 so101_real.py 必须已经在另一个终端运行, 否则下面的对齐会在 2 秒后报错退出。
        # 正式开始书写前, 先平滑同步真机姿态到轨迹起点 (需要 so101_real.py 已在运行)
        # 轨迹第 1 个关节角 (5 维, 弧度) -> 追加 gripper=0 凑成 6 维 -> 转成度。
        # 真机共 6 个关节, MPC 只控 5 个, 夹爪固定给 0。
        target_sim_deg = np.degrees(np.append(joint_angle_traj[0], 0.0))  # 补一个 gripper=0
        # 让真机从当前姿态平滑走到轨迹起点, 避免指令流一开始真机被迫大幅跳变。
        sync_real_to_sim_start(target_sim_deg, zmq_communicator, duration=5.0)

        # 步骤7 准备: 加载训练好的 Koopman 网络 (DBKN) 和 MPC 控制器。
        model = init_model(args)
        # MPC 求解走双精度 (CasADi), 模型也转 float64 以保持一致。
        model.double()
        # 权重路径: results/SOARM101/<suffix>/<model>/best_model.pt (wide_range 训练得到, 覆盖桌面平面轨迹所需的关节范围)。
        load_model_path = args.output_dir + "/best_model.pt"
        # map_location=cpu: 即使训练时用 GPU 保存, 也能在 CPU 上加载。
        model.load_state_dict(torch.load(load_model_path, map_location=torch.device("cpu")))
        # 构建 MPC 控制器 (内部是 CasADi 符号优化问题 + Koopman 提升函数)。
        # 具体数学在 control/MPC_Controler.py 里。类名带 KESO, 但本演示 config 里 use_eso=False, 扰动补偿没有启用。
        MPC_Controller = MPCController_KESO(model, args)

        # Test 是 DBKN_KESO.py 里的闭环仿真播放器, 它做的事 (详见该文件的注释版):
        #     - 创建带可视化窗口的 MuJoCo 环境, 复位到轨迹起点, 画出红色参考点
        #     - 等待用户按 Enter 后开始: 每帧 runMPC -> 步进仿真 -> 通过 ZMQ 发给真机, 约 50Hz
        #     - 轨迹播完后保存结果, 并线性插值回 Home
        test = Test(
            path=MODEL_XML_PATH,
            Controller=MPC_Controller,
            communicator=zmq_communicator,
            cartesian_points=cartesian_points,
            joint_angle_traj=joint_angle_traj,
            num_joints=NUM_JOINTS,
            draw_num=300,  # 窗口里最多画约 300 个红点; 点太多会拖慢渲染, 内部按步长抽稀
            config=config,
        )
        # 阻塞运行, 直到关闭 MuJoCo 窗口。
        test.run_loop()
    # Ctrl+C 中断: 真机动作异常时用它立刻停止指令流。
    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 无论正常结束还是被中断, 都释放 ZMQ 资源。
        zmq_communicator.cleanup()