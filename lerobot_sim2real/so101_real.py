"""
Sim2Real 通信桥的"真机端"：订阅仿真端(so101_mujoco.py / DBKN_KESO.py /
IBKN_UKF.py / write_demo.py 等)通过 ZMQ PUB :5555 广播的目标关节角度，
转发给真实的 SO-101 机械臂执行；同时把真机当前关节角度通过 :5556 广播
出去，供仿真端 sync_real_to_sim_start()（把仿真姿态平滑对齐到真机当前
姿态）以及 ROS2 /joint_states 等下游消费者使用。

整条链路是单向开环的：本文件只管"收到目标角度就转发给硬件"，不做任何
反馈控制或安全轨迹规划，真正的安全兜底在 real_robot.create_real_robot()
返回的 robot 对象内部（README 提到的 max_relative_target=10° 限幅）。

必须在单独的 conda 环境 (lerobot-env, Python 3.12) 里运行，因为
so101_follower 依赖的 LeRobot 版本和仿真训练用的 koopman-sim2real 环境
(Python 3.10, 装了 ROS2 rclpy) 依赖冲突，两者无法共存于同一进程，
详见 README 的 Sim2Real 部署一节。
"""

import json

import zmq
from real_robot import create_real_robot

# ==================== 配置常量 ====================
# ... (这部分保持不变)
# 顺序必须和仿真端(SOARM101Env.joint_names 等)一致，ZMQ 收到的角度列表
# 按下标一一对应到这些关节名，顺序错了会导致"角度发对了但装错关节"。
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ZMQ_IP = "127.0.0.1"
ZMQ_PORT = "5555"  # commands: subscribed here (target joint degrees)
STATE_ZMQ_PORT = "5556"  # state: published here (current joint degrees), for ROS2 /joint_states feedback
# Unused by this script; the port actually opened is set in real_robot.py.
REAL_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A054769-if00"


def main():
    """
    连接真机硬件 -> 订阅仿真端目标角度并转发执行 -> 广播真机当前姿态，
    循环直到 Ctrl+C 中断，最后统一清理 ZMQ 和硬件连接。
    """
    # ==================== 阶段 1 & 2 (几乎不变) ====================
    print("1. 配置机器人...")
    robot = create_real_robot("so101")

    print(f"2. 初始化ZMQ订阅者, 连接到 tcp://{ZMQ_IP}:{ZMQ_PORT}")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_IP}:{ZMQ_PORT}")
    # 订阅空字符串前缀 = 不做任何 topic 过滤，接收这个 PUB 端发布的所有消息。
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"   初始化ZMQ状态发布者, 绑定到 tcp://{ZMQ_IP}:{STATE_ZMQ_PORT}")
    state_socket = context.socket(zmq.PUB)
    state_socket.bind(f"tcp://{ZMQ_IP}:{STATE_ZMQ_PORT}")

    # ==================== 新增: 初始化 Poller ====================
    # Poller 是ZMQ中用于处理非阻塞IO的核心工具
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)  # 告诉poller监控socket上的“可读”事件

    try:
        # ==================== 阶段 3: 连接硬件 (不变) ====================
        print("3. 连接机器人硬件...")
        robot.connect()
        print("   机器人连接成功！")

        # ==================== 阶段 4: 实时控制主循环 (核心修改) ====================
        print("4. 进入实时控制循环。现在可以按 Ctrl+C 随时中断。")
        while True:
            # a. 使用 poller 带超时地等待消息
            #    等待最多100毫秒。如果在100毫秒内有消息，它会立即返回。
            #    如果没有消息，它会等待100毫秒然后返回一个空字典。
            #    这个超时机制是让 Ctrl+C 生效的关键！
            # 用带超时的 poll 而不是阻塞式 recv，是因为纯阻塞调用会让 Python
            # 主线程一直卡在 C 扩展里，收不到 SIGINT 信号；100ms 超时让循环
            # 定期返回 Python 层，Ctrl+C 才能被及时响应。
            socks = dict(poller.poll(timeout=100))

            # b. 检查我们的socket是否在返回的字典中，且事件是“可读”
            if socket in socks and socks[socket] == zmq.POLLIN:
                # c. 只有在确定有消息时才接收，这个调用不会阻塞
                data_string = socket.recv_string().strip()

                # --- 后续逻辑与原来完全相同，但现在被包裹在if块内 ---
                if not data_string:
                    continue

                joint_pos_deg_list = json.loads(data_string)

                if len(joint_pos_deg_list) != len(JOINT_NAMES):
                    print(
                        f"警告：收到的数据长度 ({len(joint_pos_deg_list)}) 与期望的关节数量 ({len(JOINT_NAMES)}) 不匹配。"
                    )
                    continue

                # LeRobot 的 robot.send_action 接口要求 {"<关节名>.pos": 角度}
                # 这种带后缀的字典格式，这里做一次从"纯列表"到该格式的转换。
                action_to_send = {f"{name}.pos": angle for name, angle in zip(JOINT_NAMES, joint_pos_deg_list)}

                # 真正下发到硬件：这一步内部会经过 real_robot.py 里
                # max_relative_target=10° 的限幅安全网，即使仿真端瞬间发来
                # 一个跳变很大的目标角度，单步实际执行位移也会被夹住。
                robot.send_action(action_to_send)
                # print(f"已发送动作: {action_to_send}")

            # 如果没有消息，循环会自然地结束本次迭代，然后开始下一次
            # 这就给了Python解释器处理 KeyboardInterrupt 的机会

            # ==================== 新增: 读取并发布当前关节状态 (供 ROS2 /joint_states 使用) ====================
            # 不管本次循环有没有收到新的目标指令，每轮都读一次真机当前姿态
            # 并广播出去——这是仿真端 sync_real_to_sim_start() 能对齐真机
            # 起始姿态、以及 ROS2 侧能拿到 /joint_states 反馈的数据来源。
            obs = robot.get_observation()
            state_deg_list = [obs[f"{name}.pos"] for name in JOINT_NAMES]
            state_socket.send_string(json.dumps(state_deg_list))

    except KeyboardInterrupt:
        print("程序被用户中断。")
    except Exception as e:
        print(f"在主循环中发生错误: {e}")
    finally:
        # ==================== 阶段 5: 安全清理资源 (不变) ====================
        print("正在安全关闭...")

        # a. 关闭 ZMQ
        if not socket.closed:
            socket.close()
        if not state_socket.closed:
            state_socket.close()
        if not context.closed:
            context.term()
        print("   ZMQ 已关闭。")

        # b. 断开机器人连接
        if robot and robot.is_connected:
            print("   断开机器人连接...")
            robot.disconnect()
            print("   机器人已安全断开。")


if __name__ == "__main__":
    main()
