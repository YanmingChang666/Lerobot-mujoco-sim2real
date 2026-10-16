"""
Koopman 训练数据的采集与加载模块。

整体流程：
1. 用 SOARM101Env 反复 reset + step，喂不同类型的控制信号（random/sin/chirp），
   把每一步的 (u_t, x_t) 记录成一条条轨迹，攒成 numpy 数组并缓存到磁盘；
2. 把这些轨迹数组包装成 PyTorch DataLoader，供 train.py 里的训练循环按批次
   读取序列窗口（Collater 会在取批次时把 [u, x] 拆开）；
3. 独立运行本文件（__main__ 部分）时，还会额外生成一条笛卡尔轨迹用于快速
   目视检查数据分布和关节轨迹形状，属于调试用途，不是数据采集主流程的一部分。

三种控制信号类型的用途区分：
- 'random'：训练/验证/测试都会用，模拟"随机激励"，让 Koopman 模型学到尽量
  通用的动力学，而不是只适应某一种输入模式。
- 'sin'/'chirp'：只在生成测试集时使用，用规律的信号（固定频率正弦 / 线性扫频）
  检验模型对训练时没见过的、结构化输入的泛化能力。
"""

import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm

# 因为本文件既会被 train.py import，也会被直接 `python SOARM101_DataCollection.py`
# 运行，这里手动把项目根目录塞进 sys.path，保证两种运行方式下 `from args import Args`
# 之类的绝对导入都能找到模块。
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import matplotlib.pyplot as plt

from args import Args
from SOARM101.SOARM101_Env import SOARM101Env


class Collater:
    """
    DataLoader 的 collate_fn：把一批形状为 [steps+1, u_dim+x_dim] 的轨迹张量
    堆叠成 [batch, steps+1, u_dim+x_dim]，再按最后一维切开成控制量 u 和状态 x。

    之所以要自定义 collate_fn 而不是用 DataLoader 默认拼接，是因为原始数据里
    u 和 x 是拼在同一个特征维里存储的（见 generate_physics_based_data 的
    np.concatenate([u, s])），拆分逻辑需要知道 u_dim/x_dim 才能正确切片，
    并且顺手把结果搬到训练用的 device 上，省得训练循环里每步都做 .to(device)。
    """

    def __init__(self, x_dim: int, u_dim: int, device: str = "cuda"):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.device = device

    def __call__(self, batch_list: list):
        # 解压批次列表：batch_list是包含多个样本的列表
        # 每个样本是一个元组，我们只需要第一个元素（数据张量）
        batch_data = list(zip(*batch_list))[0]
        # 将列表中的张量堆叠成一个批次张量
        # 维度：[batch_size, seq_len, features]
        batch_data = torch.stack(batch_data, dim=0)
        # 特征维的存储顺序固定为 [u(前 u_dim 列), x(后 x_dim 列)]，
        # 和 generate_physics_based_data 里 np.concatenate([u, s]) 的顺序对应。
        return dict(
            x=batch_data[:, :, self.u_dim : self.u_dim + self.x_dim].to(self.device),
            u=batch_data[:, :, : self.u_dim].to(self.device),
        )


class SineInputGenerator:
    """
    生成结构化（非随机）的控制输入序列，专门用来产出 'sin'/'chirp' 类型的测试数据，
    检验训练好的 Koopman 模型在"规律信号"下的多步预测/跟踪表现。

    每条轨迹(traj_i)在构造时就随机采样好了自己独有的一组频率/幅值/相位，
    保证同一批测试轨迹之间彼此不同，但同一条轨迹在整个时间序列里参数是固定的
    （chirp 模式下频率会随时间线性变化，见 __call__）。
    """

    def __init__(self, traj_num, udim=5, freq_range=(0.01, 0.05), amp_range=(0.05, 0.15), sine_mask=None, mode="sin"):
        """
        初始化每条轨迹的正弦参数：
        - freq_range: 每个维度频率范围
        - amp_range: 每个维度幅值范围
        - sine_mask: (traj_num, udim) 的 bool 矩阵，指示哪些维度使用正弦，其余为 0
        'sin' 固定频率，'chirp' 扫频
        """
        self.traj_num = traj_num
        self.udim = udim
        self.mode = mode
        self.freq_start = freq_range[0]
        self.freq_end = freq_range[1]

        # 一次性为所有轨迹、所有控制维度采样好频率/幅值/初相，
        # 后面 __call__ 只是按 (traj_i, t) 查表 + 算 sin，不再引入新的随机性。
        self.freq_table = np.random.uniform(freq_range[0], freq_range[1], size=(traj_num, udim))
        self.amp_table = np.random.uniform(amp_range[0], amp_range[1], size=(traj_num, udim))
        self.phase_table = np.random.uniform(0, 2 * np.pi, size=(traj_num, udim))

        if sine_mask is None:
            self.sine_mask = np.ones((traj_num, udim), dtype=bool)  # 默认所有维度都用正弦
        else:
            self.sine_mask = sine_mask.astype(bool)

    def __call__(self, t, traj_i, T_total=200):
        """按时间步 t 和轨迹编号 traj_i 计算当前控制输入 u(t)，shape=(udim,)。"""
        if self.mode == "sin":
            # 固定频率：整条轨迹里 freq 不随 t 变化
            freq = self.freq_table[traj_i]
        elif self.mode == "chirp":
            # 时间归一化因子
            ratio = t / T_total
            # 线性扫频：从 freq_start 到 freq_end，让测试信号覆盖更宽的频段，
            # 比单一固定频率的 'sin' 更能暴露模型在不同响应速度下的预测误差。
            freq = self.freq_table[traj_i] + (self.freq_end - self.freq_start) * ratio
            # freq = np.full(self.udim, freq)

        amp = self.amp_table[traj_i]
        phase = self.phase_table[traj_i]
        mask = self.sine_mask[traj_i]

        u = np.zeros(self.udim)
        sine_values = amp * np.sin(2 * np.pi * freq * t + phase)
        # 只有 mask=True 的控制维度真正被赋值为正弦信号，其余维度保持 0，
        # 用来支持"只激励部分关节"的测试场景。
        u[mask] = sine_values[mask]
        return u


# --- 重构后的数据生成器 ---


class SOARM101DataGenerator:
    """
    数据采集与数据加载的总入口：负责驱动 MuJoCo 环境滚动仿真、把结果落盘/读缓存，
    并封装成 train.py 直接可用的 DataLoader。
    """

    def __init__(self, args: Args) -> None:
        self.args = args
        self.udim = self.args.u_dim  # 控制维度
        self.xdim = self.args.x_dim  # Koopman模型所需的状态维度

        # 核心改动：实例化基于物理的MuJoCo环境
        # 注意：这里我们不需要可视化，所以 render_mode=None
        print("正在初始化物理仿真环境...")
        self.env = SOARM101Env(xml_path=self.args.xml_path)
        print("物理环境初始化完成。")
        self.collate_fn = Collater(self.args.x_dim, self.args.u_dim, self.args.device)

    def generate_physics_based_data(self, traj_num, steps, input_type):
        """
        使用MuJoCo物理引擎生成数据。
        数据格式: [traj_num, steps+1, u_dim + x_dim] = [traj_num, steps+1, 5+8=13]
        """
        if input_type in {"sin", "chirp"}:
            # 使用您已有的输入信号生成器
            generator = SineInputGenerator(
                traj_num=traj_num, udim=self.udim, freq_range=(0.0025, 0.05), amp_range=(-0.5, 0.5), mode=input_type
            )

        # 创建数据存储数组
        data_array = np.empty((traj_num, steps + 1, self.udim + self.xdim))

        for traj_i in tqdm(range(traj_num), desc=f"Generating '{input_type}' data"):
            # 1. 重置环境以获得初始状态
            # 每条轨迹都独立 reset：初始关节角在 SOARM101Env.perturbation_limits
            # 范围内随机采样，保证不同轨迹覆盖不同的起始区域。
            s0, _ = self.env.reset()
            # 2. 生成初始控制输入 u_0
            if input_type in {"sin", "chirp"}:
                u0 = generator(0, traj_i)
            else:  # 'random'
                # 均匀分布采样在 [-0.5, 0.5] 范围内的随机控制量，作为"随机激励"信号
                u0 = (np.random.rand(self.udim) - 0.5) * 2 * 0.5

            # 3. 存储初始数据点 (t=0)
            data_array[traj_i, 0, :] = np.concatenate([u0, s0])

            # 4. 在循环中生成轨迹
            for i in range(1, steps + 1):
                # a. 获取上一步的控制输入 u_{i-1} (即 u0)
                u_prev = data_array[traj_i, i - 1, : self.udim]

                # b. 将 u_{i-1} 应用到环境中，得到真实的下一状态 s_i
                # 这一步是 Koopman 训练样本 (x_i-1, u_i-1) -> x_i 里 "x_i" 的来源：
                # 用真实物理仿真而不是模型预测，保证标签是准确的。
                s_next, _, _, _, _ = self.env.step(u_prev)

                # c. 生成当前的控制输入 u_i
                if input_type in {"sin", "chirp"}:
                    u_current = generator(i, traj_i)
                else:  # 'random'
                    u_current = (np.random.rand(self.udim) - 0.5) * 2 * 0.5
                # d. 存储数据点 (u_i, s_i)
                data_array[traj_i, i, :] = np.concatenate([u_current, s_next])

        return data_array

    def generate_and_save_data(self):
        """生成并保存所有数据文件（逻辑与您原始代码一致）。"""
        data_dir_save = self.args.data_dir_save
        if not os.path.exists(data_dir_save):
            os.makedirs(data_dir_save)
        os.makedirs(data_dir_save, exist_ok=True)
        train_data_path = self.args.data_dir_load_train
        val_data_path = self.args.data_dir_load_val
        # 参数
        test_samples, test_steps = self.args.test_samples, self.args.test_steps
        train_samples, train_steps = self.args.train_samples, self.args.train_steps

        # 训练/验证/测试数据都做了"文件名里带样本数+步数"的缓存：只要这两个超参
        # 没变，重复运行 train.py 不会重新跑仿真采集，直接复用磁盘上的 .npy。
        # 想让新的 perturbation_limits 生效，必须换新的样本数/步数或手动删缓存文件。
        if os.path.exists(train_data_path):
            self.train_data = np.load(train_data_path)
        else:
            # 生成训练数据 (random)
            print(f"生成训练数据: {train_samples}条轨迹，每条{train_steps}步")
            self.train_data = self.generate_physics_based_data(train_samples, train_steps, "random")
            np.save(train_data_path, self.train_data)
            print(f"训练数据保存到: {train_data_path}, 形状: {self.train_data.shape}")

        if os.path.exists(val_data_path):
            self.val_data = np.load(val_data_path)
        else:
            # 生成验证数据 (random)
            print(f"生成验证数据: {test_samples}条轨迹，每条{test_steps}步")
            self.val_data = self.generate_physics_based_data(test_samples, test_steps, "random")
            np.save(val_data_path, self.val_data)
            print(f"验证数据保存到: {val_data_path}, 形状: {self.val_data.shape}")

        self.test_data_dict = {}
        # 生成测试数据 (三种类型)：random 用来看常规多步预测误差，
        # sin/chirp 用来专门检验模型对结构化输入的泛化/跟踪能力。
        for test_type in ["random", "sin", "chirp"]:
            test_data_path = os.path.join(
                project_root,
                self.args.env,
                "data",
                f"test_data_{test_type}_{self.args.test_samples}_{self.args.test_steps}.npy",
            )
            if os.path.exists(test_data_path):
                test_data = np.load(test_data_path)
            else:
                print(f"生成'{test_type}'测试数据: {test_samples}条轨迹，每条{test_steps}步")
                test_data = self.generate_physics_based_data(test_samples, test_steps, test_type)
                np.save(test_data_path, test_data)
                print(f"'{test_type}'测试数据保存到: {test_data_path}, 形状: {test_data.shape}")
            self.test_data_dict[test_type] = test_data
        # 关闭环境
        self.env.close()

    def get_train_loader(self):
        """把缓存/新生成的训练、验证 numpy 数组包装成 DataLoader，交给 train.py 使用。"""
        train_data_tensor = torch.tensor(self.train_data, dtype=torch.float32)
        train_dataset = TensorDataset(train_data_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, collate_fn=self.collate_fn, shuffle=True, drop_last=True
        )

        val_data_tensor = torch.tensor(self.val_data, dtype=torch.float32)
        val_dataset = TensorDataset(val_data_tensor)
        val_loader = DataLoader(
            val_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.collate_fn, shuffle=False
        )

        return train_loader, val_loader

    def get_test_loader(self, test_type):
        """按类型（random/sin/chirp）取出对应测试集并包装成 DataLoader。"""
        test_data = self.test_data_dict[test_type]
        test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
        test_dataset = TensorDataset(test_data_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.collate_fn, shuffle=False
        )

        return test_loader


def visualize_3d_trajectory(data_generator, cartesian_points, traj_type="train"):
    """
    可视化指定类型和索引的轨迹数据中的三维位置。

    Args:
        data_generator: 您的 SOARM101DataGenerator 实例。
        traj_type (str): 要可视化的轨迹类型 ('random', 'sin', 'chirp')。
        traj_index (int): 要可视化的轨迹在批次中的索引。
    """

    # 1. 获取数据
    if traj_type in ["random", "sin", "chirp"]:
        data_array = data_generator.test_data_dict.get(traj_type)
        if data_array is None:
            print(f"警告：找不到 {traj_type} 类型的测试数据。请确认数据已生成。")
            return
    elif traj_type == "train":
        data_array = data_generator.train_data
    else:
        print(f"无效的轨迹类型: {traj_type}")
        return

    # 只取每条轨迹的第 0 个时间步 (t=0)，画出的是"所有轨迹初始末端位置"的
    # 散点分布，用来直观检查 reset() 的随机扰动是否把工作空间覆盖得足够均匀。
    reshaped_data = data_array[:, 0, :]

    # 2. 提取 X, Y, Z 坐标
    # 提取所有轨迹的 X, Y, Z 坐标
    X = reshaped_data[:, 0]
    Y = reshaped_data[:, 1]
    Z = reshaped_data[:, 2]

    # 3. 绘制三维图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制点云
    # c=Z：颜色根据 Z 坐标变化，s=2：点的大小为 2
    scatter = ax.scatter(X, Y, Z, c=Z, cmap="viridis", s=0.1, alpha=1.0)
    # 添加颜色条（Color Bar），显示 Z 坐标的映射
    fig.colorbar(scatter, ax=ax, label="Z Position")
    # 绘制轨迹（对比：叠加显示 MPC 要跟踪的参考笛卡尔轨迹，
    # 和上面训练数据初始点云的分布范围是否匹配）
    ax.plot(cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2])
    # 设置坐标轴标签
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Data Acquisition Distribution ({traj_type})")
    plt.show()


# --- 主数据收集脚本 ---
# 下面这部分只在直接运行 `python SOARM101_DataCollection.py` 时执行，
# 用途是：(1) 触发一次完整的数据生成/缓存加载；(2) 顺带生成一条参考笛卡尔轨迹，
# 把训练数据点云和这条轨迹画在一起，快速用肉眼检查数据覆盖范围是否够用。
# 正常训练流程走的是 train.py，不会执行到这里。
if __name__ == "__main__":
    from control.TrajectoryGenerator import CartesianTrajectoryGenerator, CartesianTrajectoryGenerator_pinocchio

    # 1. 配置参数
    args = Args()
    # 2. 创建生成器实例
    data_generate = SOARM101DataGenerator(args)
    # 3. 执行数据生成和保存
    data_generate.generate_and_save_data()
    print("所有数据生成任务已完成！")
    print(data_generate.train_data[0])
    train_loader, val_loader = data_generate.get_train_loader()
    print("训练数据加载成功")
    for test_type in ["random", "sin", "chirp"]:
        test_loader = data_generate.get_test_loader(test_type)
    print("测试数据加载成功")

    MODEL_XML_PATH = args.xml_path
    EE_SITE_NAME = "gripperframe"  # 你的XML里定义的夹爪中心的 <site>
    NUM_JOINTS = args.u_dim  # 你的机器人关节数量
    use_pinocchio = False
    # 创建生成器实例
    # 两条分支功能等价（都是"生成笛卡尔参考轨迹 + 逆运动学解出关节轨迹"），
    # 区别只是 IK 求解器不同：CasADi 版本是项目自带的轻量实现，
    # pinocchio 版本依赖外部 pinocchio.casadi 绑定（默认不启用，见 README 踩坑记录）。
    if not use_pinocchio:
        traj_generator = CartesianTrajectoryGenerator(
            model_path=MODEL_XML_PATH,
            ee_site_name=EE_SITE_NAME,
            num_joints=NUM_JOINTS,
            idx=1,
            time_horizon=60,
            time_steps_per_sec=5,
        )
        # 定义末端执行器在整个轨迹中要保持的姿态 (例如，垂直向下)
        target_quat = None  # 绕X轴旋转90度
        # target_quat = np.array([0, 0, 1, 0])
        # 调用generate方法，反解出关节角度
        cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
            traj_name="Fig8",  # Circle, Fig8
            target_orientation=target_quat,
        )
    else:
        ARM_XML_PATH = args.arm_xml_path
        traj_generator = CartesianTrajectoryGenerator_pinocchio(
            arm_model_path=ARM_XML_PATH,
            ee_site_name=EE_SITE_NAME,
            num_joints=NUM_JOINTS,
            idx=1,
            time_horizon=60,
            time_steps_per_sec=5,
        )
        # 角度（度）
        angle_degrees = 0  # 0  90.0
        # 转换为弧度
        angle_radians = math.radians(angle_degrees)
        # 计算 cos 和 sin 值
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        # 构建绕 X 轴旋转的矩阵
        target_orientation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        # 调用generate方法，它会完成笛卡尔轨迹生成和IK求解两项工作
        cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
            traj_name="Fig8",  # Circle, Fig8
            target_orientation_matrix=target_orientation,
        )

    print("\n正在生成三维轨迹可视化图...")
    visualize_3d_trajectory(data_generate, cartesian_points, traj_type="train")

    print(joint_angle_traj)
    N_joints = joint_angle_traj.shape[1]
    plt.figure(figsize=(10, 6))
    # 遍历并绘制每个关节的角度轨迹
    for i in range(N_joints):
        # joint_angle_traj[:, i] 选择了所有时间步的第 i 个关节数据
        plt.plot(time_vec, joint_angle_traj[:, i], label=f"Joint {i + 1} Angle")
    # 设置图表标题和标签
    plt.title("Five Joint Angle Trajectories Over Time")
    plt.xlabel("Time (s)")  # 假设 time_vec 是以秒为单位
    plt.ylabel("Joint Angle (rad)")  # 假设关节角度是以弧度为单位

    # 显示图例、网格并调整布局
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
