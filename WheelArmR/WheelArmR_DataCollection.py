import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import matplotlib.pyplot as plt

from args import Args
from WheelArmR.WheelArmR_Env import WheelArmREnv


# 训练用的 "random" 噪声激励振幅 (目标关节速度, rad/s)。原来照抄 SO-101 用的是
# ±0.5——实测过 ±0.5 在 40 步(0.8s)内会让 Link4_R 平均摆动 1.15 rad、末端笛卡尔
# 位置平均摆动 0.77m，比正前方 Fig8 任务实际需要的尺度(~0.1rad / ~0.08m)大了近
# 一个数量级。训练数据的运动尺度和部署任务尺度不匹配，会稀释模型在小范围内的
# 局部精度。按比例缩小到量级匹配的值，重新采集/训练。
RANDOM_EXCITATION_AMPLITUDE = 0.08


class Collater:
    def __init__(self, x_dim: int, u_dim: int, device: str = "cuda"):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.device = device

    def __call__(self, batch_list: list):
        batch_data = list(zip(*batch_list))[0]
        batch_data = torch.stack(batch_data, dim=0)
        return dict(
            x=batch_data[:, :, self.u_dim : self.u_dim + self.x_dim].to(self.device),
            u=batch_data[:, :, : self.u_dim].to(self.device),
        )


class SineInputGenerator:
    def __init__(self, traj_num, udim=7, freq_range=(0.01, 0.05), amp_range=(0.05, 0.15), sine_mask=None, mode="sin"):
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

        self.freq_table = np.random.uniform(freq_range[0], freq_range[1], size=(traj_num, udim))
        self.amp_table = np.random.uniform(amp_range[0], amp_range[1], size=(traj_num, udim))
        self.phase_table = np.random.uniform(0, 2 * np.pi, size=(traj_num, udim))

        if sine_mask is None:
            self.sine_mask = np.ones((traj_num, udim), dtype=bool)
        else:
            self.sine_mask = sine_mask.astype(bool)

    def __call__(self, t, traj_i, T_total=200):
        if self.mode == "sin":
            freq = self.freq_table[traj_i]
        elif self.mode == "chirp":
            ratio = t / T_total
            freq = self.freq_table[traj_i] + (self.freq_end - self.freq_start) * ratio

        amp = self.amp_table[traj_i]
        phase = self.phase_table[traj_i]
        mask = self.sine_mask[traj_i]

        u = np.zeros(self.udim)
        sine_values = amp * np.sin(2 * np.pi * freq * t + phase)
        u[mask] = sine_values[mask]
        return u


# --- 数据生成器 (wheel_robot 右臂, 7自由度) ---


class WheelArmRDataGenerator:
    def __init__(self, args: Args) -> None:
        self.args = args
        self.udim = self.args.u_dim  # 控制维度
        self.xdim = self.args.x_dim  # Koopman模型所需的状态维度

        print("正在初始化物理仿真环境...")
        self.env = WheelArmREnv(xml_path=self.args.xml_path)
        print("物理环境初始化完成。")
        self.collate_fn = Collater(self.args.x_dim, self.args.u_dim, self.args.device)

    def generate_physics_based_data(self, traj_num, steps, input_type):
        """
        使用MuJoCo物理引擎生成数据。
        数据格式: [traj_num, steps+1, u_dim + x_dim] = [traj_num, steps+1, 7+10=17]
        """
        if input_type in {"sin", "chirp"}:
            generator = SineInputGenerator(
                traj_num=traj_num, udim=self.udim, freq_range=(0.0025, 0.05), amp_range=(-0.5, 0.5), mode=input_type
            )

        data_array = np.empty((traj_num, steps + 1, self.udim + self.xdim))

        for traj_i in tqdm(range(traj_num), desc=f"Generating '{input_type}' data"):
            s0, _ = self.env.reset()
            if input_type in {"sin", "chirp"}:
                u0 = generator(0, traj_i)
            else:  # 'random'
                u0 = (np.random.rand(self.udim) - 0.5) * 2 * RANDOM_EXCITATION_AMPLITUDE

            data_array[traj_i, 0, :] = np.concatenate([u0, s0])

            for i in range(1, steps + 1):
                u_prev = data_array[traj_i, i - 1, : self.udim]

                s_next, _, _, _, _ = self.env.step(u_prev)

                if input_type in {"sin", "chirp"}:
                    u_current = generator(i, traj_i)
                else:  # 'random'
                    u_current = (np.random.rand(self.udim) - 0.5) * 2 * RANDOM_EXCITATION_AMPLITUDE
                data_array[traj_i, i, :] = np.concatenate([u_current, s_next])

        return data_array

    def generate_and_save_data(self):
        """生成并保存所有数据文件（逻辑与 SOARM101DataGenerator 一致）。"""
        data_dir_save = self.args.data_dir_save
        if not os.path.exists(data_dir_save):
            os.makedirs(data_dir_save)
        os.makedirs(data_dir_save, exist_ok=True)
        train_data_path = self.args.data_dir_load_train
        val_data_path = self.args.data_dir_load_val
        test_samples, test_steps = self.args.test_samples, self.args.test_steps
        train_samples, train_steps = self.args.train_samples, self.args.train_steps

        if os.path.exists(train_data_path):
            self.train_data = np.load(train_data_path)
        else:
            print(f"生成训练数据: {train_samples}条轨迹，每条{train_steps}步")
            self.train_data = self.generate_physics_based_data(train_samples, train_steps, "random")
            np.save(train_data_path, self.train_data)
            print(f"训练数据保存到: {train_data_path}, 形状: {self.train_data.shape}")

        if os.path.exists(val_data_path):
            self.val_data = np.load(val_data_path)
        else:
            print(f"生成验证数据: {test_samples}条轨迹，每条{test_steps}步")
            self.val_data = self.generate_physics_based_data(test_samples, test_steps, "random")
            np.save(val_data_path, self.val_data)
            print(f"验证数据保存到: {val_data_path}, 形状: {self.val_data.shape}")

        self.test_data_dict = {}
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
        self.env.close()

    def get_train_loader(self):
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
        test_data = self.test_data_dict[test_type]
        test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
        test_dataset = TensorDataset(test_data_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.collate_fn, shuffle=False
        )

        return test_loader


def visualize_3d_trajectory(data_generator, cartesian_points, traj_type="train"):
    """可视化指定类型轨迹数据中的三维末端位置分布。"""

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

    reshaped_data = data_array[:, 0, :]

    X = reshaped_data[:, 0]
    Y = reshaped_data[:, 1]
    Z = reshaped_data[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(X, Y, Z, c=Z, cmap="viridis", s=0.1, alpha=1.0)
    fig.colorbar(scatter, ax=ax, label="Z Position")
    ax.plot(cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Data Acquisition Distribution ({traj_type})")
    plt.show()


# --- 主数据收集脚本 ---
if __name__ == "__main__":
    from control.TrajectoryGenerator import CartesianTrajectoryGenerator

    args = Args()
    data_generate = WheelArmRDataGenerator(args)
    data_generate.generate_and_save_data()
    print("所有数据生成任务已完成！")
    print(data_generate.train_data[0])
    train_loader, val_loader = data_generate.get_train_loader()
    print("训练数据加载成功")
    for test_type in ["random", "sin", "chirp"]:
        test_loader = data_generate.get_test_loader(test_type)
    print("测试数据加载成功")

    MODEL_XML_PATH = args.xml_path
    EE_SITE_NAME = "eeframe"
    JOINT_NAMES = [f"Link{i}_R_joint" for i in range(1, 8)]
    NUM_JOINTS = args.u_dim

    traj_generator = CartesianTrajectoryGenerator(
        model_path=MODEL_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS,
        joint_names=JOINT_NAMES,
        idx=1,
        time_horizon=60,
        time_steps_per_sec=5,
    )
    target_quat = None
    cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
        traj_name="Helix",
        target_orientation=target_quat,
    )

    print("\n正在生成三维轨迹可视化图...")
    visualize_3d_trajectory(data_generate, cartesian_points, traj_type="train")

    print(joint_angle_traj)
    N_joints = joint_angle_traj.shape[1]
    plt.figure(figsize=(10, 6))
    for i in range(N_joints):
        plt.plot(time_vec, joint_angle_traj[:, i], label=f"Joint {i + 1} Angle")
    plt.title("Wheel Arm R Joint Angle Trajectories Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")

    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
