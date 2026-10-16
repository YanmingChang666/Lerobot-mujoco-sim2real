"""
SO-ARM101 的 MuJoCo 仿真环境封装。

这个环境不是用来做强化学习的（虽然继承了 gym.Env、定义了 action_space /
observation_space），它唯一的用途是给 Koopman 动力学建模“采数据”：
在关节零位附近施加随机扰动作为初始状态，再喂随机/预设的控制序列，
记录 (x_t, u_t, x_{t+1}) 三元组，供 SOARM101_DataCollection.py 采集、
train.py 训练 Koopman 网络时使用。

因此 reward / terminated / truncated 这些 Gym 标准返回值全部是占位的
固定值 (0.0, False, False)，不代表真的有奖励函数或终止条件。
"""

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class SOARM101Env(gym.Env):
    """
    一个专门为Koopman算子训练收集数据的SOARM-101环境。

    状态 x (Observation)：Koopman 建模里的“系统状态”，8 维 = 3(末端位置) + 5(关节角)。
    把末端笛卡尔位置也塞进状态里，是因为 MPC/轨迹跟踪最终关心的是末端在
    笛卡尔空间的误差，让 Koopman 模型直接预测末端位置，比训练后再算一次
    正运动学更贴近下游用途，也让线性/双线性 Koopman 算子能直接对末端位置
    的演化建模。
    - 5x 关节角度 (rad)
    - 3x 末端执行器位置 (m)

    控制输入 u (Action)：
    - 5x 关节目标速度 ([-2, 2])
    """

    # 尽管不用于RL，但提供元数据是好习惯
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path: str, dt: float = 0.02, render_mode=False):
        """
        Args:
            xml_path (str): MuJoCo模型的XML文件路径。
            dt (float): 环境的控制时间步长 (s)。
            render_mode (Optional[str]): 渲染模式，'human'开启可视化。
        """
        super().__init__()

        # --- MuJoCo 初始化 ---
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"MuJoCo XML 文件未找到: {xml_path}")

        # MuJoCo 物理引擎自身的仿真步长(model.opt.timestep)通常比控制步长(dt)小很多，
        # 需要用 frame_skip 把多步物理仿真"打包"成一次 Koopman/MPC 控制周期，
        # 保证采集到的数据步长和实际控制频率一致，模型学到的才是"控制步长下"的动力学。
        self.frame_skip = max(1, int(np.round(dt / self.model.opt.timestep)))
        self.dt = self.model.opt.timestep * self.frame_skip
        print(f"环境控制步长(dt): {self.dt:.4f}s (执行 {self.frame_skip} 个物理步骤)")

        self.data = mujoco.MjData(self.model)

        # --- 关节和末端执行器定义 ---
        # 顺序必须和 XML 里的执行器顺序、以及后面 sim2real 端的 joint_offsets 顺序保持一致，
        # 否则会出现"关节角度对上了但对应错了关节"的隐蔽 bug。
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]

        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        except ValueError:
            raise ValueError("在模型中未找到名为 'gripper' 的 site。请检查XML文件。")

        # --- 状态与控制输入定义 ---(目前无用，保留强化学习拓展)
        # 控制输入 u (Action): 5个关节的目标速度
        self.udim = 5
        self.max_speed = 0.5  # 设定一个物理上的最大关节速度 (rad/s)
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(self.udim,), dtype=np.float32)
        # 状态 x (Observation): [qpos, ee_pos]
        self.xdim = 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.xdim,), dtype=np.float32)

        # --- 渲染相关 ---
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # reset() 采数据时的"基准姿态"候选池：每次 reset 从这里随机选一个，
        # 再叠加 perturbation_limits 范围内的随机扰动，得到真正的初始关节角。
        # 之所以要多个基准姿态(目前只启用了零位一个)，是为了让训练数据覆盖更大的
        # 工作空间，而不是只在零位附近打转。
        self.initial_poses = [
            np.array([0, 0, 0, 0, 0]),  # 姿态1: 零位
            # np.array([0,-0.33, 0.29 ,0.20, -1.52])  # 姿态2: 另一个特定姿态
        ]
        # 每个关节独立的扰动范围 [low, high]，决定了训练数据在关节空间的覆盖范围。
        # 这个范围极其关键：Koopman 模型只在训练数据覆盖过的关节角范围内才可靠，
        # 一旦下游轨迹跟踪(比如贴桌面画 Fig8/Heart)要求的关节角超出这个范围，
        # 模型会做"外推"，跟踪精度会明显变差(这是 README 里踩过的坑，不是 bug)。
        # 想让模型支持更大的工作空间，就要扩大这里的范围、重新采集数据、重新训练。
        self.perturbation_limits = np.array(
            [
                # [-0.3, -1.0, -0.6, -0.6, -0.1],  # 旧版本注释掉的更宽范围
                # [ 0.3,  0.3,  0.8,  0.4,  0.6]
                [-0.45, -1.35, -1.25, -0.75, -0.3],  # 每个关节的扰动下限
                [0.45, 1.45, 0.85, 0.85, 0.3],  # 每个关节的扰动上限
            ]
        )

    def _get_state(self) -> np.ndarray:
        """获取当前系统的状态 x：拼接末端位置(3)和关节角(5)，顺序固定为 [ee_pos, qpos]。"""
        qpos = self.data.qpos[self.joint_ids].copy()
        # qvel = self.data.qvel[self.joint_ids].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        return np.concatenate([ee_pos, qpos]).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """
        重置环境到一个初始状态。

        Args:
            seed: 随机种子。
            options (dict, optional): 可选参数，例如可以传入一个 'initial_state'。
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 允许通过 options 指定一个初始状态
        if options and "initial_state" in options:
            initial_qpos = options["initial_state"][:5]
            initial_qvel = options["initial_state"][5:10]
        else:
            # 默认重置到一个随机的初始角度和零速度
            # 1. 从预定义的初始姿态中随机选择一个
            selected_pose_idx = self.np_random.integers(len(self.initial_poses))
            base_qpos = self.initial_poses[selected_pose_idx].copy()  # 使用.copy()是个好习惯

            # 2. 生成每个关节独立的随机扰动
            # np_random.uniform 可以直接接收 low 和 high 数组
            perturbation = self.np_random.uniform(
                low=self.perturbation_limits[0],  # 所有关节的下限
                high=self.perturbation_limits[1],  # 所有关节的上限
                size=self.udim,
            )
            # 3. 将基础姿态和扰动相加，得到最终的初始关节角度
            # 这一步就是在给 Koopman 训练数据"撒点"：每次 reset 都从一个不同的
            # 关节角组合出发，让采集到的 (x, u, x') 轨迹尽量铺满关节空间。
            initial_qpos = base_qpos + perturbation
            initial_qvel = np.zeros(self.udim)

        self.data.qpos[self.joint_ids] = initial_qpos
        self.data.qvel[self.joint_ids] = initial_qvel

        # 必须调用 mj_forward 来更新所有派生量 (如 site_xpos)
        # 只改 qpos/qvel 不会自动重算正运动学，不调用这一步 ee_pos 会还是重置前的旧值。
        mujoco.mj_forward(self.model, self.data)

        initial_state = self._get_state()

        return initial_state, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        施加控制输入 u (action)，并返回系统的下一个状态 x'。

        Args:
            action (np.ndarray): 控制输入 u, 归一化到 [-2, 2]。

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
            (next_state, 0.0, False, False, {})
        """
        # 重力补偿
        # self.data.qfrc_applied[:] =  self.data.qfrc_bias[:]
        # 1. 将归一化的 action 缩放到真实的物理单位 (目标速度)
        target_velocity = action[: self.udim]
        # 2. 将目标速度施加到控制器
        # 注意：这里我们使用的是速度控制，所以直接设置 qvel 是不正确的。
        # 应该使用 MuJoCo 的执行器。假设XML中定义了 velocity 类型的执行器。
        # 如果是力矩控制，就设置 data.ctrl 为力矩。
        # 为简单起见，我们假设XML中配置了 velocity 执行器，其输入直接是目标速度。
        self.data.ctrl[: self.udim] = target_velocity

        # 3. 执行物理模拟
        # 循环 frame_skip 次单步物理仿真，让"一次 step 调用"对应一个完整的
        # 控制周期 dt，而不是底层物理引擎的一个仿真步——这正是 __init__ 里
        # frame_skip 的作用：把细粒度物理仿真聚合成粗粒度的控制/采样步长。
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 4. 获取模拟后的新状态
        next_state = self._get_state()

        # 5. 渲染 (如果需要)
        if self.render_mode:
            self.render()

        # 6. 返回标准 Gym 格式，但奖励和 done 信号是固定的
        # (这不是强化学习环境，reward/terminated/truncated 只是为了兼容 Gym 接口占位)
        return next_state, 0.0, False, False, {}

    def render(self):
        """渲染环境的当前帧。"""
        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

    def close(self):
        """关闭环境并释放资源。"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
