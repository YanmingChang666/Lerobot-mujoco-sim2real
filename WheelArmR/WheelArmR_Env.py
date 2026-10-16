import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class WheelArmREnv(gym.Env):
    """
    一个专门为Koopman算子训练收集数据的 wheel_robot 右臂环境 (Link1_R..Link7_R, 7自由度)。

    状态 x (Observation):
    - 3x 末端执行器位置 (m)
    - 7x 关节角度 (rad)

    控制输入 u (Action):
    - 7x 关节目标速度 (rad/s)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path: str, dt: float = 0.02, render_mode=False):
        """
        Args:
            xml_path (str): MuJoCo模型的XML文件路径。
            dt (float): 环境的控制时间步长 (s)。
            render_mode (Optional[str]): 渲染模式，'human'开启可视化。
        """
        super().__init__()

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"MuJoCo XML 文件未找到: {xml_path}")

        self.frame_skip = max(1, int(np.round(dt / self.model.opt.timestep)))
        self.dt = self.model.opt.timestep * self.frame_skip
        print(f"环境控制步长(dt): {self.dt:.4f}s (执行 {self.frame_skip} 个物理步骤)")

        self.data = mujoco.MjData(self.model)

        # --- 关节和末端执行器定义 ---
        self.joint_names = [f"Link{i}_R_joint" for i in range(1, 8)]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]

        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eeframe")
        except ValueError:
            raise ValueError("在模型中未找到名为 'eeframe' 的 site。请检查XML文件。")

        # --- 状态与控制输入定义 ---
        self.udim = 7
        self.max_speed = 0.5  # 关节目标速度上限 (rad/s)，与执行器 ctrlrange 分开，留出余量
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(self.udim,), dtype=np.float32)
        # 状态 x (Observation): [ee_pos(3), qpos(7)]
        self.xdim = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.xdim,), dtype=np.float32)

        # --- 渲染相关 ---
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.initial_poses = [
            np.zeros(7),  # 姿态1: 零位 (home)
        ]
        # 每个关节的扰动范围 [low, high]。Link1~4_R 的范围已经按 wheelarm_sim2sim.py 里
        # "正前方" Fig8/Rectangle/Heart (idx=1, 中心 x=0.35 y=-0.25 z=0.25, 从肘部预弯
        # [0,0,0,-0.6,0,0,0] 起步求解 IK) 实际需要的关节角 (逐点用 IK 反解出来, 留了余量)
        # 设定, 保证训练数据覆盖验证轨迹; Link5/6/7 是与 Link1_R_joint(Y)-Link6_R_joint(Z)-
        # Link7_R_joint(X) 三轴汇交于 eeframe 的球型手腕, 对末端位置无影响(只影响姿态),
        # 扰动范围只是为了让姿态有多样性, 数值本身不重要。换轨迹形状/中心点后如果跟踪误差
        # 偏大, 参照 README 里 "扩大扰动范围" 一节的方法重新按新轨迹的关节角范围迭代这组数值。
        self.perturbation_limits = np.array(
            [
                [-0.3, -0.9, -0.3, -2.3, -0.78, -0.26, -0.39],  # 每个关节的扰动下限
                [0.3, -0.1, 0.3, -1.6, 0.78, 0.26, 0.39],  # 每个关节的扰动上限
            ]
        )

    def _get_state(self) -> np.ndarray:
        """获取当前系统的状态 x。"""
        qpos = self.data.qpos[self.joint_ids].copy()
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

        if options and "initial_state" in options:
            initial_qpos = options["initial_state"][: self.udim]
            initial_qvel = options["initial_state"][self.udim : 2 * self.udim]
        else:
            selected_pose_idx = self.np_random.integers(len(self.initial_poses))
            base_qpos = self.initial_poses[selected_pose_idx].copy()

            perturbation = self.np_random.uniform(
                low=self.perturbation_limits[0],
                high=self.perturbation_limits[1],
                size=self.udim,
            )
            initial_qpos = base_qpos + perturbation
            initial_qvel = np.zeros(self.udim)

        self.data.qpos[self.joint_ids] = initial_qpos
        self.data.qvel[self.joint_ids] = initial_qvel

        mujoco.mj_forward(self.model, self.data)

        initial_state = self._get_state()

        return initial_state, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        施加控制输入 u (action)，并返回系统的下一个状态 x'。

        Args:
            action (np.ndarray): 控制输入 u (目标关节速度, rad/s)。

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
            (next_state, 0.0, False, False, {})
        """
        target_velocity = action[: self.udim]
        self.data.ctrl[: self.udim] = target_velocity

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        next_state = self._get_state()

        if self.render_mode:
            self.render()

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
