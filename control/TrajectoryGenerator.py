"""
"路径规划"模块：负责生成机械臂末端要跟踪的参考轨迹。

这是本项目里唯一真正做"轨迹/路径规划"的地方，和 Koopman 建模、MPC 控制
是两个独立关注点：这里只管"末端应该沿着什么样的几何曲线走"，产出
笛卡尔坐标点序列 (N,3)，再逐点求逆运动学(IK)转换成关节角序列 (N,num_joints)；
真正让机械臂沿着这条轨迹走准，是 control/MPC_Controler.py 的职责。

提供两套功能等价、只是 IK 后端不同的实现：
- CartesianTrajectoryGenerator：用 dm_control 的 qpos_from_site_pose 做 IK，
  是默认/主要使用的版本 (DBKN_KESO.py、IBKN_UKF.py、write_demo.py 都用它)。
- CartesianTrajectoryGenerator_pinocchio：用 Pinocchio+CasADi (control/casadi_ik.py)
  做 IK，依赖额外的 pinocchio.casadi 绑定，默认不启用 (见 README 踩坑记录)。

支持的轨迹形状：Fig8(8字/双纽线)、Rectangle(矩形)、Heart(心形)、
Helix(三维螺旋)、Lissajous(李萨如曲线)、FigStar(五角星)，以及
generate_multi_stroke() 生成的"多笔画写字"轨迹(抬笔->平移->落笔)。

============================================================
【写字相关速览】 (generate_multi_stroke 及其依赖的 IK 函数, 下面的注释主要围绕这条链路)
============================================================
调用链:
    generate_multi_stroke(strokes, center, write_height, lift_height, ...)
        ① 把二维笔画逐笔展开成 3D 点序列 xyz_coords, 同时记录 pen_down_mask
              落笔段 : z = write_height, 沿笔画折线逐段插值
              笔画间 : 抬笔(z 升到 lift_height) -> 平移(保持 lift_height) -> 落笔(z 降回 write_height)
        ② _solve_ik_trajectory(xyz_coords, ...) 对每个点解逆运动学, 得到关节角序列
              └─ _solve_ik(pos, quat) 调 dm_control 的 qpos_from_site_pose 解单个点
        ③ 返回 (xyz_coords, 关节角轨迹, pen_down_mask)

阅读源码得到的几个细节 (使用时值得留意):
    - 一个点 = 一个仿真控制步 (50Hz, 0.02s)。点数就是播放时长, 与笔画的物理长度无关。
    - points_per_stroke_segment 是"笔画折线上每一小段"的插值点数, 不是"每一笔"的点数:
      折线段越长, 末端速度越快; 越短, 速度越慢。笔画中线的关键点分布不均时, 书写速度会忽快忽慢。
    - 相邻两段折线的接点会重复出现一次 (前一段的终点 t=1 与后一段的起点 t=0 是同一个点),
      接点处参考轨迹会有一个"停顿"点。
    - IK 逐点求解, 每个点用上一个点的解做初值 (热启动), 保证关节角随点位连续变化。
    - IK 求解失败时用上一个成功的解顶替, 并只打印警告 —— 需要关注控制台里的"逆运动学...求解失败"。
    - dm_control 的 IK 不检查关节限位, 解出的角度可能超出 XML 里 <joint range>, 需要自己核对。
    - target_orientation=None 时只约束位置 (按 dm_control 的实现), wrist_roll 等于被 IK 自由支配,
      所以才需要 fixed_wrist_roll_rad 把它锁死。
"""

# sys    : IK 初始化失败时 sys.exit(1) 直接退出。
# numpy  : 轨迹点、关节角的数组运算。
import sys

import numpy as np

# 导入所有必要的库
# dm_control.mujoco.Physics : 对 MuJoCo 模型+数据的封装, qpos_from_site_pose 要求这种类型的输入。
# qpos_from_site_pose       : dm_control 自带的数值逆运动学: 给定 site(末端点)的目标位置(和姿态),
#                             迭代求解满足条件的关节角 (阻尼最小二乘/雅可比迭代)。
from dm_control.mujoco import Physics  # 关键：导入 dm_control 的 Physics 封装
from dm_control.utils.inverse_kinematics import qpos_from_site_pose  # 关键：导入 IK 函数


# --- 笛卡尔轨迹生成器 (使用 dm_control IK 的完整版本) ---
class CartesianTrajectoryGenerator:
    """
    生成笛卡尔空间(x, y, z)的轨迹，并同时计算出对应的关节角度。
    本版本使用 dm_control.inverse_kinematics.qpos_from_site_pose 进行逆运动学求解。

    典型调用顺序：__init__ 加载模型准备 IK 环境 -> generate()/generate_multi_stroke()
    产出 (笛卡尔点, 关节角轨迹, 时间向量) -> 交给 DBKN_KESO.py/IBKN_UKF.py 里的
    MPC 控制器逐点跟踪。
    """

    def __init__(
        self,
        model_path: str,
        ee_site_name: str,
        num_joints: int,
        idx=1,
        time_horizon=60,
        time_steps_per_sec=5,
        joint_names: list[str] | None = None,
    ):
        """
        初始化轨迹生成器。

        Args:
            model_path (str): MuJoCo XML模型的路径。
            ee_site_name (str): 末端执行器 (end-effector) 在XML中的 <site> 名称。
            num_joints (int): 机械臂的关节数量。
            idx (int): 轨迹平面设置 (0 for x-y plane, 1 for y-z plane)。
            time_horizon (float): 轨迹的总时长（秒）。
            time_steps_per_sec (int): 每秒的轨迹点数量。
            joint_names (list[str] | None): IK 可解的关节名列表。为 None 时默认为
                SO-101 的 5 个关节名（向后兼容现有调用方）；换机械臂时需要显式传入。
        """
        # 轨迹参数
        # idx 只影响 generate() 里 Fig8/Heart 等"单笔图形"画在哪个平面; 写字 (generate_multi_stroke) 直接按
        # (x, y, z) 生成点, 不使用 idx。
        self.idx = idx
        self.time_horizon = time_horizon
        # time_horizon / time_steps_per_sec 决定 generate() 画单笔图形的点数 (默认 60s*10 = 600 点)。
        # generate_multi_stroke 不依赖它们: 点数由笔画内容决定, 结束时会用 np.linspace 重新生成 time_vector。
        self.time_steps = time_steps_per_sec * time_horizon
        self.time_vector = np.linspace(0, self.time_horizon, self.time_steps)

        # 机器人和IK参数
        self.model_path = model_path
        self.ee_site_name = ee_site_name
        self.num_joints = num_joints
        # 保存用户指定的 IK 关节名 (None 表示用 SO-101 默认的 5 个关节)。
        self._joint_names_override = joint_names

        # 这些参数可以根据您的机器人工作空间进行调整
        self.traj_scale = 0.5

        # 初始化IK求解器所需的MuJoCo模型和数据
        # 构造时立即加载一份用于 IK 的 MuJoCo 模型; 之后 generate*/_solve_ik* 都直接复用 self.physics。
        self._initialize_ik_solver()

    def _initialize_ik_solver(self):
        """加载模型并准备IK计算环境。"""
        print("正在为IK求解器初始化MuJoCo模型...")
        try:
            # 使用 dm_control 的 Physics 对象封装模型和数据
            # qpos_from_site_pose 函数需要这个类型的输入
            # 这是一份专门做 IK 的独立 MuJoCo 模型副本, 与 SOARM101Env 里的仿真环境、可视化窗口互不影响:
            # IK 过程中反复修改 self.physics.data.qpos, 不会影响闭环仿真。
            self.physics = Physics.from_xml_path(self.model_path)

            # 获取所有可动的、非自由浮动的关节名称
            # 这对于调用qpos_from_site_pose至关重要
            # 参与 IK 求解的关节名, 顺序与 qpos 前 5 位一致 (qpos[0]=shoulder_pan ... qpos[4]=wrist_roll)。
            # generate_multi_stroke 里 "self.joint_names[:4]" 去掉 wrist_roll、以及 pinned_qpos={4: ...}
            # 这两处都依赖这个顺序 —— 换机械臂时务必保持"要锁定的关节"在对应位置。
            # gripper 不在列表里, 即 IK 不会动夹爪。
            self.joint_names = self._joint_names_override or [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]

            if not self.joint_names:
                raise RuntimeError("在模型中没有找到任何可驱动的关节。")

            print(f"找到 {len(self.joint_names)} 个可动关节: {self.joint_names}")
            if len(self.joint_names) < self.num_joints:
                print(
                    f"警告：模型中找到的可动关节数量 ({len(self.joint_names)}) 少于指定的 num_joints ({self.num_joints})"
                )

        except Exception as e:
            print(f"错误：无法从'{self.model_path}'加载IK模型。 {e}")
            sys.exit(1)

        # 检查 site 是否存在
        try:
            _ = self.physics.model.name2id(self.ee_site_name, "site")
        except KeyError:
            raise ValueError(f"错误: 在模型中找不到名为 '{self.ee_site_name}' 的 site。请检查XML文件。")

        # 设置一个合理的初始关节姿态 (通常是全零姿态)
        # 初始姿态: 所有关节角为 0 (nq 是广义坐标总数, 含夹爪)。generate 系列函数在开始求解前还会再设置一次初值。
        home_qpos = np.zeros(self.physics.model.nq)
        with self.physics.reset_context():
            self.physics.data.qpos[:] = home_qpos

        print("IK求解器初始化完成。")

    def _solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """
        内部IK求解函数, 使用 dm_control 的 qpos_from_site_pose。

        Args:
            target_pos (np.ndarray): 目标位置 [x, y, z]。
            target_quat (np.ndarray): 目标姿态四元数 [w, x, y, z]。

        Returns:
            np.ndarray: 求解出的关节角度（弧度），如果失败则返回 None。
        """
        # qpos_from_site_pose 会自动使用 self.physics.data.qpos 作为初始猜测值，
        # 这样可以利用上一步的解，提高求解速度和连续性。

        # 调用 dm_control 的 IK 函数
        # dm_control 逆运动学求解 (数值迭代, 非解析解)。关键点:
        #   - target_quat 为 None 时只约束末端"位置", 不约束姿态 (写字轨迹就是这种情况);
        #     只有同时给了 target_quat 才会用到 rot_weight。
        #   - joint_names 决定哪些关节允许被 IK 调整 —— 没列出的关节保持当前 qpos 不动。
        #     这正是"锁死 wrist_roll"的实现机制: 把它从 joint_names 里去掉, 它的 qpos 就保持预先设定的值。
        #   - inplace=True: 解直接写回 self.physics.data.qpos, 作为下一个点的初始猜测 (热启动)。
        #   - max_steps=100, tol=1e-6: 最多迭代 100 步, 位置误差小于 1e-6 视为成功。
        #   - regularization_strength: 阻尼项, 在靠近奇异构型时避免关节角突变。
        ik_result = qpos_from_site_pose(
            physics=self.physics,  # 传入 dm_control 的 Physics 对象
            site_name=self.ee_site_name,  # 目标 site 名称
            target_pos=target_pos,  # 目标位置
            target_quat=target_quat,  # 目标姿态
            joint_names=self.joint_names,  # 指定哪些关节可以动
            inplace=True,  # 设置为True，直接修改self.physics.data，效率更高
            max_steps=100,  # 最大迭代次数
            tol=1e-6,  # 求解精度阈值
            rot_weight=0.5,  # 姿态误差的权重 (0.0到1.0之间)
            regularization_strength=1e-2,  # 正则化强度，防止奇异点问题，增加稳定性
        )

        # 检查求解结果
        if ik_result.success:
            # 求解成功，返回与机械臂相关的关节角度
            # self.physics.data.qpos 已经被 inplace 修改，直接复制即可
            # 取前 num_joints(=5) 个关节角, 复制一份返回 (避免后续 IK 迭代原地修改影响已保存的结果)。
            # 注意: 即使 wrist_roll 被排除在 joint_names 之外, 它仍在 qpos[4] 里,
            # 所以返回的 5 维向量依然包含(被锁定的)wrist_roll 值。
            return self.physics.data.qpos[: self.num_joints].copy()
        else:
            # 如果求解失败，返回None
            return None

    def generate(
        self,
        traj_name="Fig8",
        target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        center=None,
        initial_qpos=None,
    ):
        """
        生成指定的笛卡尔轨迹和对应的关节角度轨迹。
        这个方法将一系列的笛卡尔坐标点，通过逆运动学，转换为一系列的关节角度。

        Args:
            traj_name (str): 轨迹名称 ('Fig8', 'Circle')。
            target_orientation (np.ndarray): 整个轨迹中末端执行器要保持的目标姿态 (四元数 [w, x, y, z])。
            center (tuple[float, float, float] | None): 覆盖默认中心点 (x, y, z)。为 None
                时沿用各自内部写死的中心 —— 那些是为 SO-101 的工作空间调的，换机械臂后
                默认值大概率不可达，务必传入针对新工作空间验证过的中心点。
            initial_qpos (np.ndarray | None): IK 求解的起始猜测姿态，见 `_solve_ik_trajectory`
                的说明——目标区域离零位较远时，从零位起步的 IK 容易收敛到超出物理限位的
                分支，需要给一个离目标更近的起始姿态。

        Returns:
            tuple: 包含三个元素的元组:
                - np.ndarray: 形状为 (N, 3) 的笛卡尔坐标点云。
                - np.ndarray: 形状为 (N, num_joints) 的关节角度轨迹（弧度）。
                - np.ndarray: 对应的时间向量。
        """
        # 1. 生成笛卡尔坐标点 (x, y, z)
        #    这部分定义了机械臂末端应该遵循的路径。
        t_param = 1.6 + 0.02 * np.linspace(0, self.time_horizon * 5, len(self.time_vector))
        print(f"正在生成 '{traj_name}' 笛卡尔轨迹 (平面设置 idx={self.idx})")

        if traj_name == "Fig8":
            # "8"字轨迹用的是伯努利双纽线(lemniscate)的参数方程变形：
            # 分母 (1+sin^2(t)) 让曲线在 t 增大时自然收窄成两个交叉的圆环，
            # 分子里 sin(t)*cos(t) 项使一个方向上呈现"8"字交叉、另一方向呈现
            # 单峰摆动，两者组合就是标准的双纽线轨迹；a/b 控制两个方向的振幅。
            if self.idx == 1:  # Y-Z平面
                a = 0.2 * self.traj_scale
                b = 0.2 * self.traj_scale
                center_x, center_y, center_z = center if center is not None else (0.4, 0.0, 0.2)
                x = center_x * np.ones((len(t_param), 1))
                z = np.expand_dims(
                    center_z + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1
                )
                y = np.expand_dims(center_y + b * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1)
            else:  # X-Y平面
                a = 0.2 * self.traj_scale
                b = 0.2 * self.traj_scale
                z = 0.055 * np.ones((len(t_param), 1))  # 贴近桌面高度, 与 Heart/Rectangle/FigStar 一致
                x = np.expand_dims(0.3 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1)
                y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1)
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Rectangle":
            # 矩形轨迹：先算出 4 个顶点，再在相邻顶点之间做线性插值连成 4 条边，
            # 是本文件里"关键点 + 分段线性插值"这种轨迹生成套路最简单的例子
            # （FigStar 用的是同一套路的五角星版本）。
            # --- 1. 定义几何参数 ---
            width = 0.25 * self.traj_scale
            height = 0.25 * self.traj_scale

            # 定义中心点
            if self.idx == 1:  # Y-Z 平面
                x_fixed, cx, cy = center if center is not None else (0.4, 0.0, 0.2)
            else:  # X-Y 平面
                cx, cy = 0.35, 0.0
                z_fixed = 0.055
            # --- 2. 定义 4 个顶点 (从左下角开始逆时针) ---
            # 顺序: 左下 -> 右下 -> 右上 -> 左上 -> 左下 (闭合)
            rect_points = np.array(
                [
                    [cx - width / 2, cy - height / 2],
                    [cx + width / 2, cy - height / 2],
                    [cx + width / 2, cy + height / 2],
                    [cx - width / 2, cy + height / 2],
                    [cx - width / 2, cy - height / 2],
                ]
            )
            # --- 3. 线性插值 ---
            num_steps = len(t_param)
            num_segments = 4  # 矩形有4条边
            refs = np.zeros((num_steps, 2))
            each_num = num_steps // num_segments
            current_step = 0

            for i in range(num_segments):
                start_pt = rect_points[i, :]
                end_pt = rect_points[i + 1, :]

                # 确保最后一段填满剩余步数
                n_points = each_num if i < num_segments - 1 else num_steps - current_step

                # 生成线性插值因子 (0 到 1)
                t_ = np.linspace(0, 1, n_points).reshape(-1, 1)
                segment_traj = (1 - t_) * start_pt + t_ * end_pt

                refs[current_step : current_step + n_points, :] = segment_traj
                current_step += n_points

            # --- 4. 组装坐标 ---
            if self.idx == 1:  # Y-Z 平面
                x = x_fixed * np.ones((num_steps, 1))
                y = refs[:, 0].reshape(-1, 1)
                z = refs[:, 1].reshape(-1, 1)
            else:  # X-Y 平面
                x = refs[:, 0].reshape(-1, 1)
                y = refs[:, 1].reshape(-1, 1)
                z = z_fixed * np.ones((num_steps, 1))
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Heart":
            # 心形参数方程用的是经典的"心形曲线"公式（16sin^3(t) 配合
            # 四个不同频率余弦项的叠加），比单纯用一个 cos/sin 更贴近直觉上
            # "心形"的尖端(t≈π处)和上方凹陷(t≈0处)形状，是描图类图形常用的
            # 现成公式，不是从机器人工作空间反推出来的。
            # 缩放系数 (心形方程生成的值较大，需要缩小)
            scale = 0.01 * self.traj_scale
            # 调整参数 t 使其覆盖 0 到 2pi (根据你的 t_param 范围可能需要调整)
            # 假设 t_param 是线性增加的，我们取模或归一化来画完整的圆
            t_circle = np.linspace(0, 2 * np.pi, len(t_param))
            # 心形参数方程
            # shape_x 对应水平宽，shape_y 对应垂直高（尖端在下，凹陷在上）
            shape_x = 16 * np.sin(t_circle) ** 3
            shape_y = 13 * np.cos(t_circle) - 5 * np.cos(2 * t_circle) - 2 * np.cos(3 * t_circle) - np.cos(4 * t_circle)
            shape_x = shape_x * scale
            shape_y = shape_y * scale

            if self.idx == 1:  # Y-Z平面
                center_x, center_y, center_z = center if center is not None else (0.4, 0.0, 0.2)
                x = center_x * np.ones((len(t_param), 1))
                # 注意：心形方程 y 轴对应竖直方向，所以映射到 Z，x 轴映射到 Y
                y = np.expand_dims(center_y + shape_x, axis=1)
                z = np.expand_dims(center_z + shape_y, axis=1)
            else:  # X-Y平面
                center_x, center_y = 0.35, 0.0
                z = 0.055 * np.ones((len(t_param), 1))
                # 旋转90度让心形正对
                x = np.expand_dims(center_x + shape_y, axis=1)  # 竖直方向映射到 X
                y = np.expand_dims(center_y + shape_x, axis=1)  # 水平方向映射到 Y
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Helix":
            # 唯一的三维轨迹：XY(或对应两轴)方向做匀速圆周运动，第三轴方向
            # 做线性往返推进，两者叠加形成螺旋线；t_param 的周期决定圈数
            # (t_param*2 的系数越大，同样长度里转的圈数越多)。
            radius = 0.1 * self.traj_scale
            # 螺旋的长度
            length = 0.2
            # 生成圆周运动
            # 这里的 t_param 直接用于周期，如果 t_param 范围大，螺旋圈数就多
            circle_1 = radius * np.cos(t_param * 2)
            circle_2 = radius * np.sin(t_param * 2)
            # 生成轴向推进 (往返运动，使用 sin 避免跳变，或者线性)
            # 这里使用线性往返：从 0 到 length 再回来
            # 为了简单，这里演示单向推进然后瞬移，或者用 np.linspace
            linear_move = np.linspace(-length / 2, length / 2, len(t_param))
            center_x, center_y, center_z = center if center is not None else (0.35, 0.0, 0.20)
            x = np.expand_dims(center_x + circle_1, axis=1)
            y = np.expand_dims(center_y + circle_2, axis=1)
            z = np.expand_dims(center_z + linear_move, axis=1)  # 高度变化
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Lissajous":
            # 李萨如曲线：两个方向分别用不同频率(a_freq:b_freq)、带相位差(delta)
            # 的正弦波，频率比 3:2 是产生"打结"形状的经典配置，频率比越复杂
            # 曲线自相交越多，是测试跟踪能力很常用的一种"结构化但非平凡"的轨迹。
            A = 0.12 * self.traj_scale
            B = 0.12 * self.traj_scale
            # 频率参数 (3:2 的比例会生成一个经典的纽结形状)
            a_freq = 3.0
            b_freq = 2.0
            delta = np.pi / 2  # 相位差

            # 归一化时间参数以保证闭环
            t_cycle = np.linspace(0, 2 * np.pi, len(t_param))

            val_1 = A * np.sin(a_freq * t_cycle + delta)
            val_2 = B * np.sin(b_freq * t_cycle)

            if self.idx == 1:  # Y-Z平面
                center_x, center_y, center_z = center if center is not None else (0.4, 0.0, 0.2)
                x = center_x * np.ones((len(t_param), 1))
                y = np.expand_dims(center_y + val_1, axis=1)
                z = np.expand_dims(center_z + val_2, axis=1)
            else:  # X-Y平面
                center_x, center_y = 0.35, 0.0
                z = 0.055 * np.ones((len(t_param), 1))  # 贴近桌面高度, 与 Heart/Rectangle/FigStar 一致
                x = np.expand_dims(center_x + val_2, axis=1)
                y = np.expand_dims(center_y + val_1, axis=1)
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "FigStar":
            # 五角星轨迹：不是连续曲线方程，而是先算出 11 个关键点(5个外部
            # 尖角 + 5个内部凹陷点 + 1个闭合重复点)，再在相邻关键点之间做
            # 线性插值连成折线——本质上和 Rectangle 用的是同一种"关键点+分段
            # 线性插值"套路，只是关键点数量和几何(五角星 vs 矩形)不同。
            # --- 1. 定义几何参数 ---
            # 外部圆（五角星的五个尖角所在圆）的参数
            a = 0.2 * self.traj_scale  # 使用 a 来控制整体尺寸（外部半径）
            center_x, center_y, center_z, center_z_1 = 0.4, 0.0, 0.055, 0.2  # 统一中心点
            # 定义五角星的半径和中心点
            radius = a  # 外部半径（尖角到中心）
            # 根据 self.idx 确定 2D 形状的中心偏移
            if self.idx == 1:
                # Y-Z 平面：形状中心是 (center_y, center_z)
                center_prime_0 = center_y
                center_prime_1 = center_z_1
            else:
                # X-Y 平面：形状中心是 (center_x, center_y)
                center_prime_0 = center_x
                center_prime_1 = center_y
            # 内部凹陷点半径 eradius 由外部半径按正五角星的固定几何比例
            # (sin18°/sin54°) 换算得到，保证画出的是"正"五角星而不是随意的星形。
            eradius = radius * np.sin(np.pi / 10.0) / np.sin(3 * np.pi / 10.0)
            # --- 2. 计算 11 个关键点 (5个尖角 + 5个凹陷 + 1个闭合点) ---
            Star_points_2D = np.zeros((11, 2))
            for i in range(5):
                # 尖角 (Outer Points)
                theta_outer = (2 * np.pi / 5) * i + (np.pi / 2)
                Star_points_2D[2 * i, 0] = np.cos(theta_outer) * radius + center_prime_0
                Star_points_2D[2 * i, 1] = np.sin(theta_outer) * radius + center_prime_1

                # 凹陷点 (Inner Points)
                theta_inner = (2 * np.pi / 5) * i + (np.pi / 2) + (np.pi / 5)
                Star_points_2D[2 * i + 1, 0] = np.cos(theta_inner) * eradius + center_prime_0
                Star_points_2D[2 * i + 1, 1] = np.sin(theta_inner) * eradius + center_prime_1
            # 闭合轨迹：第 11 个点 = 第 1 个点
            Star_points_2D[-1, :] = Star_points_2D[0, :]
            # --- 3. 轨迹插值（使用与您代码类似的线性插值方法） ---
            # 计算总时间步数
            num_steps = len(t_param)  # 使用您的 t 或 self.time_vector
            # 假设轨迹分段均匀 (10个线段)
            num_segments = 10
            refs = np.zeros((num_steps, 2))
            # 计算每段轨迹包含的步数
            each_num = num_steps // num_segments
            current_step = 0
            for i in range(num_segments):
                start_point = Star_points_2D[i, :]
                end_point = Star_points_2D[i + 1, :]
                # 确保最后一段占满剩余所有步数
                num_interp_points = each_num if i < num_segments - 1 else num_steps - current_step
                for j in range(num_interp_points):
                    t_ = j / (num_interp_points - 1) if num_interp_points > 1 else 0.0

                    refs[current_step + j, :] = t_ * end_point + (1 - t_) * start_point
                current_step += num_interp_points
            # --- 4. 组装 3D 坐标 (Y-Z平面，X固定) ---
            if self.idx == 1:
                x = center_x * np.ones((num_steps, 1))
                y = refs[:, 0].reshape(-1, 1)  # Y 对应 2D 坐标的第一个分量 (x')
                z = refs[:, 1].reshape(-1, 1)  # Z 对应 2D 坐标的第二个分量 (y')
            else:
                x = refs[:, 0].reshape(-1, 1)
                y = refs[:, 1].reshape(-1, 1)
                z = center_z * np.ones((num_steps, 1))
            xyz_coords = np.concatenate((x, y, z), axis=1)

        else:
            raise ValueError(f"未知的轨迹名称: {traj_name}")

        print(f"笛卡尔轨迹生成完毕, 共 {len(xyz_coords)} 个点。")
        # 笛卡尔路径规划到此结束；下面这一步是"路径"到"控制指令"的桥梁——
        # 对轨迹上的每个点求逆运动学，得到机械臂应该摆出的关节角。
        joint_angles_trajectory = self._solve_ik_trajectory(xyz_coords, target_orientation, initial_qpos=initial_qpos)

        # 返回生成的所有数据，这些数据将用于后续的控制和可视化
        return xyz_coords, joint_angles_trajectory, self.time_vector

    def _solve_ik_trajectory(
        self,
        xyz_coords: np.ndarray,
        target_orientation,
        pinned_qpos: dict | None = None,
        initial_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        对一串笛卡尔坐标点逐点求解逆运动学，返回对应的关节角度轨迹。
        由 generate() 和 generate_multi_stroke() 共用。

        Args:
            pinned_qpos: {qpos索引: 固定值(弧度)}，可选。传了的话这些自由度会被锁死
                (配合调用方把对应关节名从 self.joint_names 里临时去掉，不交给 IK 求解)。
            initial_qpos: IK 第一次求解时的起始猜测 (默认全零)。dm_control 的
                qpos_from_site_pose 不感知关节限位, 从全零沿目标轨迹逐点走过去时可能
                收敛到一个数学上有解但超出物理限位的分支; 换一个离目标更近、更"自然"
                的起始姿态可以让它落到限位内的分支。排查方法: 逐点核对解出的角度是否落
                在 model.jnt_range 内 (dm_control 不会报错, 需要自己检查)。
        """
        # 整体做法: 沿轨迹逐点求 IK, 前一个点的解直接当作后一个点的初值 (热启动)。
        # 好处: 速度快, 且解在关节空间里连续 —— 不会在相邻两点之间突然跳到另一个"肘上/肘下"分支。
        print("开始将笛卡尔轨迹转换为关节角度 (使用 dm_control IK)...")
        joint_angles_trajectory = []

        # 重置物理状态到起始猜测姿态，作为第一次IK求解的起点
        with self.physics.reset_context():
            # 把 IK 的"起始猜测"设成 initial_qpos (默认全零)。
            # IK 只会沿着一个分支收敛, 初值不同, 可能落到数学上都成立但物理上不同的解
            # (例如超出 <joint range> 的那一支), 所以目标区域离零位很远时需要传一个更接近目标的 initial_qpos。
            self.physics.data.qpos[:] = (
                np.zeros(self.physics.model.nq) if initial_qpos is None else np.asarray(initial_qpos)
            )
            # 把需要锁定的自由度写成固定值, 例如 {4: -1.2} 表示 qpos[4](wrist_roll) 恒为 -1.2 rad。
            # 配合调用方把该关节名从 self.joint_names 里去掉: IK 不再调整它, 它的 qpos 就一直保持这个值。
            if pinned_qpos:
                for idx, val in pinned_qpos.items():
                    self.physics.data.qpos[idx] = val

        for i, pos in enumerate(xyz_coords):
            # 保存当前成功解，以备下次失败时使用
            # ---- 逐点求解循环 (xyz_coords 的每一行是一个目标点) ----
            # 先备份当前 qpos: 万一这个点 IK 失败, 就用这份备份把状态回滚到"上一个成功点", 避免坏姿态污染后面的求解。
            last_successful_qpos = self.physics.data.qpos.copy()

            # 对每个笛卡尔坐标点调用IK求解器
            # 对当前目标点 pos 解 IK; 成功返回 5 维关节角, 失败返回 None。
            q_sol = self._solve_ik(pos, target_orientation)

            if q_sol is not None:
                # 如果求解成功，将关节角度添加到轨迹列表中
                joint_angles_trajectory.append(q_sol)
                # 因为在 _solve_ik 中设置了 inplace=True，
                # self.physics.data.qpos 已经被更新为新的解，
                # 它将自动作为下一次求解的初始猜测值。
            else:
                # 如果求解失败
                print(f"警告: 逆运动学在时间步 {i} (目标位置: {np.round(pos, 3)}) 求解失败。")

                # 使用上一个成功的结果来填充，以保持轨迹的连续性
                if joint_angles_trajectory:
                    joint_angles_trajectory.append(joint_angles_trajectory[-1])
                    # 【重要】将物理状态重置回上一个成功点，避免从一个坏的姿态开始下一次求解
                    with self.physics.reset_context():
                        self.physics.data.qpos[:] = last_successful_qpos
                else:
                    # 如果连第一个点都失败了，说明目标点可能完全不可达，或者初始姿态太差
                    raise RuntimeError("轨迹的第一个点IK求解失败,请检查目标位置是否在机器人工作空间内。")

        print("关节角度轨迹转换完成。")
        # 输出形状 (N, num_joints), 单位弧度, N 与 xyz_coords 的点数相同 (一一对应)。
        # 注意: 这里没有检查关节限位、也没有检查相邻点的角度跳变; 写字场景下建议用
        # write_demo.py 的 check_reachability(), 必要时自己核对 model.jnt_range。
        return np.array(joint_angles_trajectory)

    def generate_multi_stroke(
        self,
        strokes,
        center=(0.35, 0.0),
        write_height=0.055,
        lift_height=0.12,
        points_per_stroke_segment=60,
        lift_transit_points=30,
        target_orientation=None,
        fixed_wrist_roll_rad: float | None = None,
    ):
        """
        生成"多笔画"轨迹：每一笔画贴桌面书写，笔画之间抬笔 -> 平移 -> 落笔。

        注意：每个点对应一个仿真控制步 (env.dt = 0.02s, 50Hz)，点数就是播放时长——
        `points_per_stroke_segment`/`lift_transit_points` 越大，点越密、播放越慢。
        比如一笔 16cm 长的直线段用 60 个点插值，播放时长是 60*0.02=1.2s，约 13cm/s；
        想再慢一点/更精细，直接调大这两个参数即可。

        Args:
            strokes: 笔画列表，每一笔画是一串 (x, y) 二维坐标 (单位: 米，相对于 center 的偏移量)。
                     例如 [[(-0.08, 0), (0.08, 0)], [(0, -0.08), (0, 0.08)]] 是两笔画的"十"字。
            center: 笔画坐标的世界系原点偏移 (cx, cy)。
            write_height: 落笔书写时的 Z 高度 (贴桌面)。
            lift_height: 抬笔平移时的 Z 高度 (需高于 write_height, 但仍在训练覆盖的关节范围内)。
            points_per_stroke_segment: 每一笔画内部相邻两点之间插值的点数。
            lift_transit_points: 抬笔/落笔/平移各阶段插值的点数。
            target_orientation: 传给 IK 的目标姿态四元数，默认为 None (不约束姿态，与其余轨迹保持一致)。
            fixed_wrist_roll_rad: 若不为 None，则整条轨迹的 wrist_roll 全程锁死在这个值 (弧度)，
                                   IK 只用另外 4 个关节去够 XYZ 位置 (运动学一致，不是事后改数值)。
                                   回归 Home 阶段不受此参数影响，仍然按 Home 关键帧的 wrist_roll 走。

        Returns:
            (xyz_coords, joint_angle_trajectory, pen_down_mask)
            pen_down_mask: 与 xyz_coords 等长的布尔数组，True 表示该点处于"落笔书写"阶段，
                           可用于后续可视化时区分书写轨迹和空中转移轨迹。
        """
        # center 是整个字在桌面(世界系)上的位置, 单位米: (cx, cy)。
        # strokes 里的坐标是"相对字中心的偏移", 所以最终点位 = center + 偏移。
        # 坐标对应: 字的横向 -> 世界 X, 字的纵向(向上为正) -> 世界 Y (没有翻转/旋转)。
        cx, cy = center
        # points         : 累积所有 3D 点 (x, y, z)。
        # pen_down_mask  : 与 points 等长, True = 落笔书写, False = 抬笔/平移/落笔过渡。
        # current_xy     : 上一笔的终点 (x, y)。第一笔之前为 None, 表示不需要抬笔转移。
        points = []
        pen_down_mask = []
        current_xy = None

        # 按笔顺逐笔处理。每一笔: (转移到本笔起点: 抬笔 -> 平移 -> 落笔) + (沿折线书写)。
        for stroke in strokes:
            # 把这一笔的所有点从"相对字中心"换算为世界系 (x, y)。
            stroke = [(cx + p[0], cy + p[1]) for p in stroke]
            # 本笔起点 (落笔位置)。
            start_xy = stroke[0]

            # 不是第一笔: 需要先离开上一笔终点, 转移到本笔起点。
            if current_xy is not None:
                # 阶段A 抬笔: 在上一笔终点正上方 (x, y 不变), z 从 write_height 线性升到 lift_height,
                # 共 lift_transit_points 个点。作用: 离开纸面, 避免平移时在桌面上拖出多余的线。
                # 抬笔 (write_height -> lift_height)
                for z in np.linspace(write_height, lift_height, lift_transit_points):
                    points.append((current_xy[0], current_xy[1], z))
                    pen_down_mask.append(False)
                # 阶段B 平移: z 保持 lift_height, (x, y) 从上一笔终点线性移动到本笔起点。
                # 点数同样是 lift_transit_points —— 平移距离越远, 末端速度越快。
                # 平移 (在 lift_height 高度从上一笔终点移动到下一笔起点)
                for t in np.linspace(0, 1, lift_transit_points):
                    x = current_xy[0] + t * (start_xy[0] - current_xy[0])
                    y = current_xy[1] + t * (start_xy[1] - current_xy[1])
                    points.append((x, y, lift_height))
                    pen_down_mask.append(False)
                # 阶段C 落笔: 在本笔起点正上方, z 从 lift_height 降回 write_height。这三段的 pen_down_mask 全为 False。
                # 落笔 (lift_height -> write_height)
                for z in np.linspace(lift_height, write_height, lift_transit_points):
                    points.append((start_xy[0], start_xy[1], z))
                    pen_down_mask.append(False)

            # 阶段D 书写: 笔画是一条折线 (关键点序列), 对每一段 (seg_start -> seg_end) 在 z=write_height 上
            # 线性插值 points_per_stroke_segment 个点, pen_down_mask 为 True。
            # 注意 np.linspace(0, 1, N) 同时包含两个端点, 因此相邻两段的接点会重复一次。
            # 沿笔画的各段插值 (贴桌面书写)
            for seg_start, seg_end in zip(stroke[:-1], stroke[1:]):
                for t in np.linspace(0, 1, points_per_stroke_segment):
                    x = seg_start[0] + t * (seg_end[0] - seg_start[0])
                    y = seg_start[1] + t * (seg_end[1] - seg_start[1])
                    points.append((x, y, write_height))
                    pen_down_mask.append(True)

            # 记录本笔终点, 作为下一笔"抬笔"的起点。
            current_xy = stroke[-1]

        # list of tuple -> (N, 3) 数组, 列为 x, y, z (米)。
        xyz_coords = np.array(points)
        # list of bool -> (N,) 布尔数组, 便于后面用 .sum() / ~mask 统计。
        pen_down_mask = np.array(pen_down_mask)
        print(f"多笔画轨迹生成完毕, 共 {len(strokes)} 笔, {len(xyz_coords)} 个点 (含 {(~pen_down_mask).sum()} 个抬笔/平移点)。")

        # ---- 笛卡尔点 -> 关节角 (逆运动学) ----
        # 两个分支:
        #   1) 指定 fixed_wrist_roll_rad: 锁死 wrist_roll, 只让前 4 个关节去够 XYZ 位置
        #      (5 个关节够 3 维位置本来有冗余, 少一个自由度仍然够用, 但解会更"受限")。
        #   2) 不指定: 5 个关节全部参与, wrist_roll 随点位自由变化 (可能在轨迹中大幅转动)。
        if fixed_wrist_roll_rad is not None:
            # wrist_roll 是 self.joint_names 第 5 个 (index 4)，临时从 IK 可解关节里去掉，
            # 只让另外 4 个关节去够 XYZ 位置，wrist_roll 全程锁死。
            # 先备份完整的关节名列表, 因为下面要临时修改 self.joint_names (它是实例属性, 改动会影响后续所有调用)。
            original_joint_names = self.joint_names
            # 只保留前 4 个 (pan, lift, elbow, wrist_flex), 即把第 5 个 wrist_roll 从"允许 IK 调整"的集合里拿掉。
            self.joint_names = original_joint_names[:4]
            # 用 try/finally 保证: 即使 IK 过程中抛异常, 也一定会把 joint_names 恢复, 不留下被篡改的状态。
            try:
                # pinned_qpos={4: fixed_wrist_roll_rad}: 在 IK 起始状态里把 qpos[4] 写成固定值;
                # 因为 wrist_roll 不在 joint_names 里, IK 迭代时它不会再变化 ->
                # 记录下来的每一帧 xyz 与关节角在运动学上是一致的 (而不是事后改数组导致位置对不上)。
                joint_angles_trajectory = self._solve_ik_trajectory(
                    xyz_coords, target_orientation, pinned_qpos={4: fixed_wrist_roll_rad}
                )
            # 恢复完整关节名列表。
            finally:
                self.joint_names = original_joint_names
        else:
            joint_angles_trajectory = self._solve_ik_trajectory(xyz_coords, target_orientation)

        # 用等间隔时间向量覆盖 self.time_vector (长度可能与构造函数里预设的不同)
        # 重新生成与点数匹配的时间向量 (覆盖构造函数里按 time_horizon*time_steps_per_sec 预设的版本)。
        # 这个时间向量只是为了接口一致; 真正的播放节奏由 Test.runFunc 的"每帧一个点, 50Hz"决定,
        # 所以点数 * 0.02s 才是实际播放时长。
        self.time_vector = np.linspace(0, self.time_horizon, len(xyz_coords))
        # 返回:
        #     xyz_coords             (N, 3)  笛卡尔参考点 (米)
        #     joint_angles_trajectory (N, 5) 对应关节角 (弧度)
        #     pen_down_mask          (N,)    落笔标记
        # (与 generate() 不同: generate() 第三个返回值是 time_vector, 这里是 pen_down_mask。)
        return xyz_coords, joint_angles_trajectory, pen_down_mask


class CartesianTrajectoryGenerator_pinocchio:
    """
    生成笛卡尔空间轨迹，并使用 Pinocchio+CasADi 进行逆运动学求解，
    最终输出关节角度轨迹。

    下面 generate() 里各轨迹形状(Fig8/Rectangle/Heart/FigStar/Helix/Lissajous)
    的参数方程和 CartesianTrajectoryGenerator 完全一致（同一套路径规划算法），
    唯一区别是逆运动学求解器换成了 Pinocchio+CasADi (见 _solve_ik/casadi_ik.py)，
    因此这里不再重复注释每种曲线的数学原理，详见上面
    CartesianTrajectoryGenerator.generate() 的注释。这条分支依赖 pinocchio.casadi
    绑定，默认不启用 (use_pinocchio=False)，见 README 的踩坑记录。
    """

    def __init__(
        self, arm_model_path: str, ee_site_name: str, num_joints: int, idx=1, time_horizon=60, time_steps_per_sec=5
    ):
        """
        初始化轨迹生成器及内置的IK求解器。

        Args:
            arm_model_path (str): 用于IK的机械臂模型路径 (e.g., "so101_new_calib.xml")。
            ee_site_name (str): 末端执行器在XML中的 <site> 名称。
            num_joints (int): 机械臂的关节数量。
            idx (int): 轨迹平面设置 (0 for x-y plane, 1 for y-z plane)。
            time_horizon (float): 轨迹的总时长（秒）。
            time_steps_per_sec (int): 每秒的轨迹点数量。
        """
        # 轨迹参数
        self.idx = idx
        self.time_horizon = time_horizon
        self.time_steps_per_sec = time_steps_per_sec
        self.total_steps = int(time_horizon * time_steps_per_sec)
        self.time_vector = np.linspace(0, self.time_horizon, self.total_steps)
        self.traj_scale = 0.5

        # IK参数
        self.arm_model_path = arm_model_path
        self.ee_site_name = ee_site_name
        self.num_joints = num_joints

        # 初始化IK求解器
        self._initialize_ik_solver()

    def _initialize_ik_solver(self):
        """加载模型并准备IK计算环境。"""
        print("正在为IK求解器初始化 Pinocchio+CasADi 模型...")
        try:
            from .casadi_ik import Kinematics

            self.ik_solver = Kinematics(self.ee_site_name)
            self.ik_solver.buildFromMJCF(self.arm_model_path)
            print("IK求解器初始化完成。")
        except Exception as e:
            print(f"错误：无法从'{self.arm_model_path}'初始化IK模型。 {e}")
            sys.exit(1)

    def _solve_ik(self, target_tf: np.ndarray) -> np.ndarray:
        """
        内部IK求解函数，使用 Pinocchio+CasADi。

        Args:
            target_tf (np.ndarray): 4x4的目标变换矩阵。

        Returns:
            np.ndarray: 求解出的关节角度（弧度），如果失败则返回 None。
        """
        q_sol, info = self.ik_solver.ik(target_tf)
        if info["success"]:
            return q_sol[: self.num_joints]
        else:
            return None

    def generate(self, traj_name="Fig8", target_orientation_matrix=np.eye(3)):
        """
        生成指定的笛卡尔轨迹并求解对应的关节角度轨迹。

        Args:
            traj_name (str): 轨迹名称 ('Fig8', 'Circle')。
            target_orientation_matrix (np.ndarray): 3x3的旋转矩阵，定义末端执行器姿态。

        Returns:
            tuple: 包含三个元素的元组:
                - np.ndarray: 形状为 (N, 3) 的笛卡尔坐标点云。
                - np.ndarray: 形状为 (N, num_joints) 的关节角度轨迹（弧度）。
                - np.ndarray: 对应的时间向量。
        """
        # 1. 生成笛卡尔坐标点 (x, y, z)
        t_param = 1.6 + 0.02 * np.linspace(0, self.time_horizon * 5, len(self.time_vector))
        print(f"正在生成 '{traj_name}' 笛卡尔轨迹...")

        if traj_name == "Fig8":
            if self.idx == 1:  # Y-Z平面
                a = 0.2 * self.traj_scale
                b = 0.2 * self.traj_scale
                x = 0.4 * np.ones((len(t_param), 1))
                z = np.expand_dims(0.2 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1)
                y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1)
            else:  # X-Y平面
                a = 0.25 * self.traj_scale
                b = 0.25 * self.traj_scale
                z = 0.055 * np.ones((len(t_param), 1))
                x = np.expand_dims(0.3 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1)
                y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param) ** 2), axis=1)
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Rectangle":
            # --- 1. 定义几何参数 ---
            width = 0.25 * self.traj_scale
            height = 0.25 * self.traj_scale

            # 定义中心点
            if self.idx == 1:  # Y-Z 平面
                cx, cy = 0.0, 0.2
                x_fixed = 0.4
            else:  # X-Y 平面
                cx, cy = 0.35, 0.0
                z_fixed = 0.055
            # --- 2. 定义 4 个顶点 (从左下角开始逆时针) ---
            # 顺序: 左下 -> 右下 -> 右上 -> 左上 -> 左下 (闭合)
            rect_points = np.array(
                [
                    [cx - width / 2, cy - height / 2],
                    [cx + width / 2, cy - height / 2],
                    [cx + width / 2, cy + height / 2],
                    [cx - width / 2, cy + height / 2],
                    [cx - width / 2, cy - height / 2],
                ]
            )
            # --- 3. 线性插值 ---
            num_steps = len(t_param)
            num_segments = 4  # 矩形有4条边
            refs = np.zeros((num_steps, 2))
            each_num = num_steps // num_segments
            current_step = 0

            for i in range(num_segments):
                start_pt = rect_points[i, :]
                end_pt = rect_points[i + 1, :]

                # 确保最后一段填满剩余步数
                n_points = each_num if i < num_segments - 1 else num_steps - current_step

                # 生成线性插值因子 (0 到 1)
                t_ = np.linspace(0, 1, n_points).reshape(-1, 1)
                segment_traj = (1 - t_) * start_pt + t_ * end_pt

                refs[current_step : current_step + n_points, :] = segment_traj
                current_step += n_points

            # --- 4. 组装坐标 ---
            if self.idx == 1:  # Y-Z 平面
                x = x_fixed * np.ones((num_steps, 1))
                y = refs[:, 0].reshape(-1, 1)
                z = refs[:, 1].reshape(-1, 1)
            else:  # X-Y 平面
                x = refs[:, 0].reshape(-1, 1)
                y = refs[:, 1].reshape(-1, 1)
                z = z_fixed * np.ones((num_steps, 1))
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Heart":
            # 缩放系数 (心形方程生成的值较大，需要缩小)
            scale = 0.01 * self.traj_scale
            # 调整参数 t 使其覆盖 0 到 2pi (根据你的 t_param 范围可能需要调整)
            # 假设 t_param 是线性增加的，我们取模或归一化来画完整的圆
            t_circle = np.linspace(0, 2 * np.pi, len(t_param))
            # 心形参数方程
            # shape_x 对应水平宽，shape_y 对应垂直高（尖端在下，凹陷在上）
            shape_x = 16 * np.sin(t_circle) ** 3
            shape_y = 13 * np.cos(t_circle) - 5 * np.cos(2 * t_circle) - 2 * np.cos(3 * t_circle) - np.cos(4 * t_circle)
            shape_x = shape_x * scale
            shape_y = shape_y * scale

            if self.idx == 1:  # Y-Z平面
                center_y, center_z = 0.0, 0.2
                x = 0.4 * np.ones((len(t_param), 1))
                # 注意：心形方程 y 轴对应竖直方向，所以映射到 Z，x 轴映射到 Y
                y = np.expand_dims(center_y + shape_x, axis=1)
                z = np.expand_dims(center_z + shape_y, axis=1)
            else:  # X-Y平面
                center_x, center_y = 0.35, 0.0
                z = 0.055 * np.ones((len(t_param), 1))
                # 旋转90度让心形正对
                x = np.expand_dims(center_x + shape_y, axis=1)  # 竖直方向映射到 X
                y = np.expand_dims(center_y + shape_x, axis=1)  # 水平方向映射到 Y
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "FigStar":
            # --- 1. 定义几何参数 ---
            # 外部圆（五角星的五个尖角所在圆）的参数
            a = 0.2 * self.traj_scale  # 使用 a 来控制整体尺寸（外部半径）
            center_x, center_y, center_z, center_z_1 = 0.4, 0.0, 0.055, 0.2  # 统一中心点
            # 定义五角星的半径和中心点
            radius = a  # 外部半径（尖角到中心）
            # 根据 self.idx 确定 2D 形状的中心偏移
            if self.idx == 1:
                # Y-Z 平面：形状中心是 (center_y, center_z)
                center_prime_0 = center_y
                center_prime_1 = center_z_1
            else:
                # X-Y 平面：形状中心是 (center_x, center_y)
                center_prime_0 = center_x
                center_prime_1 = center_y
            eradius = radius * np.sin(np.pi / 10.0) / np.sin(3 * np.pi / 10.0)
            # --- 2. 计算 11 个关键点 (5个尖角 + 5个凹陷 + 1个闭合点) ---
            Star_points_2D = np.zeros((11, 2))
            for i in range(5):
                # 尖角 (Outer Points)
                theta_outer = (2 * np.pi / 5) * i + (np.pi / 2)
                Star_points_2D[2 * i, 0] = np.cos(theta_outer) * radius + center_prime_0
                Star_points_2D[2 * i, 1] = np.sin(theta_outer) * radius + center_prime_1

                # 凹陷点 (Inner Points)
                theta_inner = (2 * np.pi / 5) * i + (np.pi / 2) + (np.pi / 5)
                Star_points_2D[2 * i + 1, 0] = np.cos(theta_inner) * eradius + center_prime_0
                Star_points_2D[2 * i + 1, 1] = np.sin(theta_inner) * eradius + center_prime_1
            # 闭合轨迹：第 11 个点 = 第 1 个点
            Star_points_2D[-1, :] = Star_points_2D[0, :]
            # --- 3. 轨迹插值（使用与您代码类似的线性插值方法） ---
            # 计算总时间步数
            num_steps = len(t_param)  # 使用您的 t 或 self.time_vector
            # 假设轨迹分段均匀 (10个线段)
            num_segments = 10
            refs = np.zeros((num_steps, 2))
            # 计算每段轨迹包含的步数
            each_num = num_steps // num_segments
            current_step = 0
            for i in range(num_segments):
                start_point = Star_points_2D[i, :]
                end_point = Star_points_2D[i + 1, :]
                # 确保最后一段占满剩余所有步数
                num_interp_points = each_num if i < num_segments - 1 else num_steps - current_step
                for j in range(num_interp_points):
                    t_ = j / (num_interp_points - 1) if num_interp_points > 1 else 0.0

                    refs[current_step + j, :] = t_ * end_point + (1 - t_) * start_point
                current_step += num_interp_points
            # --- 4. 组装 3D 坐标 (Y-Z平面，X固定) ---
            if self.idx == 1:
                x = center_x * np.ones((num_steps, 1))
                y = refs[:, 0].reshape(-1, 1)  # Y 对应 2D 坐标的第一个分量 (x')
                z = refs[:, 1].reshape(-1, 1)  # Z 对应 2D 坐标的第二个分量 (y')
            else:
                x = refs[:, 0].reshape(-1, 1)
                y = refs[:, 1].reshape(-1, 1)
                z = center_z * np.ones((num_steps, 1))
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Helix":
            radius = 0.1 * self.traj_scale
            # 螺旋的长度
            length = 0.2
            # 生成圆周运动
            # 这里的 t_param 直接用于周期，如果 t_param 范围大，螺旋圈数就多
            circle_1 = radius * np.cos(t_param * 2)
            circle_2 = radius * np.sin(t_param * 2)
            # 生成轴向推进 (往返运动，使用 sin 避免跳变，或者线性)
            # 这里使用线性往返：从 0 到 length 再回来
            # 为了简单，这里演示单向推进然后瞬移，或者用 np.linspace
            linear_move = np.linspace(-length / 2, length / 2, len(t_param))
            center_x, center_y = 0.35, 0.0
            center_z = 0.15
            x = np.expand_dims(center_x + circle_1, axis=1)
            y = np.expand_dims(center_y + circle_2, axis=1)
            z = np.expand_dims(center_z + linear_move, axis=1)  # 高度变化
            xyz_coords = np.concatenate((x, y, z), axis=1)

        elif traj_name == "Lissajous":
            A = 0.12 * self.traj_scale
            B = 0.12 * self.traj_scale
            # 频率参数 (3:2 的比例会生成一个经典的纽结形状)
            a_freq = 3.0
            b_freq = 2.0
            delta = np.pi / 2  # 相位差

            # 归一化时间参数以保证闭环
            t_cycle = np.linspace(0, 2 * np.pi, len(t_param))

            val_1 = A * np.sin(a_freq * t_cycle + delta)
            val_2 = B * np.sin(b_freq * t_cycle)

            if self.idx == 1:  # Y-Z平面
                center_y, center_z = 0.0, 0.2
                x = 0.4 * np.ones((len(t_param), 1))
                y = np.expand_dims(center_y + val_1, axis=1)
                z = np.expand_dims(center_z + val_2, axis=1)
            else:  # X-Y平面
                center_x, center_y = 0.35, 0.0
                z = 0.055 * np.ones((len(t_param), 1))
                x = np.expand_dims(center_x + val_2, axis=1)
                y = np.expand_dims(center_y + val_1, axis=1)
            xyz_coords = np.concatenate((x, y, z), axis=1)

        else:
            raise ValueError(f"未知的轨迹名称: {traj_name}")

        # 2. 求解逆运动学
        print("开始将笛卡尔轨迹转换为关节角度 (使用 Pinocchio+CasADi IK)...")
        joint_angles_trajectory = []
        target_tf = np.eye(4)
        target_tf[:3, :3] = target_orientation_matrix

        for i, pos in enumerate(xyz_coords):
            # 更新目标变换矩阵的位置部分
            target_tf[:3, 3] = pos

            # 调用内部IK求解器
            q_sol = self._solve_ik(target_tf)

            if q_sol is not None:
                joint_angles_trajectory.append(q_sol)
            else:
                print(f"警告: 逆运动学在时间步 {i} (目标位置: {np.round(pos, 3)}) 求解失败。")
                if joint_angles_trajectory:
                    # 使用上一个成功的结果来填充，保持轨迹连续性
                    joint_angles_trajectory.append(joint_angles_trajectory[-1])
                else:
                    raise RuntimeError("轨迹的第一个点IK求解失败, 请检查目标位置和姿态。")

        print("关节角度轨迹转换完成。")
        return xyz_coords, np.array(joint_angles_trajectory), self.time_vector