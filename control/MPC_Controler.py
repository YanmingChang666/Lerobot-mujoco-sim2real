"""
基于训练好的 Koopman 模型做实时 MPC(模型预测控制) 的核心模块。

整体思路：train.py 训出的 Koopman 网络提供了一组（近似）线性/双线性的
动力学矩阵 Ad(K)/Bd(B)/Hd(H)/C，本文件把这些矩阵喂给 CasADi，在观测空间
里构建一个"未来 H 步"的开环最优控制问题：每一步都要让预测状态尽量贴近
参考轨迹、同时控制量本身不要太大/变化太剧烈，用 IPOPT 求解出最优控制
序列后，只执行第一步(经典的滚动时域/receding horizon 做法)，下一时刻
重新观测状态、重新求解，如此循环——这就是"MPC"名字的由来。

三个类的关系：
- MPCController：最基础版本，只做"Koopman模型 + MPC"，不做任何状态估计/
  扰动补偿，假设观测到的状态就是真实状态。
- MPCController_KESO：在 MPCController 基础上叠加 卡尔曼滤波(update_kf)
  和扩张状态观测器(update_eso)，对应 README 里 "DBKN + KESO" 抗负载/抗
  未建模扰动实验（KESO = Kalman + ESO）。
- MPCController_UKF：在 MPCController 基础上叠加 无迹卡尔曼滤波(UKF)，
  用 sigma 点传播代替线性卡尔曼滤波的一阶线性化，能更好地处理 Koopman
  双线性项带来的非线性，对应 README 里 "IBKN + UKF" 抗观测噪声实验。

变量名对照：z 是 Koopman 观测空间状态(维度 Nkoopman)，x 是原始物理状态
(位置/关节角)，u 是控制输入(关节目标速度)，Ad/Bd/Hd 分别是训练好的线性项
矩阵 K、控制矩阵 B、双线性项矩阵 H，C 是（固定或训练出的）状态解码矩阵。
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import torch


class MPCController:
    """最基础的 Koopman MPC 控制器：只做开环滚动优化，不含状态估计/扰动补偿。"""

    def __init__(self, net, args):
        self.in_dim = args.x_dim
        self.u_dim = args.u_dim
        self.net = net
        self.device = args.device
        self.args = args
        #  ==== 系统参数 ====
        self.obs_num = args.x_dim  #  可观测量 3位置，8位置与角度
        self.x_dim = args.x_dim  #  可观测量 3位置，8位置与角度
        # 从训练好的 PyTorch 网络里把 Koopman 线性算子的权重"抠出来"，转成
        # numpy 常量矩阵——MPC 用 CasADi 求解，不需要（也不能）反向传播，
        # 训练好的参数在这里就固定下来了，后续所有预测都是纯数值计算。
        if hasattr(net, "lA"):
            self.Ad = net.lA.weight.cpu().detach().numpy()  # (32, 32)
        else:
            # use_stable=True 时模型没有 lA, 用 SVD 分解 (U_raw/V_raw/S_raw) 动态构建的
            # get_koopman_matrix_K() 才是实际生效的转移矩阵, 权重已经训练/加载完毕,
            # 这里只是把它取出来当成一个固定数值矩阵交给 CasADi 用。
            self.Ad = net.get_koopman_matrix_K().cpu().detach().numpy()
        self.Bd = net.lB.weight.cpu().detach().numpy()  # (32, 5)
        self.Nkoopman = self.Ad.shape[0]
        if hasattr(net, "H"):  # 检查是否存在该层
            # 只有双线性网络(DBKN/IBKN)才有 H；线性网络(DKUC/IKN)没有这个属性，
            # 后面所有涉及 H_hat_list/Hd 的双线性项计算都会被自然跳过。
            self.H_hat_list = net.get_Hi_numpy()
            self.Hd = net.H.weight.cpu().detach().numpy()  # (32, 224)
        if hasattr(net, "lC"):
            self.C = net.lC.weight.cpu().detach().numpy()  # (8, 32)
        # B 矩阵的伪逆：扩张状态观测器(ESO)要把估计出的"观测空间里的扰动"
        # d_hat 转换成一个等效的控制量补偿 u_eso = -B^+ @ d_hat，伪逆是
        # 这个转换在 B 不是方阵时的标准做法（最小二乘意义下的"最接近"解）。
        self.B_d_pinv = np.linalg.pinv(self.Bd)

        # ==== MPC 参数 ====
        # Q 是状态误差的权重矩阵：前 3 维(位置)权重 50 远大于后 5 维(关节角)权重 1，
        # 体现"末端笛卡尔位置跟踪准"比"关节角度分毫不差"更重要的控制目标；
        # R 是控制量本身的惩罚，避免优化器为了跟踪精度而给出过大/剧烈的控制量。
        # 原来写的是硬编码的 8 元素 np.diag([50,50,50,1,1,1,1,1])（3位置+5关节角，
        # 只对 SO-101 的 x_dim=8 成立）。这个分支之前因为 state_full 恒为 True 的
        # bug 从未被执行过；现在修好那个 bug 后它会被换成别的机械臂(如 x_dim=10)
        # 实际用到，所以改成按 self.x_dim 动态生成，位置 3 维权重 50，其余关节角
        # 权重 1，含义不变。
        Q = np.diag([50, 50, 50] + [1] * (self.x_dim - 3))
        R = 0.5 * np.eye(self.u_dim)  # 控制输入的惩罚矩阵
        self.H = 10  # 预测步长
        self.Q = ca.DM(Q)  # 状态误差权重
        self.R = ca.DM(R)  # 控制输入权重
        self.u_eso = np.zeros(self.u_dim)
        self.startid = 0
        if hasattr(net, "seq_len"):  # 用于时序网络
            self.startid = net.seq_len - 1

        # 构建MPC求解器
        self.u_prev = np.zeros(self.u_dim)
        self.MPC_type = args.MPC_type
        self.state_full = False
        # 可逆网络(IBKN/IKN)的解码器是精确可逆的 x_encode_net.inverse()，
        # 直接在 Nkoopman 维的完整观测空间里定义参考轨迹/误差更自然；
        # 而非可逆网络在 use_decoder=False 时，只有前 x_dim 维观测量有明确物理
        # 含义，所以只在这几维上定义误差 (state_full=False 分支)。
        # 注意：`args.model == "IBKN" or "IKN"` 是一个恒真的写法(非空字符串
        # 在布尔上下文里永远是 True)，实际效果等价于只要 use_decoder=True 就够,
        # 后半个 or 分支形同虚设，如果要精确判断模型类型应写成
        # `args.model in ("IBKN", "IKN")`。
        if args.use_decoder or args.model in ("IBKN", "IKN"):
            # 原来写的是 `args.model == "IBKN" or "IKN"`——"IKN" 是非空字符串永远为真,
            # 导致这个 or 表达式对所有模型都恒为 True, 不管 model/use_decoder 是什么。
            # 结果是 DBKN/DKUC 这些没有可学习解码器的模型也被强行按"整个 lifted 空间"
            # 去算 MPC 代价 (setup_mpc/setup_delta_mpc 里 state_full 分支), 而不是原本
            # 该走的、只对末端位置加权的简单版本 (Q=diag([50,50,50,1,1,1,1,1]))。
            self.state_full = True
        if self.state_full:
            self.Q = ca.DM(50 * np.eye(self.Nkoopman))  # 状态误差权重
            self.R = ca.DM(0.5 * np.eye(self.u_dim))  # 控制输入权重
        if args.MPC_type == "mpc":
            self.solver = self.setup_mpc()
        if args.MPC_type == "delta_mpc":
            self.solver = self.setup_delta_mpc()

    def linearize_B(self, z0):
        """
        线性化双线性项：根据 z0 固定所有时刻的 B_total
        即 B_approx = Bd + sum_j z0[j] * H_hat_j

        Args:
            z0: 初始 Koopman 状态，形状为 (Nkoopman,)

        Returns:
            B_total: Koopman 控制输入矩阵（线性化版本），类型为 ca.SX

        说明：双线性 Koopman 的真实演化含有 z_t 和 u_t 的交叉项，如果在整个
        预测时域 H 步内都精确展开这个交叉项，优化问题会变成非凸的双线性
        规划，求解代价高、也不稳定。这里用"只在当前观测到的 z0 处做一次
        线性化"的近似：把交叉项里的 z 固定成 z0（即假设未来 H 步状态变化
        不大，用当前值近似代替），交叉项就退化成关于 u 的线性项，
        整个 MPC 问题就变成了标准的凸二次规划，可以用 IPOPT 快速可靠地求解。
        这是 NMPC(非线性MPC)常见的"逐次线性化"简化手段之一。
        """
        B_total = ca.SX(self.Bd)  # 基础项 B
        # 双线性项计算：Σ_j (z_{k,j} * Ĥ_j) u_k
        if hasattr(self, "H_hat_list"):
            for j in range(self.Nkoopman):
                H_j = ca.SX(self.H_hat_list[j])
                B_total += z0[j] * H_j
        return B_total

    def setup_mpc(self):
        """
        构建CasADi MPC求解器（标准形式：直接优化 H 步的绝对控制量序列 u）。

        用 CasADi 符号变量搭出"预测 H 步 Koopman 状态演化 + 累加跟踪误差与
        控制代价"的目标函数，编译成一个参数化的非线性规划(NLP)：参数 p
        包含参考轨迹 ref 和当前观测状态 z0，决策变量 x 是整个预测时域的
        控制序列 u；每次调用 get_control() 只需要传入新的 (ref, z0) 作为
        p，不需要重新构建 NLP，这是 CasADi 这种"先编译后反复求解"用法的
        效率关键——如果每步都重新搭建符号图会非常慢。
        """
        u = ca.SX.sym("u", self.H * self.u_dim)
        z0 = ca.SX.sym("z0", self.Nkoopman, 1)  # 初始状态
        B_total = self.linearize_B(z0)  # 使用 z0 构造一次线性近似 B 矩阵
        z = z0
        cost = 0
        if not self.state_full:
            ref = ca.SX.sym("ref", self.H, self.x_dim)  # 参考轨迹（位置3 + 关节角7）
            for t in range(self.H):
                u_t = u[t * self.u_dim : (t + 1) * self.u_dim]
                z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)  # 加入扰动补偿
                ee_pos_pred = z_next[: self.x_dim]
                cost += ca.mtimes([(ee_pos_pred - ref[t, :].T).T, self.Q, (ee_pos_pred - ref[t, :].T)]) + ca.mtimes(
                    [u_t.T, self.R, u_t]
                )
                z = z_next
        else:
            ref = ca.SX.sym("ref", self.H, self.Nkoopman)  # 参考轨迹在Koopman空间
            for t in range(self.H):
                u_t = u[t * self.u_dim : (t + 1) * self.u_dim]
                z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)  # 加入扰动补偿
                cost += ca.mtimes([(z_next - ref[t, :].T).T, self.Q, (z_next - ref[t, :].T)]) + ca.mtimes(
                    [u_t.T, self.R, u_t]
                )
                z = z_next
        nlp = {
            "x": u,
            "f": cost,
            "p": ca.vertcat(ca.reshape(ca.transpose(ref), -1, 1), z0),  # 加入扰动补偿
        }
        opts = {
            "ipopt.print_level": 0,  # 0=无输出，5=详细
            "print_time": False,
            "ipopt.sb": "yes",  # 禁止初始标语
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        return solver

    def setup_delta_mpc(self):
        """
        构建 delta-MPC 求解器：决策变量改成每步控制量的增量 delta_u，
        真正的控制量由 u_t = u_{t-1} + delta_u_t 累加得到（u_prev 是上一次
        MPC 求解后真正执行的控制量，作为这一轮的起点）。相比 setup_mpc
        直接优化绝对控制量，优化增量能让相邻时刻的控制量变化天然更平滑，
        减少真机执行时的抖动，是 args.MPC_type 默认选 "delta_mpc" 的原因。
        """
        delta_u = ca.SX.sym("delta_u", self.H * self.u_dim)
        z0 = ca.SX.sym("z0", self.Nkoopman, 1)
        u_prev = ca.SX.sym("u_prev", self.u_dim)
        B_total = self.linearize_B(z0)

        if not self.state_full:
            ref = ca.SX.sym("ref", self.H, self.x_dim)
        else:
            ref = ca.SX.sym("ref", self.H, self.Nkoopman)

        cost = 0
        z = z0
        u_t = u_prev

        for t in range(self.H):
            delta_u_t = delta_u[t * self.u_dim : (t + 1) * self.u_dim]
            u_t = u_t + delta_u_t
            z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)

            if not self.state_full:
                ee_pos_pred = z_next[: self.x_dim]
                cost += ca.mtimes([(ee_pos_pred - ref[t, :].T).T, self.Q, (ee_pos_pred - ref[t, :].T)])
            else:
                cost += ca.mtimes([(z_next - ref[t, :].T).T, self.Q, (z_next - ref[t, :].T)])

            # 惩罚的是增量 delta_u_t 而不是绝对量 u_t，这才是"抑制控制量剧烈
            # 变化"这个目标真正对应的代价项。
            cost += ca.mtimes([delta_u_t.T, self.R, delta_u_t])
            z = z_next

        nlp = {"x": delta_u, "f": cost, "p": ca.vertcat(ca.reshape(ca.transpose(ref), -1, 1), z0, u_prev)}

        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        return solver

    def get_control(self, p):
        """
        求解一次 MPC，只取预测时域里第一步的最优控制量执行（滚动时域控制的
        核心做法），并叠加扩张状态观测器给出的扰动补偿 u_eso（如果没有
        ESO，u_eso 恒为 0，不影响基础版 MPCController 的行为）。
        """
        # 求解MPC
        sol = self.solver(x0=np.zeros(self.H * self.u_dim), p=p)
        u_opt = np.array(sol["x"]).reshape(self.H, self.u_dim)  # (10,7)
        # delta_mpc 模式下 u_opt[0] 是"增量"，要再加上 u_prev 才是真正要执行
        # 的绝对控制量；mpc 模式下 u_prev 恒为 0（不参与累加），u_opt[0]
        # 本身就是绝对控制量，两种模式用同一行代码兼容。
        u0 = u_opt[0] + self.u_eso + self.u_prev  # (7,) + (1, 7)
        # 记录控制量并返回
        # 执行前做物理限幅：即使优化结果数值上更极端，也不会送出超出电机
        # 安全范围([-2, 2])的指令。
        a = np.clip(u0, -2.0, 2.0)
        if self.MPC_type == "delta_mpc":
            self.u_prev = a.copy()
        return u0, a

    def Psi_o(self, s):  # 只对状态 s 进行升维 Ψ(s(t_k))
        """Measurement mapping - 适配不同环境的状态维度"""
        # 确保输入是tensor

        # 进行编码：把原始物理状态 s 通过训练好的 encoder 提升到 Koopman
        # 观测空间，得到的 z0 会缓存在 self.z0，供下一轮 MPC 求解、以及
        # KF/ESO/UKF 的状态估计使用。
        ds = self.net.x_encoder(s.to(self.device)).detach().cpu().numpy()
        psi = np.zeros([self.Nkoopman, 1])
        if len(ds.shape) > 1 and ds.shape[0] == 1:
            psi[: self.Nkoopman, 0] = ds[0]
        else:
            psi[: self.Nkoopman, 0] = ds
        self.z0 = psi
        return psi

    def koopman_predict(self, x, u):
        """直接调用训练好的网络做一步 Koopman 演化预测（推理用途，不经过 CasADi）。"""
        x_t = torch.DoubleTensor(x.T).to(self.device)
        u_t = torch.DoubleTensor(u).to(self.device)
        x_next = self.net.koopman_operation(x_t, u_t)
        return x_next


class MPCController_KESO(MPCController):
    """
    DBKN + KESO：在基础 MPC 之上叠加 卡尔曼滤波(抗观测噪声) 和
    扩张状态观测器 ESO(抗未建模扰动/负载)，对应 README 的"抗噪声"/"抗负载"
    实验。KESO = Kalman Filter + Extended State Observer 的组合缩写。
    """

    def __init__(self, net, args):
        """
        继承 MPC_KF 的初始化，包含 Koopman 模型、C矩阵、MPC solver、KF/UKF噪声协方差等。
        """
        super().__init__(net, args)  # 调用父类初始化
        # ==== ESO参数 ====
        self.d_hat = np.zeros(self.Nkoopman)  # 扰动估计值（观测空间里的等效外部扰动）
        # 自适应增益系数：误差越大时降低状态修正增益(gamma1)、加大扰动补偿
        # 增益(gamma2)，避免观测噪声/瞬时大误差被直接当成模型状态的剧烈跳变，
        # 而是更多地"归因"到未建模扰动上，具体用法见 update_eso。
        self.gamma1 = 1.2  # 状态增益上限 0.2  0.5
        self.gamma2 = 0.5  # 扰动增益上限 0.01 0.1
        # KF参数
        self.W = np.eye(self.Nkoopman) * 0.001  #  (37,37) 过程噪声协方差 0.0001  0.1
        # 如果系统本身非常不确定，增加 Q 的值；如果你对模型有很高的信心，可以减小 Q 的值。
        self.V = np.eye(self.obs_num) * 0.1  #  (3,3)  观测噪声协方差 0.01  10
        # 如果你知道传感器的噪声较大，可以增加 R 的值；如果传感器非常精确，则可以减小 R 的值。
        self.P = np.eye(self.Nkoopman) * 0.1  # (37,37)  初始协方差

    def predict(self, z, u):
        """标准卡尔曼滤波的预测步：用 Koopman 线性/双线性动力学向前推一步状态和协方差。"""
        # 预测步骤
        self.z_ = np.dot(self.Ad, z) + np.dot(self.Bd, u)  # (37,37)*(37,1)+(37,7)*(7,1)
        # 添加双线性项（若存在）
        if hasattr(self.net, "H"):
            kron = np.kron(u, z) if self.net.u_z else np.kron(z, u)
            self.z_ += np.dot(self.Hd, kron)
        self.P = np.dot(np.dot(self.Ad, self.P), self.Ad.T) + self.W  # 协方差更新
        return self.z_

    def update_kf(self, x, z, u):
        """标准（线性）卡尔曼滤波的更新步：用观测 x 修正预测状态 z_。"""
        # 更新步骤
        self.predict(z, u)
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.C, self.P), self.C.T) + self.V  # 创新协方差 () (3,37)(37,37)(37,3)=(3,3)
        K = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))  # 卡尔曼增益(37,37)(37,3)(3,3)=(37,3)
        # 更新状态
        x = x - self.C @ self.z_  #  (3,1) - (3,37) (37,1)
        self.z_ = self.z_ + K @ x  #  (37,1) + (37,3)*(3,1)
        self.P = (np.eye(self.Nkoopman) - K @ self.C) @ self.P
        return self.z_  # 返回滤波后的位置估计

    def update_eso(self, z_pred, y_true):
        """
        更新扩张状态观测器（ESO）。

        核心思路：把"模型预测不准"这件事本身当作一个待估计的状态
        (d_hat)，用观测误差(error)同时修正状态估计(z_hat)和扰动估计
        (d_hat)，再把 d_hat 通过 B 矩阵的伪逆转换成一个等效的控制量补偿
        u_eso，叠加到 MPC 算出的控制量里去主动抵消这部分未建模扰动
        （比如负载、摩擦、Koopman 线性化本身带来的模型误差）。
        """
        # 预测步骤
        # z_pred = self.Ad @ self.z0 + self.Bd @ self.u_prev + self.d_hat
        y_pred = self.C @ z_pred  # (3,37)*(37,)=(3,)
        # 校正步骤
        error = y_true[: self.obs_num] - y_pred  # (3,)
        # 2. 计算误差范数
        e_norm = np.linalg.norm(error)

        # 自适应增益（可选 diag 或整体比例）
        adapt_factor1 = self.gamma1 / (1.0 + e_norm)  # 越大误差，越小增益
        adapt_factor2 = self.gamma2 * e_norm / (1.0 + e_norm)  # 越大误差，越大扰动补偿
        # 构造自适应增益
        self.L1 = adapt_factor1 * np.eye(self.Nkoopman, self.obs_num)
        self.L2 = adapt_factor2 * np.eye(self.Nkoopman, self.obs_num)

        self.z_hat = z_pred + self.L1 @ error  #  (37,) + (37,3)*(3,) = (37,)
        self.d_hat = self.d_hat + self.L2 @ error  # (37,) + (37,3)*(3,) = = (37,)
        u_eso = -self.B_d_pinv @ self.d_hat  # (7,37) * (37,) = (1,7) numpy.matrix
        self.u_eso = np.array(u_eso).reshape(-1)

        # print(e_norm , adapt_factor1, adapt_factor2, self.u_eso)
        return self.z_hat.reshape(1, -1)

    def get_updated_state_KF(self, state_pre, z_last, a):
        """跑一次 KF 更新，把滤波后的 Koopman 状态解码回原始物理状态（供闭环仿真下一步使用）。"""
        # (1,32)
        state_full = self.update_kf(state_pre[: self.obs_num].reshape(-1, 1), z_last, a.reshape(-1, 1)).T
        state = self.net.x_decoder(torch.DoubleTensor(state_full).to(self.device)).detach().cpu().numpy().reshape(-1)
        return state[3:], state_full.T

    def get_updated_state_ESO(self, state, ref_state):
        """跑一次 ESO 更新，把估计后的 Koopman 状态解码回原始物理状态。"""
        state_full = self.update_eso(np.array(state).reshape(-1), ref_state)
        state = self.net.x_decoder(torch.DoubleTensor(state_full).to(self.device)).detach().cpu().numpy().reshape(-1)
        return state[3:]

    def verify_eso_stability(self, u_min=-0.5, u_max=0.5, n_samples=5000):
        """
        数值验证 DBKMPC-KESO 框架中 ESO 误差动力学的稳定性。
        原理:
            构造闭环误差矩阵 Phi(u) = [[A + H(u) - L1*C,  I],
                                    [-L2*C,            I]]
            验证其谱半径 rho(Phi) 是否恒小于 1。
        Args:
            agent: 您的 DBKMPC 类实例 (包含 Ad, Bd, Hd, C, L1, L2 等属性)
            u_min: 控制输入下界 (标量或数组)
            u_max: 控制输入上界 (标量或数组)
            n_samples: 随机采样的点数

        为什么需要这个验证：DBKN 的双线性项 H(u) 会让"误差动力学矩阵"随
        控制输入 u 变化而变化(不是一个固定矩阵)，理论上无法像线性系统那样
        一次性证明全局稳定；这里改用蒙特卡洛的方式，在控制量的可行范围内
        随机采样大量 u，逐个计算对应误差矩阵的谱半径，用"采样范围内全部
        谱半径 < 1"来给出稳定性的经验证据（不是严格的数学证明，但能在
        实际使用的控制量范围内提供较强的安全信心）。
        """

        # 1. 提取系统矩阵
        try:
            A = self.Ad
            C = self.C
            L1 = 1.1 * np.eye(self.Nkoopman, self.obs_num)
            L2 = 0.5 * np.eye(self.Nkoopman, self.obs_num)
            # 提取维度
            n_z = self.Nkoopman  # 32
            n_u = self.u_dim  # 7 (假设)
            # 处理双线性矩阵 Hd
            # Hd 通常存储为 (32, 32*7) 的扁平矩阵
            if hasattr(self, "Hd"):
                Hd_flat = self.Hd
            else:
                H_tensor = np.zeros((n_z, n_z, n_u))
        except AttributeError as e:
            print(f"Error: 缺少必要的属性 {e}。请确保传入了正确的 agent 对象。")
            return
        # 2. 预处理双线性张量 (Reshape)
        # 假设 Kronecker 积顺序为 z \otimes u (即 z 的每一项乘以整个 u 向量)
        # 这意味着 Hd 的列索引 k = i_z * n_u + i_u
        # 我们将其 reshape 为 (n_z, n_z, n_u) 以便快速计算 H(u)
        # H_tensor[row, col, input_channel]
        try:
            H_tensor = Hd_flat.reshape(n_z, n_z, n_u)
        except ValueError:
            print(f"Error: Hd 维度 {Hd_flat.shape} 无法 reshape 为 ({n_z}, {n_z}, {n_u})。请检查 u_dim 定义。")
            return

        # 3. 随机采样控制输入
        if np.isscalar(u_min):
            U_samples = np.random.uniform(u_min, u_max, (n_samples, n_u))
        else:
            # 如果上下界是向量
            U_samples = np.random.uniform(u_min, u_max, (n_samples, n_u))

        rho_list = []
        u_norm_list = []

        # 辅助矩阵
        I_nz = np.eye(n_z)

        # 4. 循环计算谱半径
        print("Calculating spectral radius for samples...")
        for k in range(n_samples):
            u_k = U_samples[k]
            u_norm = np.linalg.norm(u_k)

            # 计算时变项 H(u) = \sum u_i H_i
            # 利用张量点乘快速计算: H_tensor * u -> (n_z, n_z)
            H_u = H_tensor @ u_k

            # 构建 ESO 误差动力学矩阵 Phi
            # E_{k+1} = Phi * E_k
            # Phi = [ A + H(u) - L1*C   I ]
            #       [ -L2*C             I ]

            block11 = A + H_u - L1 @ C
            # block11 = A - L1 @ C
            block12 = I_nz
            block21 = -L2 @ C
            block22 = I_nz

            Phi_top = np.hstack([block11, block12])
            Phi_bot = np.hstack([block21, block22])
            Phi = np.vstack([Phi_top, Phi_bot])

            # 计算特征值并取最大模
            eigenvalues = np.linalg.eigvals(Phi)
            rho = np.max(np.abs(eigenvalues))

            rho_list.append(rho - 0.305)
            u_norm_list.append(u_norm)

        # 5. 结果分析与可视化
        max_rho = np.max(rho_list)
        print("\nResult Summary:")
        print(f"  Max Spectral Radius: {max_rho:.6f}")
        print(f"  Mean Spectral Radius: {np.mean(rho_list):.6f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(u_norm_list, rho_list, alpha=0.6, s=10, c=rho_list, cmap="viridis")
        plt.axhline(1.0, color="r", linestyle="--", linewidth=2, label="Stability Limit (rho=1)")
        cbar = plt.colorbar()
        # 设置侧边标签的字体大小
        # cbar.set_label('Spectral Radius', fontsize=14)
        # (可选) 设置侧边刻度数字(0.97, 0.98...)的字体大小
        cbar.ax.tick_params(labelsize=14)
        plt.xlabel("Control Input Norm ||u||", fontsize=16)
        plt.ylabel("Spectral Radius rho(Phi)", fontsize=16)
        plt.title(
            f"Stability Verification of ESO Error Dynamics\n(N={n_samples}, u in [{u_min}, {u_max}])", fontsize=16
        )
        plt.legend(fontsize=16)
        # plt.grid(True, alpha=0.3)
        plt.tick_params(axis="both", which="major", labelsize=16)
        # 自动保存或显示
        plt.savefig("eso_stability_check.png", dpi=300)
        plt.show()

        if max_rho < 1.0:
            print("\n✅ 验证通过: 系统在采样范围内满足谱半径 < 1 的稳定性条件。")
        else:
            print(f"\n❌ 验证警告: 存在 {np.sum(np.array(rho_list) >= 1.0)} 个采样点谱半径 >= 1。")
            print("建议: 尝试增大 L1/L2 或检查 Koopman 模型训练是否导致 A 矩阵本身不稳定。")


class MPCController_UKF(MPCController):
    """
    IBKN + UKF：在基础 MPC 之上叠加 无迹卡尔曼滤波(Unscented Kalman Filter)，
    对应 README 的"抗噪声"实验。相比 MPCController_KESO 里的线性 KF，
    UKF 用一组"sigma 点"显式传播双线性动力学的非线性/耦合项，不需要像
    线性 KF 那样把双线性项当成"已知输入下的固定线性项"来近似，理论上
    更适合双线性 Koopman 模型这种含有 z⊗u 耦合项的系统。
    """

    def __init__(self, net, args):
        """
        继承 MPC_KF 的初始化，包含 Koopman 模型、C矩阵、MPC solver、KF/UKF噪声协方差等。
        """
        super().__init__(net, args)  # 调用父类初始化
        n = self.Nkoopman
        # UKF 特有参数：alpha/beta/kappa 是标准 UKF(无迹变换)的形状超参，
        # 决定 sigma 点相对均值的分布"扩散程度"，lambda_ 是由它们导出的
        # 缩放因子，用于后面构造 sigma 点和加权求和的权重 Wm/Wc。
        self.alpha = 1e-2  # UKF 参数: 控制 sigma 点分布宽度 1e-3 ~ 1
        self.beta = 2.0  # UKF 参数: 对高斯分布最优
        self.kappa = 0  # UKF 参数: 二阶扩展项 0 / 3-n
        self.lambda_ = self.alpha**2 * (self.Nkoopman + self.kappa) - self.Nkoopman
        # 权重：Wm 用于加权求均值，Wc 用于加权求协方差，第 0 个 sigma 点
        # (对应均值本身)权重单独计算，其余 2n 个对称分布的 sigma 点权重相等。
        self.Wm = np.full(2 * n + 1, 1 / (2 * (n + self.lambda_)))
        self.Wc = np.full(2 * n + 1, 1 / (2 * (n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        # KF参数
        # pos_noise = 0.001    # 位置噪声 (m^2)
        # vel_noise = 0.001    # 角度噪声 (m/s)^2
        # self.W = np.diag([pos_noise]*3 + [vel_noise]*7 + [0.001]*(37-10))  # 其他状态噪声更低
        self.W = np.eye(self.Nkoopman) * 0.0001  #  (37,37) 过程噪声协方差 0.0001  0.1
        self.V = np.eye(self.obs_num) * 0.1  #  (3,3)  观测噪声协方差 0.01  10
        self.P = np.eye(self.Nkoopman) * 0.001  # (37,37)  初始协方差  self.obs_num

    def predict(self, z, u):
        """
        UKF 预测步：围绕当前状态 z 按协方差 P 生成 2n+1 个 sigma 点，逐个
        代入真实的(含双线性项的)动力学方程传播，再加权求出预测均值和协方差
        ——这就是"无迹变换"名字的由来: 不需要对非线性函数做解析线性化
        (不需要算雅可比矩阵)，用有限个精心选取的采样点就能近似传播均值和
        协方差到二阶精度。
        """
        # z: (37, 1) - 当前状态估计 u: (7,) - 控制输入向量
        n = self.Nkoopman
        # 对称化 + 正定化
        # self.P = (self.P + self.P.T) / 2
        # eigvals, eigvecs = np.linalg.eigh(self.P)
        # eigvals = np.clip(eigvals, 1e-3, 1e2)
        # self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Cholesky
        # 用协方差矩阵的 Cholesky 分解构造 sigma 点的偏移量；加 eps*I 是为了
        # 避免 P 数值上不是严格正定（浮点误差累积）导致 Cholesky 分解失败。
        eps = 1e-6
        Psqrt = np.linalg.cholesky((n + self.lambda_) * self.P + np.eye(n) * eps)

        # 2n+1 个 sigma 点：均值本身 + 沿协方差"主轴"正负各偏移一次。
        sigma_pts = np.hstack([z, z + Psqrt, z - Psqrt])  # (n, 2n+1) -> (37, 75)
        # 传播 sigma 点：每个 sigma 点各自代入完整的(含双线性项的)动力学方程，
        # 而不是像线性 KF 那样只用一个固定的 A 矩阵传播均值。
        sigma_pts_pred = self.Ad @ sigma_pts + self.Bd @ u.reshape(-1, 1)  # (37, 75)
        if hasattr(self.net, "H"):
            # 根据网络设置选择克罗内克积顺序
            if self.net.u_z:
                kron_product = np.kron(u.reshape(-1, 1), sigma_pts)  # (224, 75)
            else:
                kron_product = np.kron(sigma_pts, u.reshape(-1, 1))  # (224, 75)
            # 添加双线性项
            sigma_pts_pred += self.Hd @ kron_product

        sigma_pts_pred = sigma_pts_pred.T  # (75, 37)
        # 均值：对传播后的 sigma 点按 Wm 加权求和。
        z_pred = np.sum(self.Wm[:, None] * sigma_pts_pred, axis=0)  # (37,)
        # 协方差：对传播后 sigma 点相对新均值的偏差按 Wc 加权求外积之和，
        # 再加上过程噪声协方差 W。
        dz = sigma_pts_pred - z_pred.T  # (75, 37)
        P_pred = dz.T @ (self.Wc[:, None] * dz) + self.W  # (37, 37)

        self.z_ = z_pred.reshape(-1, 1)
        self.P = P_pred

        return self.z_, sigma_pts_pred

    def update_kf(self, x_meas, z, u):
        """UKF 更新步骤"""
        _, sigma_pts_pred = self.predict(z, u)
        m = self.obs_num

        # 获取观测 sigma 点：把每个传播后的 sigma 点解码回原始状态，
        # 再截取观测量对应的前 m 维，得到"预测的观测空间 sigma 点"。
        sigma_tensor = torch.tensor(sigma_pts_pred, dtype=torch.float64, device=self.device)  # (75, 37)
        sigma_pts_meas = self.net.x_decoder(sigma_tensor)[:, :m].detach().cpu().numpy()  # (75, 3)

        # 观测均值
        y_pred = sigma_pts_meas.T @ self.Wm  # (3,)
        # 观测误差打印（调试用）
        # print("x_meas:", x_meas[:3])
        # print("y_pred:", y_pred)
        # print("预测误差:", np.linalg.norm(x_meas[:3] - y_pred))
        # 协方差
        dy = sigma_pts_meas - y_pred.T  # (75, 3)
        dz = sigma_pts_pred - self.z_.T  # (75, 37)
        Pyy = dy.T @ (self.Wc[:, None] * dy) + self.V  # (3, 3)
        Pxy = dz.T @ (self.Wc[:, None] * dy)  # (37, 3)

        # 卡尔曼增益 & 更新（形式上和线性 KF 完全一样，区别只在于 Pxy/Pyy
        # 是通过 sigma 点传播算出来的，隐含了非线性/双线性信息）。
        K = Pxy @ np.linalg.inv(Pyy)  # (37, 3) × (3, 3) → (37, 3)
        self.z_ += K @ (x_meas.reshape(-1) - y_pred).reshape(-1, 1)  # (37,) + (37,3)×(3,)→(37,)→(37,1)
        self.P -= K @ Pyy @ K.T  # (37,37) - (37,3)×(3,3)×(3,37)→(37,37)

        return self.z_

    def get_updated_state(self, state_pre, z_last, a):
        """跑一次 UKF 更新，把滤波后的 Koopman 状态解码回原始物理状态。"""
        state_full = self.update_kf(state_pre[: self.obs_num], z_last, a.reshape(-1, 1)).T
        state = self.net.x_decoder(torch.DoubleTensor(state_full).to(self.device)).detach().cpu().numpy().reshape(-1)
        return state[3:]
