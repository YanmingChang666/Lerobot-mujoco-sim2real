"""
所有 Koopman 网络共用的基类与核心算子实现。

Koopman 算子理论的核心想法：原始的机械臂关节角/末端位置随时间的演化是
非线性的，很难直接用线性模型描述；但如果先把状态 x 通过一个非线性的
encoder 映射到一个更高维的"观测空间" z = phi(x)，理论上存在一个（近似）
线性的算子 K，使得 z 在这个高维空间里的演化是线性的：

    z_{t+1} = K @ z_t + B @ u_t          (线性 Koopman，见 KoopmanBase.py 的 DKUC)
    z_{t+1} = f(z_t, u_t)  (双线性项)     (双线性 Koopman，见 KoopmanBase.py 的 DBKN)

再用一个 decoder 把 z 解码回原始状态 x_hat = phi^{-1}(z)。整个网络要学的
就是 encoder、算子 K/B、decoder 这三部分，训练目标是让多步滚动预测的
x_hat 和真实物理仿真的 x 尽量接近。这样训练完之后，K/B 是线性/双线性的，
就可以直接喂给 CasADi 的 MPC 求解器做实时优化（否则用原始非线性动力学
做 MPC 会慢得多，无法实时控制）。

本文件定义：
- KoopmanNet：抽象基类，规定所有子类必须实现的接口
  (x_encoder / u_encoder / koopman_operation / x_decoder / u_decoder)；
- StableKoopmanOperator：Koopman 线性演化算子的一种具体实现，
  重点是 use_stable=True 时如何"从数学上保证"多步预测不发散。
"""

import torch
from torch import Tensor, nn


class KoopmanNet(nn.Module):
    """
    Base class for all Koopman network

    所有具体的 Koopman 网络（DKUC/DBKN/IKN/IBKN/Koopformer/...）都遵循同一套
    "编码 -> 观测空间线性/双线性演化 -> 解码"接口，方便 train.py 里的训练/评估
    代码统一调用，不需要关心具体是哪种网络结构。
    """

    def __init__(self):
        super().__init__()

    def x_encoder(self, x: Tensor):
        """把原始状态 x 编码到 Koopman 观测空间，子类必须实现。"""
        raise NotImplementedError

    def u_encoder(self, x: Tensor, u: Tensor):
        """把控制输入 u（可能还依赖当前状态 x，用于双线性网络）编码到观测空间，子类必须实现。"""
        raise NotImplementedError

    def koopman_operation(self, x_emb: Tensor, u_emb: Tensor):
        """在 Koopman 观测空间里做一步线性/双线性演化，子类必须实现。"""
        raise NotImplementedError

    def x_decoder(self, x_emb: Tensor):
        """把 Koopman 观测空间的状态解码回原始状态 x，子类必须实现。"""
        raise NotImplementedError

    def u_decoder(self, u_emb: Tensor):
        """把 Koopman 观测空间的控制表示解码回原始控制量 u（可逆网络才需要），子类必须实现。"""
        raise NotImplementedError


# --- 辅助函数：保持不变 ---
def _orth(w: torch.Tensor) -> torch.Tensor:
    """
    一个辅助函数, 用于将输入的方阵正交化 (orthogonalize)。

    用 QR 分解取 Q 部分：任意方阵 w 经过 QR 分解后，Q 一定是正交矩阵
    (Q @ Q.T = I)。StableKoopmanOperator 用它把可训练的自由参数矩阵
    "投影"成正交矩阵，是构造严格稳定 Koopman 算子的关键一步（见下方
    get_koopman_matrix_K 的奇异值分解式构造：K = U @ diag(Sigma) @ V.T）。
    """
    return torch.linalg.qr(w)[0]


def gaussian_init_(n_units, std=1):
    """高斯初始化函数"""
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class StableKoopmanOperator(KoopmanNet):
    """
    一个严格稳定的离散时间库普曼算子模块。
    实现了动力学: z_{t+1} = K @ z_t + B @ u_t

    "稳定"指的是：不管初始状态 z_0 是什么，只要没有外部控制输入 u 持续注入
    能量，z_t 在 t->infinity 时不会发散到无穷大——这要求转移矩阵 K 的谱半径
    (最大特征值模长) 严格小于 1。对于神经网络自由学出来的 K，训练过程中
    没有任何机制能保证这一点，多步滚动预测时误差可能被不稳定的 K 指数放大，
    这正是 use_stable=True 要解决的问题（代价是牺牲一部分表达能力，
    K 被约束成只能表示"收缩"性质的线性动力学）。
    """

    def __init__(
        self, x_dim: int, u_dim: int, encoder_layers, use_stable=False, use_decoder=False, rho_max: float = 0.99
    ):
        super().__init__()
        self.u_dim = u_dim
        self.x_dim = x_dim
        # 提取非线性特征的维度
        # use_decoder=False 时，Koopman 观测空间由 [原始状态 x, encoder 非线性特征]
        # 拼接而成（"lifting"里常见的做法），这样即使解码矩阵 C 不训练、只是
        # 简单截取前 x_dim 维，也能保证信息不丢失；use_decoder=True 时则完全
        # 信任 encoder 的输出就是整个观测空间维度，交给可训练的解码器去恢复 x。
        if use_decoder:
            self.Nkoopman = encoder_layers[-1]
        else:
            self.Nkoopman = encoder_layers[-1] + x_dim
        self.use_stable = use_stable
        self.use_decoder = use_decoder
        self.rho_max = rho_max

        if use_stable:
            # 状态转移矩阵 K 的参数 (Orthogonal-Diagonal-Orthogonal 分解)
            # 不直接学 K 本身，而是学习两个待正交化的矩阵 U_raw/V_raw 和一组
            # "奇异值"参数 S_raw，真正的 K 在 get_koopman_matrix_K() 里动态构建，
            # 这样才能在训练过程中始终维持 K 的谱半径 < rho_max 这一约束。
            self.U_raw = nn.Parameter(torch.randn(self.Nkoopman, self.Nkoopman))
            self.V_raw = nn.Parameter(torch.randn(self.Nkoopman, self.Nkoopman))
            self.S_raw = nn.Parameter(torch.randn(self.Nkoopman))
        else:
            # koopman矩阵
            # 不加稳定性约束：K 是普通的可训练线性层，只在初始化时做了一次
            # SVD 归一化（让初始奇异值都在 0.9 附近），避免训练一开始就发散，
            # 但训练过程中 K 完全可以被优化器推到任意谱半径。
            self.lA = nn.Linear(self.Nkoopman, self.Nkoopman, bias=False)
            self.lA.weight.data = gaussian_init_(self.Nkoopman, std=1)
            U, _, V = torch.svd(self.lA.weight.data)
            self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        # 控制矩阵
        self.lB = nn.Linear(u_dim, self.Nkoopman, bias=False)
        nn.init.xavier_uniform_(self.lB.weight)
        # 解码矩阵 C：将潜在状态 z 解码回原始状态 x
        if not use_decoder:
            self.lC = nn.Linear(self.Nkoopman, self.x_dim, bias=False)
            # 手动初始化权重（单位矩阵 + 零填充）
            # 因为 Nkoopman 维观测空间的前 x_dim 维就是原始状态 x 本身
            # （见上面的拼接逻辑），解码只需要"取前 x_dim 维、丢弃后面的
            # 非线性特征"，用单位矩阵 + 零填充恰好实现这个截取操作，
            # 且 requires_grad=False 冻结它，不用浪费训练资源去学一个
            # 本该是恒等映射的东西。
            with torch.no_grad():
                self.lC.weight.data[: self.x_dim, : self.x_dim] = torch.eye(self.x_dim)
                self.lC.weight.data[:, self.x_dim :] = 0.0
            self.lC.weight.requires_grad = False

    def get_koopman_matrix_K(self) -> torch.Tensor:
        """
        动态构建严格稳定的库普man矩阵 K。
        通过限制其奇异值来保证谱半径小于 rho_max。
        """
        # U、V 是正交矩阵，Sigma 是对角阵且每个元素都被 sigmoid 压到 (0, rho_max)
        # 区间内，因此 K = U @ diag(Sigma) @ V.T 的所有奇异值都 < rho_max < 1，
        # 而谱半径不超过最大奇异值，所以 K 的谱半径必然 < rho_max，
        # 天然保证了 z_{t+1}=K@z_t 这个自治系统是渐近稳定的。
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        # Sigmoid 将奇异值限制在 (0, 1) 区间, 再乘以 rho_max
        Sigma = torch.sigmoid(self.S_raw) * self.rho_max
        K = U @ torch.diag(Sigma) @ V.T
        return K

    def koopman_operation(self, x_emb: Tensor, u_emb: Tensor):
        """
        在Koopman空间中进行线性演化。
        """
        if self.use_stable:
            # 每次前向都要重新用 U_raw/V_raw/S_raw 构建 K（而不是缓存），
            # 因为这三个参数在训练过程中会被更新，K 必须跟着重新计算。
            K = self.get_koopman_matrix_K()
            return x_emb @ K.T + self.lB(u_emb)
        else:
            return self.lA(x_emb) + self.lB(u_emb)
