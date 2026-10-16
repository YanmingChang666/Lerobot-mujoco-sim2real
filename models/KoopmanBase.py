"""
DKUC(线性 Koopman) / DBKN(双线性 Koopman) 的具体网络实现。

Koopmanlinear 对应 args.model == "DKUC"，KoopmanBlinear 继承自它、
对应 args.model == "DBKN"（在线性演化项之外再加一个双线性项）。
文件末尾的 DKN/DKAC 是两个更早期的双线性变体，目前没有被 models/init_model.py
暴露给 --model 参数，保留在这里仅供参考/实验，不在主训练流程中使用。
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .base_model import StableKoopmanOperator


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class Koopmanlinear(StableKoopmanOperator):
    """
    线性 Koopman 网络 (DKUC)：z_{t+1} = K @ z_t + B @ u_t。

    x_encoder 用一个普通 MLP 提取非线性特征，再和原始状态 x 拼接成完整的
    Koopman 观测向量 z（"lifting"），这样即使解码器是固定的截取操作
    （见 base_model.StableKoopmanOperator 里 use_decoder=False 的 lC），
    也不会丢失原始状态信息。控制输入 u 不做任何非线性变换，直接送进
    线性演化项 B @ u_t。
    """

    def __init__(self, x_dim, u_dim, encode_layers, use_stable=False, use_decoder=False):
        super().__init__(x_dim, u_dim, encode_layers, use_stable, use_decoder)
        Layers = OrderedDict()
        # 按 encode_layers 描述的维度列表逐层搭建 MLP，除最后一层外都接 ReLU，
        # 例如 args.layers=[8,64,64,64,24] 就是 4 层线性 + 3 个 ReLU。
        for layer_i in range(len(encode_layers) - 1):
            Layers[f"linear_{layer_i}"] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers[f"relu_{layer_i}"] = nn.ReLU()

        # 状态输入编码器
        self.x_encode_net = nn.Sequential(Layers)
        # 控制输入编码器：恒等映射（直接返回原始控制输入）
        self.u_encode_net = nn.Identity()

    def x_encoder(self, x):
        """x -> [x, MLP(x)]：把原始状态和 MLP 提取的非线性特征拼接成完整的 Koopman 观测向量。"""
        feat = self.x_encode_net(x)
        return torch.cat([x, feat], dim=-1)
        # concat 原始 + 特征

    def x_decoder(self, x_emb):
        """线性 Koopman 的解码：用固定的 [I, 0] 矩阵截取观测向量的前 x_dim 维还原状态。"""
        return self.lC(x_emb)

    def u_encoder(self, x, u):
        """线性模型里控制量不需要编码，原样返回。"""
        return self.u_encode_net(u)

    def u_decoder(self, u_emb):
        return u_emb


class KoopmanBlinear(Koopmanlinear):
    """
    双线性 Koopman 网络 (DBKN)：在线性项 K@z + B@u 基础上，再加一个
    "状态-控制交叉项" H @ (z⊗u)（或 u⊗z），用来表达控制量如何调制系统的
    响应特性——这是纯线性模型无法表示的效应，也是 DBKN 相比 DKUC 表达力
    更强的来源。
    """

    def __init__(self, x_dim, u_dim, encode_layers, u_z, use_stable):
        super().__init__(x_dim, u_dim, encode_layers, use_stable)
        # 双线性部分
        # H 把 Kronecker 积展开后的 (Nkoopman*u_dim) 维向量线性映射回 Nkoopman 维，
        # 等价于对每个控制维度学一个独立的 Nkoopman x Nkoopman 矩阵 H_i
        # （具体拆分见 get_Hi_list/get_Hi_numpy）。零初始化是为了让训练刚开始时
        # 双线性项不干扰线性项已经学到的动力学，平稳地"叠加"额外的非线性修正。
        self.H = nn.Linear(self.Nkoopman * self.u_dim, self.Nkoopman, bias=False)
        nn.init.zeros_(self.H.weight)
        # u_z 决定 Kronecker 积的顺序是 u⊗z 还是 z⊗u，两者数学上等价（可以用
        # 一个置换矩阵相互转换，见 build_permutation_matrix），但训练出的 H
        # 权重排列方式不同，MPC_Controler 里读取 H 时要用同样的约定来切片。
        self.u_z = u_z

    def koopman_operation(self, x_emb, u_emb):
        # u_emb: u_dim   x_emb:Nkoopman
        if self.use_stable:
            K = self.get_koopman_matrix_K()
            linear_term = x_emb @ K.T + self.lB(u_emb)
        else:
            linear_term = self.lA(x_emb) + self.lB(u_emb)
        # 双线性项  u_dim*Nkoopman
        # einsum("bi,bj->bij", a, b) 对批次里每个样本计算外积 a⊗b，再 reshape
        # 展平成一维向量，就是 Kronecker 积的批量实现；self.u_z 只是决定
        # 外积的两个操作数谁在前，对应上面 __init__ 里的注释。
        if self.u_z:
            z_kron_u = torch.einsum("bi,bj->bij", u_emb, x_emb).reshape(x_emb.shape[0], -1)
        else:
            z_kron_u = torch.einsum("bi,bj->bij", x_emb, u_emb).reshape(x_emb.shape[0], -1)
        bilinear_term = self.H(z_kron_u)
        return linear_term + bilinear_term

    def build_permutation_matrix(self, n, m):
        # 返回 P ∈ R^{nm × nm}，将 vec(u⊗z) → vec(z⊗u)
        # 这是纯粹的线性代数工具：Kronecker 积 vec(u⊗z) 和 vec(z⊗u) 只是把同样
        # n*m 个元素按不同顺序排列，可以用一个 0/1 置换矩阵在两者之间转换。
        # get_Hi_numpy() 用它把训练时按 u_z 约定存储的 H 权重统一转成
        # "z⊗u" 顺序，方便 MPC 那边用固定的方式解析每个控制维度对应的 H_i。
        if self.u_z:
            P = np.zeros((n * m, n * m))
            for i in range(n):
                for j in range(m):
                    row = i * m + j
                    col = j * n + i
                    P[col, row] = 1
        else:
            P = np.eye(n * m)
        return P

    def get_Hi_list(self):
        """把展平的 H 权重按控制维度切成 u_dim 个 (Nkoopman, Nkoopman) 或按状态维度
        切成 Nkoopman 个 (Nkoopman, u_dim) 子矩阵（取决于 u_z），保留为 torch 张量，
        用于训练脚本内部检查/正则化双线性项。"""
        Hd = self.H.weight.clone()
        if self.u_z:
            H_blocks = [
                Hd[:, i * self.Nkoopman : (i + 1) * self.Nkoopman] for i in range(self.u_dim)
            ]  # 每块 shape: (N, N)
        else:
            H_blocks = [
                Hd[:, j * self.u_dim : (j + 1) * self.u_dim] for j in range(self.Nkoopman)
            ]  # 每块 shape: (N, m)
        return H_blocks

    def get_Hi_numpy(self):
        """
        导出训练好的双线性算子 H_i（每个控制维度一个 Nkoopman x Nkoopman 矩阵），
        供 control/MPC_Controler.py 构建 CasADi 优化问题时使用：MPC 需要把
        双线性动力学 z_{t+1} = K z_t + B u_t + sum_i u_{t,i} * H_i @ z_t 显式地
        写成 CasADi 符号表达式，因此要从 PyTorch 权重里把每个 H_i 单独抠出来，
        转成 numpy 数组。
        """
        P = self.build_permutation_matrix(self.u_dim, self.Nkoopman)  # 交换矩阵
        Hd = self.H.weight.cpu().detach().numpy() @ P.T  # (32, 224) 转成 z⊗u 的 H
        H_hat_list = []
        for j in range(self.Nkoopman):
            start_idx = j * self.u_dim
            end_idx = (j + 1) * self.u_dim
            H_hat_j = Hd[:, start_idx:end_idx].copy()  # (32, 7)
            H_hat_list.append(H_hat_j)
        return H_hat_list


class DKN(Koopmanlinear):
    """
    早期的双线性变体：用一个额外的 MLP，把 [x, u] 拼接编码成新的控制表示，
    而不是像 KoopmanBlinear 那样用显式的 Kronecker 积项。
    未接入 models/init_model.py，目前不在主训练/实验流程中使用。
    """

    def __init__(self, x_dim, u_dim, encode_layers):
        super().__init__(x_dim, u_dim, encode_layers)
        # 双线性部分
        BLayers = OrderedDict()
        bilinear_layers = [x_dim + u_dim, 64, 64, 64, u_dim]
        for i in range(len(bilinear_layers) - 1):
            BLayers[f"linear_{i}"] = nn.Linear(bilinear_layers[i], bilinear_layers[i + 1])
            if i != len(bilinear_layers) - 2:
                BLayers[f"relu_{i}"] = nn.ReLU()
        self.u_encode_net = nn.Sequential(BLayers)

    def u_encoder(self, x, u):
        return self.u_encode_net(torch.cat([x, u], dim=-1))


class DKAC(Koopmanlinear):
    """
    早期的双线性变体：学一个只依赖状态 x 的"增益" gu = g(x)，再和原始控制量
    逐元素相乘 (gu * u) 得到调制后的控制表示，是一种更简化的状态-控制耦合方式。
    未接入 models/init_model.py，目前不在主训练/实验流程中使用。
    """

    def __init__(self, x_dim, u_dim, encode_layers):
        super().__init__(x_dim, u_dim, encode_layers)
        # 双线性部分
        BLayers = OrderedDict()
        bilinear_layers = [x_dim, 64, 64, 64, u_dim]
        for i in range(len(bilinear_layers) - 1):
            BLayers[f"linear_{i}"] = nn.Linear(bilinear_layers[i], bilinear_layers[i + 1])
            if i != len(bilinear_layers) - 2:
                BLayers[f"relu_{i}"] = nn.ReLU()
        self.u_encode_net = nn.Sequential(BLayers)

    def u_encoder(self, x, u):
        gu = self.u_encode_net(x)
        return gu * u
