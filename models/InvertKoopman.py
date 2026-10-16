"""
IKN(可逆线性 Koopman) / IBKN(可逆双线性 Koopman) 的具体网络实现。

和 KoopmanBase.py 里 DKUC/DBKN 的做法不同：DKUC 的 encoder 是普通 MLP，
decoder 是一个固定/近似的映射，编码-解码不保证严格互逆，只能靠训练时的
重构损失去"拉近"；这里的 IKN/IBKN 用类似 NICE/RealNVP 的**加性耦合层**
搭出一个数学上严格可逆的网络 (iMLPNet)，x_decoder 直接调用
x_encoder 网络的 .inverse()，编码和解码永远是精确的互逆关系，不需要
额外的重构损失来"凑"。

核心构件从下到上：
    split/merge          : 把一个向量在通道维对半拆开/拼回去
    injective_pad         : 用零填充把输入维度补到偶数且能整除，
                             使得输入输出维度不一致时耦合层依然可逆
    imlp_block             : 一个加性耦合层 (y1=F(x2)+x1, y2=x2)，
                             对应 NICE 论文里的加性耦合，天然可逆
    iMLPNet                 : 堆叠多个 imlp_block，每层可以改变通道宽度，
                             整体构成一个可逆的深层非线性映射
    InvertKoopmanNetLinear   : 用 iMLPNet 做 encoder/decoder 的线性 Koopman (IKN)
    InvertKoopmanNetBLinear  : 在 IKN 基础上加双线性项的版本 (IBKN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_model import StableKoopmanOperator


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


def split(x):
    """按通道维对半拆分成两份，是加性耦合层的输入预处理。"""
    n = int(x.size()[1] / 2)
    x1 = x[:, :n].contiguous()
    x2 = x[:, n:].contiguous()
    return x1, x2


def merge(x1, x2):
    """split 的逆操作：把两份张量按通道维拼回一个整体。"""
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    """
    Expansion of the input dimension by pad zero

    Input dim: [B, D] = batch size * feature dimension

    Args:
        pad_size (int): The number of pad zero

    加性耦合层 (imlp_block) 要求输入能对半拆成两份、输出也能对半拆成两份，
    但 Koopman 网络里输入输出维度往往不一致（比如状态维 x_dim=8 要提升到
    更高维的观测空间）。这个模块通过"补零再对半拆"的技巧，把维度提升
    这件事也变成可逆操作：forward 补零，inverse 直接把补的零截掉，
    补零本身不引入任何信息损失，所以整体依然严格可逆。
    """

    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size), "constant", 0)
        return x

    def inverse(self, x):
        return x[:, : x.size(1) - self.pad_size]


class imlp_block(nn.Module):
    """
    Args:
        in_ch (int): The dimension of input features, total
        hidden_ch (int): The dimension of hidden dimension
        out_ch (int): The dimension of output features, local (total = local * 2)

    一个"加性耦合层"（参考 NICE/RealNVP 的 additive coupling）：
    输入拆成 (x1, x2)，只用一个普通 MLP（bottleneck_block）算出 x2 的
    非线性变换 F(x2)，然后 y1 = F(x2) + x1、y2 = x2（本实现里输出顺序是
    (x2, y1)，等价于做了一次通道交换，方便堆叠多层时让每一半都有机会
    被变换到）。

    可逆性的关键：inverse 时已知 (x2, y1)，直接用同一个 bottleneck_block
    重新算一遍 F(x2)，就能精确解出 x1 = y1 - F(x2)——因为加法是可逆的，
    不需要对 bottleneck_block 本身做任何可逆性限制，MLP 可以是任意非线性
    结构，这是加性耦合层的核心优势。
    """

    def __init__(self, in_ch, out_ch, hidden_ch):
        """build invertible MLP bottleneck block"""
        super().__init__()
        # pad_size 用来把输入维度补齐到 2*out_ch，保证输入输出的通道对半拆分
        # 关系能对上；当输入维度已经等于 2*out_ch 时 pad_size=0，不需要补零。
        self.pad_size = 2 * out_ch - in_ch
        self.inj_pad = injective_pad(self.pad_size)

        if self.pad_size != 0:
            in_ch = out_ch * 2

        layers = []
        layers.append(nn.Linear(in_ch // 2, hidden_ch))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_ch, hidden_ch))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_ch, out_ch))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """bijective or injective block forward"""
        if self.pad_size != 0:
            # 维度不匹配时先合并再补零、再重新拆分，让 (x1, x2) 的通道数
            # 变成耦合公式要求的形状。
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        y1 = Fx2 + x1
        # 返回 (x2, y1) 而不是 (y1, x2)：交换顺序是为了让下一层耦合层
        # 变换的是这一层没变过的那一半，堆叠多层后两半都能被充分混合。
        return (x2, y1)

    def inverse(self, x):
        """bijective or increasing block inverse"""
        x2, y1 = x[0], x[1]
        # 用同一个 bottleneck_block 重新计算 F(x2)，减法精确还原 x1，
        # 这就是加性耦合层"零成本可逆"的由来——不需要训练额外的逆网络。
        Fx2 = -self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.pad_size != 0:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iMLPNet(nn.Module):
    """
    Build the iMLPNet

    Args:
        nBlocks (list): The number of imlp_block in the defined net (depth)
        nChannels (list): The output dimension of the given block
        nHiddens (list): The hidden dimension of the given block

    把多个 imlp_block 堆叠起来的可逆网络：nChannels 里每个"阶段"的通道数
    逐级增大，配合 injective_pad 让网络在整体上实现"低维状态 -> 高维
    Koopman 观测空间"的提升，同时保持从头到尾严格可逆。forward 对应
    Koopman 网络的 x_encoder，inverse 对应 x_decoder。
    """

    def __init__(self, nBlocks, nChannels, nHiddens, in_shape=None):
        super().__init__()
        self.in_ch = in_shape
        self.nBlocks = nBlocks

        self.stack = self.imlp_stack(imlp_block, nBlocks, nChannels, nHiddens, in_ch=self.in_ch)

    def imlp_stack(self, _block, nBlocks, nChannels, nHiddens, in_ch):
        """Create stack of imlp blocks"""
        # 把 (nChannels[i], nBlocks[i], nHiddens[i]) 这种"每个阶段重复几次"的
        # 紧凑配置，展开成逐层的 (channel, hidden) 列表，例如 x_blocks=[2,2]、
        # x_channels=[12,16] 表示先用 12 通道的耦合层连续堆 2 层，
        # 再用 16 通道的耦合层连续堆 2 层，一共 4 层。
        block_list = nn.ModuleList()
        hiddens = []
        channels = []
        for channel, depth, hidden in zip(nChannels, nBlocks, nHiddens):
            hiddens = hiddens + ([hidden] * depth)
            channels = channels + ([channel] * depth)
        for channel, hidden in zip(channels, hiddens):
            block_list.append(
                _block(
                    in_ch,
                    channel,
                    hidden,
                )
            )
            # 下一层的输入通道数 = 本层输出通道数的两倍（因为 imlp_block
            # 输出的是一对 (x2, y1)，每份都是 channel 维）。
            in_ch = channel * 2
        return block_list

    def forward(self, x):
        """imlpnet forward"""
        n = self.in_ch // 2
        out = (x[:, :n], x[:, n:])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        return out_bij

    def inverse(self, out_bij):
        """imlpnet inverse"""
        out = split(out_bij)
        # 逆向传播必须按"后进先出"的顺序，从最后一层往第一层依次调用
        # inverse，这和 forward 时从第一层到最后一层的堆叠顺序正好相反，
        # 是可逆网络求逆的标准做法。
        for i in range(len(self.stack)):
            out = self.stack[-1 - i].inverse(out)
        out = merge(out[0], out[1])
        x = out
        return x


class InvertKoopmanNetLinear(StableKoopmanOperator):
    """
    可逆线性 Koopman 网络 (IKN)：用 iMLPNet 代替普通 MLP 做 x_encoder，
    x_decoder 直接调用 iMLPNet.inverse()，保证编码解码严格互逆。
    控制量 u 不做变换（Identity），线性演化项沿用 StableKoopmanOperator。
    """

    def __init__(
        self,
        x_dim,  # 状态维度
        x_blocks,  # 状态编码器块数
        x_channels,  # 状态编码器通道数
        x_hiddens,  # 状态编码器隐藏层大小
        u_dim,  # 控制输入维度
        u_blocks,  # 控制输入编码器块数
        u_channels,  # 控制输入编码器通道数
        u_hiddens,  # 控制输入编码器隐藏层大小
        use_stable,
        use_decoder=True,
    ):
        # 注意这里传给基类的 encoder_layers 是 [x_channels[-1]*2]，即最终
        # Koopman 观测空间维度 = 最后一级耦合层通道数的两倍（对应 iMLPNet
        # 输出时 merge 拼接的两份）；use_decoder 固定传 True，因为可逆网络
        # 的解码器就是 x_encode_net.inverse()，不需要 StableKoopmanOperator
        # 里那套"拼接原始状态 + 固定截取矩阵"的解码方式。
        super().__init__(x_dim, u_dim, [x_channels[-1] * 2], use_stable, use_decoder)
        self.x_dim = x_dim
        self.x_blocks = x_blocks
        self.x_channels = x_channels
        self.x_hiddens = x_hiddens

        self.u_dim = u_dim
        self.u_blocks = u_blocks
        self.u_channels = u_channels
        self.u_hiddens = u_hiddens

        self.Nkoopman = x_channels[-1] * 2
        self.x_encode_net = iMLPNet(
            nBlocks=self.x_blocks, nChannels=self.x_channels, nHiddens=self.x_hiddens, in_shape=self.x_dim
        )

        # 控制输入编码器：恒等映射（直接返回原始控制输入）
        self.u_encode_net = nn.Identity()

    def x_encoder(self, x: Tensor):
        return self.x_encode_net(x)

    def u_encoder(self, x: Tensor, u: Tensor):
        return self.u_encode_net(u)

    def x_decoder(self, x_emb: Tensor):
        """精确解码：调用可逆网络自身的 inverse，而非另一个独立训练的解码网络。"""
        return self.x_encode_net.inverse(x_emb)

    def u_decoder(self, u_emb: Tensor):
        return u_emb


class InvertKoopmanNetBLinear(InvertKoopmanNetLinear):
    """
    可逆双线性 Koopman 网络 (IBKN)：在 IKN 的可逆 encoder/decoder 基础上，
    额外加一个和 KoopmanBase.KoopmanBlinear 完全同样形式的双线性项 H@(z⊗u)。
    是 README 里"IBKN + UKF"抗噪声实验用的主力模型。
    """

    def __init__(
        self,
        x_dim,  # 状态维度
        x_blocks,  # 状态编码器块数
        x_channels,  # 状态编码器通道数
        x_hiddens,  # 状态编码器隐藏层大小
        u_dim,  # 控制输入维度
        u_blocks,  # 控制输入编码器块数
        u_channels,  # 控制输入编码器通道数
        u_hiddens,  # 控制输入编码器隐藏层大小
        u_z,
        use_stable,
    ):
        super().__init__(x_dim, x_blocks, x_channels, x_hiddens, u_dim, u_blocks, u_channels, u_hiddens, use_stable)

        # 和 KoopmanBase.KoopmanBlinear 里的 H 定义/初始化完全一致：
        # 零初始化让训练初期双线性项不干扰线性项，逐步学出额外的耦合修正。
        self.H = nn.Linear(self.Nkoopman * self.u_dim, self.Nkoopman, bias=False)
        nn.init.zeros_(self.H.weight)
        self.u_z = u_z

    def koopman_operation(self, x_emb, u_emb):
        # u_emb: u_dim   x_emb:Nkoopman
        if self.use_stable:
            K = self.get_koopman_matrix_K()
            linear_term = x_emb @ K.T + self.lB(u_emb)
        else:
            linear_term = self.lA(x_emb) + self.lB(u_emb)
        # 双线性项  u_dim*Nkoopman（做法和 KoopmanBase.KoopmanBlinear 完全一致，
        # 因为可逆网络只改变了 x/u 的编码-解码方式，不改变 Koopman 演化项本身
        # 的数学形式）
        if self.u_z:
            z_kron_u = torch.einsum("bi,bj->bij", u_emb, x_emb).reshape(x_emb.shape[0], -1)
        else:
            z_kron_u = torch.einsum("bi,bj->bij", x_emb, u_emb).reshape(x_emb.shape[0], -1)
        bilinear_term = self.H(z_kron_u)
        return linear_term + bilinear_term

    def build_permutation_matrix(self, n, m):
        # 返回 P ∈ R^{nm × nm}，将 vec(u⊗z) → vec(z⊗u)
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
        """导出每个控制维度对应的双线性矩阵 H_i，供 MPC_Controler 构建 CasADi
        优化问题时使用，含义和用法与 KoopmanBase.KoopmanBlinear.get_Hi_numpy 相同。"""
        P = self.build_permutation_matrix(self.u_dim, self.Nkoopman)  # 交换矩阵
        Hd = self.H.weight.cpu().detach().numpy() @ P.T  # (32, 224) 转成 z⊗u 的 H
        H_hat_list = []
        for j in range(self.Nkoopman):
            start_idx = j * self.u_dim
            end_idx = (j + 1) * self.u_dim
            H_hat_j = Hd[:, start_idx:end_idx].copy()  # (32, 7)
            H_hat_list.append(H_hat_j)
        return H_hat_list
