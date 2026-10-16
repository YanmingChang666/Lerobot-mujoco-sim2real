"""
模型工厂：根据 args.model 这个字符串，把 args 里对应的一组超参组装成
构造函数参数，实例化出具体的 Koopman 网络并搬到目标 device 上。

train.py 只依赖这一个函数就能拿到任意架构的模型实例，不需要在训练代码里
写一长串 if/elif 去分别处理每种网络的构造参数——这些参数的来源和含义见
args.py 的 process_args()（各架构的 layers/blocks/channels/hiddens 等都是
在那里按模型家族分组定义的）。
"""

from .InvertKoopman import InvertKoopmanNetBLinear, InvertKoopmanNetLinear
from .KANKoopman import KANKoopmanNet
from .Koopform import Koopformer, Koopformer_KAN
from .KoopmanBase import KoopmanBlinear, Koopmanlinear
from .KoopmanLSTM import KoopmanLSTMlinear, KoopmanLSTMlinear_KAN


def init_model(args):
    """按 args.model 选择对应的网络类并实例化，返回已经 .to(args.device) 的模型。"""
    print(f"Initiating {args.model}")
    if args.model == "DKUC":
        # 线性 Koopman：z_{t+1} = K @ z_t + B @ u_t，最基础的版本，
        # 不含双线性耦合项，训练/推理最快，通常作为基线模型。
        model = Koopmanlinear(args.x_dim, args.u_dim, args.layers, args.use_stable).to(args.device)
        return model
    elif args.model == "DBKN":
        # 双线性 Koopman：控制量 u 会先经过 u_z 相关的编码再和状态耦合，
        # 能表达"控制量如何调制系统响应"这种线性模型学不出来的效应，
        # 是 README 里默认主推的模型（结合 KESO 做抗扰动实验）。
        model = KoopmanBlinear(args.x_dim, args.u_dim, args.layers, args.u_z, args.use_stable).to(args.device)
        return model
    elif args.model == "Koopformer":
        # 用 Transformer 编码一段历史窗口(args.seq_len)得到观测空间表示，
        # 比 MLP encoder 更能捕捉时序依赖，但训练/推理开销也更大。
        model = Koopformer(args.x_dim, args.u_dim, args.seq_len, args.d_model, args.use_stable, args.use_decoder).to(
            args.device
        )
        return model
    elif args.model == "Koopformer_KAN":
        # Koopformer 的变体：把内部的 MLP 换成 KAN（Kolmogorov-Arnold Network）。
        model = Koopformer_KAN(
            args.x_dim, args.u_dim, args.seq_len, args.d_model, args.use_stable, args.use_decoder
        ).to(args.device)
        return model
    elif args.model == "KANKoopman":
        # 用 KAN 结构替代普通 MLP 做 encoder，理论上用更少参数拟合复杂非线性映射。
        model = KANKoopmanNet(
            args.x_dim, args.u_dim, args.kan_layers, args.kan_params, args.use_stable, args.use_decoder
        ).to(args.device)
        return model
    elif args.model == "KoopmanLSTMlinear":
        # 用 LSTM 编码历史窗口(args.lstm_seq_len)代替 Transformer，
        # 参数更少，适合数据量有限或想要更轻量模型的场景。
        model = KoopmanLSTMlinear(
            args.x_dim,
            args.u_dim,
            args.lstm_seq_len,
            args.LSTM_encode_layers,
            args.LSTM_Hidden,
            args.use_stable,
            args.use_decoder,
        ).to(args.device)
        return model
    elif args.model == "KoopmanLSTMlinear_KAN":
        # KoopmanLSTMlinear 的变体：encoder 部分换成 KAN。
        model = KoopmanLSTMlinear_KAN(
            args.x_dim,
            args.u_dim,
            args.lstm_seq_len,
            args.kan_layers,
            args.LSTM_Hidden,
            args.use_stable,
            args.use_decoder,
        ).to(args.device)
        return model
    elif args.model == "IKN":
        # 可逆网络版线性 Koopman：encoder/decoder 是同一个可逆双射网络的正/反向，
        # 天然保证解码能"精确"还原编码前的信息（不像普通 autoencoder 那样
        # 编码/解码是两个独立网络、只能靠重构损失去逼近可逆性）。
        model = InvertKoopmanNetLinear(
            args.x_dim,
            args.x_blocks,
            args.x_channels,
            args.x_hiddens,
            args.u_dim,
            args.u_blocks,
            args.u_channels,
            args.u_hiddens,
            args.use_stable,
        ).to(args.device)
        return model
    elif args.model == "IBKN":
        # 可逆网络版双线性 Koopman：在 IKN 基础上加入 u_z 双线性耦合，
        # README 里另一大主力实验模型（结合 UKF 做抗观测噪声实验）。
        model = InvertKoopmanNetBLinear(
            args.x_dim,
            args.x_blocks,
            args.x_channels,
            args.x_hiddens,
            args.u_dim,
            args.u_blocks,
            args.u_channels,
            args.u_hiddens,
            args.u_z,
            args.use_stable,
        ).to(args.device)
        return model

    else:
        raise ValueError(f"Model {args.model} not implemented!")
