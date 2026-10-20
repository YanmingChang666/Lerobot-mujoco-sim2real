"""
全项目统一的命令行参数与派生配置模块。

设计上用一个 Args 类把 argparse 包了一层：
- _add_args() 只声明"用户可能想从命令行改"的参数（模型类型、训练超参、
  数据规模、控制实验开关等）；
- process_args() 负责把这些原始参数"展开"成实际代码里要用的派生值——
  文件路径、各个模型架构的层维度配置等，这些一般不需要用户手动指定，
  写在这里统一维护，避免散落在 train.py / models 各处。

train.py、control/*、DBKN_KESO.py、IBKN_UKF.py 等几乎所有入口脚本都通过
`args = Args()` 拿到同一份配置，是整个项目的参数总线。
"""

import argparse
import os


class Args:
    def __init__(self):
        # 初始化参数解析器
        self.parser = argparse.ArgumentParser(description="Koopman 模型训练/测试参数配置")
        self._add_args()  # 添加所有参数
        self.args = self.parser.parse_args()  # 解析命令行参数
        self.process_args()  # 动态处理路径等衍生参数

    def _add_args(self):
        # 核心参数
        self.parser.add_argument(
            "--model",
            type=str,
            default="DBKN",
            choices=[
                "DKUC",
                "DBKN",
                "IKN",
                "IBKN",
                "Koopformer",
                "all",
                "KoopmanLSTMlinear",
                "Koopformer",
                "KANKoopman",
            ],
            help="模型类型,可选值: DKUC, DBKN, IKN, IBKN, all (默认: all)",
        )
        self.parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "test"],
            help="运行模式, 可选值: train / test (默认: train)",
        )
        # env 决定用哪套机械臂资产（XML/URDF）、数据缓存目录、结果输出目录，
        # 具体映射关系见下面的 ROBOT_ASSET_LAYOUT。
        self.parser.add_argument("--env", type=str, default="SOARM101", help="环境名称（用于路径生成）(默认: SOARM101)")
        # suffix 是同一 env+model 下用来区分"不同批次实验"的标签，
        # 决定 results/<env>/<suffix>/<model>/ 这个输出目录，方便同时保留多组
        # checkpoint（比如不同 perturbation_limits 训出来的模型）互不覆盖。
        self.parser.add_argument("--suffix", type=str, default="12_22", help="实验后缀（用于路径区分）")
        self.parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")

        # 数据收集参数
        # 训练/测试用的样本数和步数会拼进缓存文件名(见 process_args)，
        # 改这两个值即视为"新的数据集"，会触发 SOARM101_DataCollection 重新采集。
        self.parser.add_argument("--train_samples", type=int, default=50000, help="训练数据样本数 (默认: 50000)")
        self.parser.add_argument("--train_steps", type=int, default=20, help="每个训练样本的时间步长 (默认: 16)")
        self.parser.add_argument("--test_samples", type=int, default=2000, help="测试数据样本数 (默认: 2000)")
        self.parser.add_argument("--test_steps", type=int, default=200, help="每个测试样本的时间步长 (默认: 200)")
        self.parser.add_argument(
            "--test_type",
            type=str,
            default="all",
            choices=["sin", "random", "chirp", "all"],
            help="测试数据类型, 可选值: sin / random / chirp / all (默认: all)",
        )
        # 预测参数
        # 训练时会用学到的 Koopman 算子做 pre_length 步"开环滚动预测"再算 loss，
        # 而不是只看单步预测误差，这样才能约束模型的多步预测/长期稳定性。
        self.parser.add_argument("--pre_length", type=int, default=10, help="预测长度 (默认: 5)")

        # 网络设计参数
        # u_z=True 时，控制量 u 会先编码到一个隐空间再参与 Koopman 双线性运算
        # (对应 KoopmanBlinear/InvertKoopmanNetBLinear 里的 u_encoder)，
        # 而不是让原始 u 直接乘状态，双线性项能表达 "控制量如何影响系统响应速度"
        # 这种更复杂的耦合，线性模型(DKUC/IKN)不需要这个开关。
        self.parser.add_argument(
            "--u_z", action="store_true", default=False, help="是否使用u-z关联(Koopman线性/双线性网络）(默认: False)"
        )
        self.parser.add_argument(
            "--is_load", action="store_true", default=False, help="是否加载预训练模型 (默认: False)"
        )
        self.parser.add_argument("--x_dim", type=int, default=8, help="状态维度 (默认: 8)")
        self.parser.add_argument("--u_dim", type=int, default=5, help="控制输入维度 (默认: 5)")

        # 训练参数
        self.parser.add_argument("--lr", type=float, default=1e-3, help="学习率 (默认: 1e-3)")
        self.parser.add_argument("--num_epochs", type=int, default=500, help="训练轮数 (默认: 300)")
        self.parser.add_argument("--batch_size", type=int, default=256, help="训练批次大小 (默认: 128)")
        self.parser.add_argument("--eval_batch_size", type=int, default=256, help="评估批次大小 (默认: 128)")
        self.parser.add_argument(
            "--log_interval", type=int, default=100, help="日志打印间隔（单位：迭代次数）(默认: 100)"
        )
        # 每隔 eval_interval 个 epoch 才跑一次验证集评估、才可能刷新 best_model.pt，
        # 训练日志（train_losses.json）本身仍是每个 epoch 都记录。
        self.parser.add_argument("--eval_interval", type=int, default=10, help="评估间隔(单位: epoch) (默认: 10)")
        self.parser.add_argument(
            "--loss_name",
            type=str,
            default="mse",
            choices=["weighted_mse", "mse", "nmse"],
            help="损失函数类型,可选值:weighted_mse / mse / nmse (默认: mse)",
        )
        # 仅 loss_name="weighted_mse" 时生效：多步预测的每一步误差按 gamma 做指数衰减
        # 加权（越远的未来步权重越小），避免长预测窗口里远期误差主导梯度。
        self.parser.add_argument("--gamma", type=float, default=0.8, help="加权损失系数 (默认: 0.8)")
        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cpu", "cuda"],
            help="计算设备, 可选值: cpu / cuda (默认: cuda)",
        )

        # 控制实验参数
        # delta_mpc: MPC 优化的是"控制量增量"(平滑、抖动小)；
        # mpc: 直接优化绝对控制量。具体用法见 control/MPC_Controler.py。
        self.parser.add_argument(
            "--MPC_type",
            type=str,
            default="delta_mpc",
            choices=["delta_mpc", "mpc"],
            help="MPC类型,可选值: delta_mpc / mpc (默认: mpc)",
        )
        self.parser.add_argument(
            "--traj_name",
            type=str,
            default="Fig8",
            choices=["FigStar", "Fig8", "love"],
            help="轨迹形状, 可选值: FigStar / Fig8 / love",
        )
        # 注意：DBKN_KESO.py / IBKN_UKF.py 这两个主实验脚本的轨迹形状/噪声/UKF/KESO
        # 开关实际是在脚本内部 config_dict 里硬编码的，不读取下面这几个命令行参数
        # （README 里专门提到过这一点，容易踩坑）。
        self.parser.add_argument("--UKF", action="store_true", default=False, help="是否使用无迹卡尔曼滤波")
        self.parser.add_argument("--noise", action="store_true", default=False, help="是否添加噪声")
        self.parser.add_argument("--render", action="store_true", default=True, help="是否渲染")

        # use_stable=True 时 Koopman 转移矩阵 K 用正交分解构造、强制谱半径 < rho_max，
        # 保证多步滚动预测不会数值发散，代价是表达能力受限（见 models/base_model.py）。
        self.parser.add_argument("--use_stable", type=bool, default=False, help="是否使用稳定Koopman")
        # use_decoder=True 时状态解码矩阵 C 也参与训练；默认 False 时 C 是固定的
        # [I, 0] 矩阵（直接截取潜变量的前 x_dim 维当作解码结果，不需要学习）。
        self.parser.add_argument("--use_decoder", type=bool, default=False, help="是否训练解码器")

    # 每个 env 的资产子目录/文件名布局。换机械臂时在这里加一条，不做成通用插件框架。
    ROBOT_ASSET_LAYOUT = {
        "X02CTPoseWriting": {"robot_dir": "x02cTpose", "subdir": "assets", "scene_file": "right_write.xml", "arm_file": "right_write.xml"},
        "X02Writing": {"robot_dir": "X02", "subdir": "assets", "scene_file": "x02_right_write.xml", "arm_file": "x02_right_write.xml"},
        "WheelArmRWriting": {"robot_dir": "WheelArmR", "subdir": "assets", "scene_file": "wheel_arm_r_write.xml", "arm_file": "wheel_arm_r_write.xml"},
        "PIPER": {"subdir": "assets", "scene_file": "piper.xml", "arm_file": "piper.xml"},
        "SOARM101": {"subdir": "SO101", "scene_file": "scene_with_table_v.xml", "arm_file": "so101_new_calib_v.xml"},
        "WheelArmR": {"subdir": "assets", "scene_file": "wheel_arm_r.xml", "arm_file": "wheel_arm_r.xml"},
    }

    def process_args(self):
        """
        把命令行原始参数 (self.args) 展开成代码里实际要用的派生配置：
        资产/数据/结果的文件路径，以及各个 Koopman 网络架构的层维度参数。
        这些值不暴露成命令行参数，是因为它们要么是纯粹的路径拼接，
        要么是网络结构的内部超参（改了要连带训练出的 checkpoint 一起换），
        没必要每次训练都在命令行上敲一遍。
        """
        project_root = os.path.abspath(".")
        """动态处理衍生参数（和原 Tap 类的 process_args 功能一致）"""
        if self.args.env == "PIPER":
            # Piper has six arm joints; SO-101 checkpoints are incompatible.
            self.args.x_dim, self.args.u_dim = 9, 6
        if self.args.env in ("WheelArmRWriting", "X02Writing"):
            self.args.x_dim, self.args.u_dim = 10, 7
        if self.args.env == "X02CTPoseWriting":
            self.args.x_dim, self.args.u_dim = 7, 4
        layout = self.ROBOT_ASSET_LAYOUT[self.args.env]
        self.args.xml_path = os.path.join(project_root, layout.get("robot_dir", self.args.env), layout["subdir"], layout["scene_file"])
        self.args.arm_xml_path = os.path.join(project_root, layout.get("robot_dir", self.args.env), layout["subdir"], layout["arm_file"])
        # 动态生成输出目录：results/<env>/<suffix>/<model>/，
        # 训练日志、best_model.pt、评估结果都落在这个目录下。
        self.args.output_dir = os.path.join(project_root, "results", self.args.env, self.args.suffix, self.args.model)

        # 动态生成数据目录：文件名里编码了样本数/步数，天然起到"缓存 key"的作用，
        # 同一组 (train_samples, train_steps) 多次训练会复用同一份采集数据。
        self.args.data_dir_save = os.path.join(project_root, self.args.env, "data")
        self.args.data_dir_load_train = os.path.join(
            project_root, self.args.env, "data", f"train_data_{self.args.train_samples}_{self.args.train_steps}.npy"
        )
        self.args.data_dir_load_test = os.path.join(
            project_root,
            self.args.env,
            "data",
            f"test_data_{self.args.test_type}_{self.args.test_samples}_{self.args.test_steps}.npy",
        )
        self.args.data_dir_load_val = os.path.join(
            project_root, self.args.env, "data", f"val_data_{self.args.test_samples}_{self.args.test_steps}.npy"
        )

        # 网络层维度配置
        # DKUC/DBKN(KoopmanBase.py) 共用的 encoder 全连接层尺寸：
        # 输入 x_dim 维状态 -> 64 -> 64 -> 64 -> 24 维 Koopman 观测空间。
        self.args.layers = [self.args.x_dim, 64, 64, 64, 24]  # 调整输出维度以适应SOARM101
        """Koopman线性/双线性网络的全连接层维度（输入维度=x_dim)"""

        # 可逆网络参数（供 InvertKoopman.py 的 IKN/IBKN 使用）
        # 可逆网络把 x/u 分别拆成多个"块(block)"，每块内部是若干层耦合层，
        # 通道数(channels)/隐藏层维度(hiddens)逐块给出，块数越多表达能力越强，
        # 但训练/推理开销也越大；因为是可逆结构，这些超参同时决定了编码和
        # 解码两个方向共用的网络容量。
        self.args.x_blocks = [2, 2]
        """可逆网络x分支的块数"""
        self.args.x_channels = [12, 16]
        """可逆网络x分支的通道数"""
        self.args.x_hiddens = [64, 128]
        """可逆网络x分支的隐藏层维度"""
        self.args.u_blocks = [2, 2]
        """可逆网络u分支的块数"""
        self.args.u_channels = [10, 16]
        """可逆网络u分支的通道数"""
        self.args.u_hiddens = [32, 64]
        """可逆网络u分支的隐藏层维度"""

        # transform网络参数（供 Koopform.py 的 Koopformer 使用）
        # seq_len 是 Transformer 编码器一次看到的历史时间步长度，
        # d_model 是内部特征维度，patch_len 目前未被 init_model 使用（预留）。
        self.args.seq_len = 12
        self.args.patch_len = 4
        self.args.d_model = 16
        # KAN网络（供 KANKoopman.py 使用）：kan_layers 定义每层的节点数，
        # kan_params=None 表示用 KAN 实现的默认样条基函数参数。
        self.args.kan_layers = [self.args.x_dim, 32, 16]
        self.args.kan_params = None
        # LSTM网络（供 KoopmanLSTM.py 使用）：lstm_seq_len 是输入历史窗口长度，
        # LSTM_Hidden 是隐状态维度，LSTM_encode_layers 是 LSTM 之后接的编码全连接层。
        self.args.lstm_seq_len = 6
        self.args.LSTM_Hidden = 64
        self.args.LSTM_encode_layers = [self.args.LSTM_Hidden, 64, 64, 24]

    def __getattr__(self, name):
        """方便直接通过 Args 实例访问参数（如 args.model 而非 args.args.model)"""
        return getattr(self.args, name)


# 使用示例（和原代码完全兼容）
if __name__ == "__main__":
    # 解析命令行参数（用法和原 Tap 完全一致）
    args = Args()
    # 打印解析后的参数（验证效果）
    print("=" * 50)
    print("运行参数汇总：")
    print(f"模型: {args.model}")
    print(f"模式: {args.mode}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练数据路径: {args.data_dir_load_train}")
    print(f"学习率: {args.lr}")
    print(f"设备: {args.device}")
    print("=" * 50)
