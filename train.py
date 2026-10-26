"""
Koopman 动力学模型的训练/测试主入口。

这是一次纯监督学习：用 SOARM101DataGenerator（或 WheelArmR 对应版本）采集
的 (x_t, u_t, x_{t+1}, ...) 轨迹数据，训练目标是让模型多步滚动预测的状态
和真实物理仿真状态之间的误差（loss）下降，和"奖励"无关。

顶层调用关系：
    python train.py --model <DKUC|DBKN|IKN|IBKN|...> --mode <train|test>
        -> main(args)
            -> DataGenerator(args).generate_and_save_data()   # 采集/加载数据
            -> init_model(args)                                 # 按 --model 建网络
            -> train(...) 或 evaluate(...)                       # 训练循环 / 测试评估

不同模型架构（Koopformer/KoopmanLSTM 系列 vs. 其余线性/双线性/可逆网络）用的
损失函数实现不同（models/losses.py 里的 koopformer_loss vs. k_linear_loss），
所以 train()/evaluate() 内部都要按 `type(model).__name__` 做一次分支。
"""

import os
import random
import shutil
import time
from copy import copy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from args import Args
from models.base_model import KoopmanNet
from models.init_model import init_model
from models.losses import k_linear_loss, koopformer_eval_loss_new, koopformer_loss, pred_and_eval_loss_new
from utility import dump_json, plot_contour, plot_predictions


def set_seed(seed: int = 42) -> None:
    """
    为所有可能引入随机性的库设置一个固定的随机种子，以确保实验的可复现性。

    Args:
        seed (int): 要设置的随机种子值。
    """
    # 1. Python 内置的 random 模块
    random.seed(seed)
    # 2. NumPy
    np.random.seed(seed)
    # 3. PyTorch
    # 为所有GPU和CPU设置种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 为当前GPU设置种子
        torch.cuda.manual_seed(seed)


def evaluate(model: KoopmanNet, test_loader, scaler):
    """
    在给定的 test_loader（可以是验证集，也可以是 random/sin/chirp 测试集）上
    跑一遍模型的多步滚动预测，汇总预测误差、笛卡尔位置误差(dis_loss)和
    关节角误差(angle_loss)，并把所有预测/真实值拼接起来供后续画图/保存。

    模型处于 eval 模式 + no_grad，只做前向推理不更新参数。
    """
    scores = []
    dis_scores = []
    angle_scores = []
    all_preds = []
    all_labels = dict(x=[], u=[])

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            # Koopformer/KoopmanLSTM 系列网络需要整段历史序列做编码，
            # 用专门的 koopformer_eval_loss_new 计算预测误差；
            # 其余线性/双线性/可逆网络逐步滚动预测，用 pred_and_eval_loss_new。
            if type(model).__name__.startswith("Koopformer") or type(model).__name__.startswith("KoopmanLSTM"):
                pred_and_error = koopformer_eval_loss_new(
                    batch_data=batch,
                    net=model,
                )
            else:
                pred_and_error = pred_and_eval_loss_new(
                    batch_data=batch,
                    net=model,
                )

            pred = pred_and_error["pred"]
            error = pred_and_error["pred_loss"]
            dis_loss = pred_and_error["dis_loss"]
            angle_loss = pred_and_error["angle_loss"]

            scores.append(error.unsqueeze(0).cpu().detach())
            dis_scores.append(dis_loss.unsqueeze(0).cpu().detach())
            angle_scores.append(angle_loss.unsqueeze(0).cpu().detach())

            all_preds.append(pred.cpu().detach())
            for key in all_labels.keys():
                all_labels[key].append(batch[key].cpu().detach())

        scores = torch.cat(scores, dim=0).numpy()
        avg_error = np.mean(scores)
        dis_error = np.mean(torch.cat(dis_scores, dim=0).numpy())
        angle_error = np.mean(torch.cat(angle_scores, dim=0).numpy())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        for key in all_labels.keys():
            all_labels[key] = torch.cat(all_labels[key], dim=0).numpy()

    return dict(
        avg_error=avg_error,  # 整个测试集的平均预测误差 1
        dis_error=dis_error,
        angle_error=angle_error,
        scores=scores,  # 每个批次的误差值（数组） (16,)
        all_preds=all_preds,  # 所有预测结果  (2000,200,10)
        all_labels_x=all_labels["x"],  # 所有状态 x 的真实值 (2000,200,10)
        all_labels_u=all_labels["u"],  # 所有控制输入 u 的真实值 (2000,200,7)
    )


def train(
    model,  # 要训练的KoopmanNet模型实例
    train_loader,  # 训练数据集
    test_loader,  # 测试数据集
    output_dir: str,  # 输出目录路径
    num_epochs: int,  # 训练总轮数
    lr: float,  # 学习率
    batch_size: int,  # 训练批次大小
    eval_batch_size: int,  # 评估批次大小
    log_interval: int,  # 日志记录间隔(步数)
    eval_interval: int,  # 评估间隔(轮数)
    x_dim: int,  # 状态变量维度
    u_dim: int,  # 控制输入维度
    loss_name: str,  # 使用的损失函数名称
    gamma: float,  # 损失函数中的权重参数
    device: str,  # 计算设备('cpu'或'cuda'
    pre_length,
    scaler,
):
    """
    标准的 epoch 训练循环：每个 batch 算多步预测损失、反传、更新参数；
    每 eval_interval 个 epoch 在验证集上跑一次 evaluate()，只要预测误差
    刷新历史最优就保存 checkpoint(best_model.pt) 和可视化图/指标 json，
    这样训练中途中断也能拿到当前最好的模型，不需要等训练完全结束。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 指数学习率衰减：每个 epoch 后 lr 乘以 0.99，训练后期用更小步长精调。
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start_time = time.time()
    global_step = 0

    # 初始化损失记录字典（贯穿整个训练过程，每个 epoch 结束后追加一个平均值，
    # 训练结束后整体存成 train_losses.json，用于事后画完整的损失曲线）。
    all_train_losses = dict(
        total_loss=[],  # 总损失
        koopman_loss=[],  # Koopman算子损失
        pred_loss=[],  # 预测损失
        recon_loss=[],  # 重构损失
    )

    writer = SummaryWriter(log_dir=output_dir)
    best_loss = 1000.0
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        # 初始化当前epoch的损失记录（按 batch 累积，epoch 结束时取平均）
        epoch_train_losses = dict(
            total_loss=[],
            koopman_loss=[],
            pred_loss=[],
            recon_loss=[],
            dis_loss=[],
            angle_loss=[],
            stable_Loss=[],
            H_Loss=[],
        )
        for step, batch in enumerate(train_loader):
            model.train()
            # 同 evaluate()：不同网络家族用不同的损失函数实现，
            # 因为 Koopformer/LSTM 需要先对历史窗口做序列编码，
            # 而线性/双线性/可逆网络是逐步做 "编码->Koopman演化->解码"。
            if type(model).__name__.startswith("Koopformer") or type(model).__name__.startswith("KoopmanLSTM"):
                losses = koopformer_loss(
                    epoch,
                    batch_data=batch,
                    net=model,
                    loss_name=loss_name,
                    gamma=gamma,
                    pre_length=pre_length,
                    device=device,
                )
            else:
                losses = k_linear_loss(
                    epoch,
                    batch_data=batch,
                    net=model,
                    loss_name=loss_name,
                    gamma=gamma,
                    pre_length=pre_length,
                    device=device,
                )
            loss = losses["total_loss"]

            # Backward
            loss.backward()
            # === 【修改 3】 梯度裁剪 ===
            # 这能防止某一个 batch 数据异常导致更新步长太大把模型踢飞
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            #  记录损失值
            for key in losses.keys():
                # 将损失值从张量转换为标量并添加到列表
                epoch_train_losses[key] = epoch_train_losses[key] + [losses[key].item()]
            global_step += 1

            # 达到日志记录间隔时打印日志
            if global_step % log_interval == 0 or step == len(train_loader) - 1:
                # 初始化平均损失字典
                avg_losses = dict(
                    total_loss=[],
                    koopman_loss=[],
                    pred_loss=[],
                    recon_loss=[],
                    dis_loss=[],
                    angle_loss=[],
                    stable_Loss=[],
                    H_Loss=[],
                )
                # 计算各项损失的平均值（分母加 1e-5 避免除零，
                # 某些损失项在特定模型/配置下可能全程为 0，如 recon_loss 在
                # use_decoder=False 时不参与训练）
                for key in avg_losses.keys():
                    avg_losses[key] = sum(epoch_train_losses[key]) / (len(epoch_train_losses[key]) + 1e-5)
                # 构建日志信息
                log_info = {
                    "epoch": epoch,
                    "step": step,
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                    "total_loss": f"{avg_losses['total_loss']:.3e}",
                    "koopman_loss": f"{avg_losses['koopman_loss']:.3e}",
                    "pred_loss": f"{avg_losses['pred_loss']:.3e}",
                    "dis_loss": f"{avg_losses['dis_loss']:.3e}",
                    "angle_loss": f"{avg_losses['angle_loss']:.3e}",
                    "recon_loss": f"{avg_losses['recon_loss']:.3e}",
                    "stable_Loss": f"{avg_losses['stable_Loss']:.3e}",
                    "H_Loss": f"{avg_losses['H_Loss']:.3e}",
                    "time": round(time.time() - start_time),
                }
                print(log_info)

        # 每个 epoch 结束后，把这个 epoch 最后一次打印的日志值写入 TensorBoard
        # （不是逐 step 记录，是取本 epoch 内最后一个 log_interval 窗口的均值）。
        writer.add_scalar("Train/total_loss", log_info["total_loss"], epoch)
        writer.add_scalar("Train/koopman_loss", log_info["koopman_loss"], epoch)
        writer.add_scalar("Train/pred_loss", log_info["pred_loss"], epoch)
        writer.add_scalar("Train/recon_loss", log_info["recon_loss"], epoch)
        writer.add_scalar("Train/stable_Loss", log_info["stable_Loss"], epoch)
        writer.add_scalar("Train/H_Loss", log_info["H_Loss"], epoch)
        # 达到评估间隔时进行评估
        if (epoch + 1) % eval_interval == 0:
            # 在测试集上评估模型
            eval_info = evaluate(model=model, test_loader=test_loader, scaler=scaler)
            print(f"Prediction error: {eval_info['avg_error']}")
            writer.add_scalar("Eval/total_loss", eval_info["avg_error"], epoch)

            # 只有验证误差刷新历史最优时才落盘，best_model.pt 始终对应
            # "迄今为止验证集上表现最好的一次" checkpoint，而不是最后一个 epoch。
            if eval_info["avg_error"] < best_loss:
                best_loss = copy(eval_info["avg_error"])
                # 创建检查点目录
                ckpt_dir = output_dir + "/ckpt"
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                # 保存模型检查点
                ckpt_path = ckpt_dir + f"/model-{epoch}.pt"
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(model.state_dict(), ckpt_path)
                torch.save(model.state_dict(), output_dir + "/best_model.pt")
                # 根据状态维度选择可视化方式：低维状态(如本项目 x_dim=8)
                # 直接画每一维的时间序列曲线更直观；高维状态画等高线图更紧凑。
                if x_dim < 20:
                    # 低维数据使用标准预测图
                    plot_predictions(
                        all_preds=eval_info["all_preds"],  # 所有预测结果
                        all_labels_x=eval_info["all_labels_x"],  # 真实状态标签
                        all_labels_u=eval_info["all_labels_u"],  # 真实控制标签
                        plot_idx=np.random.randint(0, len(eval_info["all_preds"]), size=8),
                        # 随机选8个样本
                        save_dir=ckpt_dir,  # 保存目录
                        epoch=epoch,
                    )
                else:
                    # 高维数据使用等高线图
                    plot_contour(
                        all_preds=eval_info["all_preds"],
                        all_labels_x=eval_info["all_labels_x"],
                        all_labels_u=eval_info["all_labels_u"],
                        plot_idx=np.random.randint(0, len(eval_info["all_preds"]), size=8),
                        save_dir=ckpt_dir,
                        epoch=epoch,
                    )

                # 保存评估指标（best_scores.json 只在验证误差刷新时更新，
                # 是 README 里提到的"看训练是否收敛"的权威数据源之一）
                epoch_scores = dict(
                    epoch=epoch,
                    train_total_loss=float(
                        sum(epoch_train_losses["total_loss"]) / (len(epoch_train_losses["total_loss"]) + 1e-5)
                    ),
                    koopman_loss=float(
                        sum(epoch_train_losses["koopman_loss"]) / (len(epoch_train_losses["total_loss"]) + 1e-5)
                    ),
                    pred_loss=float(
                        sum(epoch_train_losses["pred_loss"]) / (len(epoch_train_losses["pred_loss"]) + 1e-5)
                    ),
                    recon_loss=float(
                        sum(epoch_train_losses["recon_loss"]) / (len(epoch_train_losses["recon_loss"]) + 1e-5)
                    ),
                    evaluate_loss=float(eval_info["avg_error"]),
                    stable_Loss=float(
                        sum(epoch_train_losses["stable_Loss"]) / (len(epoch_train_losses["stable_Loss"]) + 1e-5)
                    ),
                    H_Loss=float(sum(epoch_train_losses["H_Loss"]) / (len(epoch_train_losses["H_Loss"]) + 1e-5)),
                    time=time.time() - epoch_start_time,
                )
                dump_json(epoch_scores, ckpt_dir + f"/scores-{epoch}.json")
                dump_json(epoch_scores, output_dir + "/best_scores.json")

        scheduler.step()
        for key in all_train_losses.keys():
            all_train_losses[key] = all_train_losses[key] + [
                (float(sum(epoch_train_losses[key]) / (len(epoch_train_losses[key]) + 1e-5)))
            ]

    # train_losses.json 只在全部 epoch 跑完后一次性写入，训练过程中途查看不到；
    # 想看实时曲线只能用上面已经在写的 TensorBoard。
    all_train_losses["time"] = time.time() - start_time
    dump_json(all_train_losses, output_dir + "/train_losses.json")


def main(args):
    """单个模型(args.model)在单个 env(args.env) 上完整跑一次 train 或 test 流程。"""
    # Loading dataset
    # set_seed(args.seed)
    # 不同机器人(env)用各自的数据采集器，但接口(generate_and_save_data/
    # get_train_loader/get_test_loader)完全一致，train()/evaluate() 不需要
    # 关心具体是哪个机器人。
    # 书写机器人（robots/*.yaml）由注册表给出采集器；其余是两个非书写的旧环境。
    writing_robot = args.robot_spec is not None
    if writing_robot:
        DataGenerator = args.robot_spec.data_generator()
    elif args.env == "WheelArmR":
        from WheelArmR.WheelArmR_DataCollection import WheelArmRDataGenerator as DataGenerator
    else:
        from SOARM101.SOARM101_DataCollection import SOARM101DataGenerator as DataGenerator
    data_generate = DataGenerator(args)
    # Model
    if writing_robot:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    model = init_model(args)
    if writing_robot and args.mode == "train":
        from PIPER.data_collection import initialize_piper_model
        initialize_piper_model(model, data_generate.env, data_generate.reference)
    print(model)
    scaler = None
    # 采集/加载数据这一步对 train 和 test 都要做：train 模式下用它拿 train/val
    # loader，test 模式下用它拿 random/sin/chirp 测试集。
    data_generate.generate_and_save_data()
    if args.mode == "train":
        train_loader, val_loader = data_generate.get_train_loader()
        print("==== Saving args ====")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # 把这次训练用的完整 args.py 拷贝进输出目录，方便事后追溯"这个
        # checkpoint 到底是用什么超参训出来的"，比只记录部分参数更保险。
        shutil.copy("./args.py", args.output_dir + "/train_args.py")
        if writing_robot:
            shutil.copy(args.robot_spec.source, args.output_dir + "/robot.yaml")
        train(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            loss_name=args.loss_name,
            gamma=args.gamma,
            device=args.device,
            pre_length=args.pre_length,
            scaler=scaler,
        )

    elif args.mode == "test":
        # test 模式要求之前已经训练过、有 best_model.pt 存在，
        # 只做离线评估 + 画图，不打开 MuJoCo 可视化窗口。
        load_model_path = args.output_dir + "/best_model.pt"
        print(f"Loading the checkpoint from {load_model_path}")
        if not os.path.exists(load_model_path):
            raise FileNotFoundError("No checkpoint!")
        model.load_state_dict(torch.load(load_model_path))

        test_types = ["random", "sin", "chirp"] if args.test_type == "all" else [args.test_type]
        for test_type in test_types:
            test_loader = data_generate.get_test_loader(test_type)

            eval_info = evaluate(model=model, test_loader=test_loader, scaler=scaler)
            print(f"Prediction error: {eval_info['avg_error']}")
            # Save checkpoint
            save_path = args.output_dir + f"/test_{test_type}"
            plot_predictions(
                all_preds=eval_info["all_preds"],
                all_labels_x=eval_info["all_labels_x"],
                all_labels_u=eval_info["all_labels_u"],
                plot_idx=np.random.randint(0, len(eval_info["all_preds"]), size=8),
                save_dir=save_path,
            )
            if args.x_dim > 5:
                plot_contour(
                    all_preds=eval_info["all_preds"],
                    all_labels_x=eval_info["all_labels_x"],
                    all_labels_u=eval_info["all_labels_u"],
                    plot_idx=np.random.randint(0, len(eval_info["all_preds"]), size=8),
                    save_dir=save_path,
                )
            save_info = dict(
                avg_error=float(eval_info["avg_error"]),
                dis_error=float(eval_info["dis_error"]),
                angle_error=float(eval_info["angle_error"]),
            )
            dump_json(save_info, save_path + "/scores.json")
            np.save(save_path + "/all_preds.npy", eval_info["all_preds"])
            np.save(save_path + "/all_labels_x.npy", eval_info["all_labels_x"])
            np.save(save_path + "/all_labels_u.npy", eval_info["all_labels_u"])


if __name__ == "__main__":
    args = Args()
    if args.model == "all":
        # --model all：依次训练/测试这 4 种主力架构，方便一次性做横向对比
        # （更多变体架构在下面被注释掉，默认不参与"all"批量跑，避免耗时过长）。
        methods = [
            "IBKN",
            "DBKN",
            "DKUC",
            "IKN",
        ]
        # methods = ["IKN","IBKN",
        #            "Koopformer","KANKoopman",
        #            "Koopformer_KAN","KoopmanLSTMlinear","KoopmanLSTMlinear_KAN",
        #         ]
        if args.mode == "test":
            # 批量测试用 CPU 就够快，且能避免多个模型反复抢占同一块 GPU 显存。
            args.device = "cpu"
        for method in methods:
            args.model = method
            args.output_dir = "./results/" + args.env + "/" + args.suffix + "/" + args.model
            main(args)

    else:
        main(args)
