# SO-ARM101 Koopman Sim2Real

基于 Koopman 算子理论的 SO-ARM101 六自由度机械臂仿真到真机迁移 (Sim2Real) 项目，实现数据驱动的动力学建模、预测与 MPC 控制。

[English](README.md) | 简体中文

## ✨ 特性

- 🤖 支持多种 Koopman 网络架构（DKUC、DBKN、IKN、IBKN、Koopformer、KANKoopman）
- 🎮 基于 MuJoCo 的高保真物理仿真环境
- 🎯 CasADi 实现的 MPC 控制器
- 🔄 ZMQ 实现的 Sim2Real 实时通信
- 📊 完整的训练、评估与可视化流程

## 📁 项目结构

```
├── SOARM101/                 # MuJoCo 仿真环境
│   ├── SOARM101_Env.py       # Gymnasium 风格环境封装
│   ├── SOARM101_DataCollection.py  # 数据采集脚本
│   └── SO101/                # 机器人 URDF/XML 模型
├── models/                   # Koopman 网络实现
│   ├── base_model.py         # KoopmanNet 基类
│   ├── KoopmanBase.py        # DKUC/DBKN 线性/双线性模型
│   ├── InvertKoopman.py      # IKN/IBKN 可逆网络模型
│   └── losses.py             # 损失函数定义
├── control/                  # MPC 控制模块
│   ├── MPC_Controler.py      # MPC 控制器实现
│   ├── TrajectoryGenerator.py # 轨迹生成器
│   └── config.py             # 实验配置管理
├── lerobot_sim2real/         # Sim2Real 通信模块
│   ├── so101_mujoco.py       # MuJoCo 端 ZMQ 发布者
│   └── so101_real.py         # 真机端 ZMQ 订阅者
├── args.py                   # 命令行参数配置
└── train.py                  # 训练/测试入口
```

## 🛠️ 环境搭建

建议使用独立的 conda 环境（`requirements.txt` 里锁定的几个版本已经和现在 PyPI 上的实际依赖对不上了，见下方"踩坑记录"），完整可用的安装顺序如下：

```bash
conda create -n koopman-sim2real python=3.10 -y
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real

# 1. torch 需要从官方 CUDA 索引装（国内镜像通常没有 +cu121 这个变体）
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 2. lerobot 会带出一整套较新的依赖（numpy 2.x / gymnasium 1.x 等），先单独装
pip install lerobot==0.4.4

# 3. 仿真与控制相关的包
pip install casadi==3.7.2 dm_control==1.0.23 matplotlib mujoco==3.2.3 pyzmq tqdm tensorboard

# 4. Pinocchio（正确的 PyPI 包名是 pin，不是 pinocchio）
pip install pin
```

> 显卡驱动只要不太旧（本项目在 RTX 4090 / driver 580.x 上验证过），装 cu121 的 torch 完全没问题，向后兼容。

### 踩坑记录（`requirements.txt` 里的已知问题）

直接 `pip install -r requirements.txt` 目前会失败，原因如下，供参考 / 后续修复：

| 问题 | 现象 | 解决方式 |
|---|---|---|
| `torch==2.4.1+cu121` | 国内 PyPI 镜像没有这个 CUDA 变体 wheel | 改用 `--index-url https://download.pytorch.org/whl/cu121` 单独安装 |
| `pinocchio==0.4.3` | **包名写错**，PyPI 上 `pinocchio` 是个无关的 12KB 小包，不是机器人学的 Pinocchio 库 | 代码里 `import pinocchio as pin` 真正需要的包名是 `pin`，应改为 `pin==...` |
| `rerun==1.0.30` | **同样的包名错误**，PyPI 上 `rerun` 也是个无关的 19KB 小包 | 代码里 `import rerun as rr` 真正需要的是 `rerun-sdk`（lerobot 的依赖会自动带上） |
| `gymnasium==1.0.0` / `numpy==1.24.4` | 与 `lerobot==0.4.4` 实际依赖的版本范围冲突（`gymnasium>=1.1.1`，`opencv-python-headless` 间接要求 `numpy>=2`） | 不要固定版本号，交给 pip 解析（最终会装 `gymnasium>=1.2`、`numpy` 2.x，代码未使用 numpy 2.x 中移除的旧接口，安全） |
| 缺少 `tensorboard` | `train.py` 里 `from torch.utils.tensorboard import SummaryWriter` 找不到模块 | `pip install tensorboard` |
| `control/TrajectoryGenerator.py` 顶部无条件 `from .casadi_ik import Kinematics` | pip 装的 `pin` 包不包含 `pinocchio.casadi` 绑定，即使默认代码路径（`use_pinocchio=False`）根本用不到它，也会在 import 阶段直接崩溃 | 已在本仓库修复：把该 import 挪到 `CartesianTrajectoryGenerator_pinocchio._initialize_ik_solver` 内部懒加载，不影响默认路径 |

## 🚀 快速开始

### 训练模型

```bash
# 训练 DBKN 模型
python train.py --model DBKN --mode train

# 训练可逆 Koopman 网络
python train.py --model IBKN --mode train

# 小规模冒烟测试（验证流程能否跑通，几分钟内出结果）
python train.py --model DBKN --mode train --suffix smoke_test \
  --train_samples 2000 --test_samples 200 --num_epochs 5 --eval_interval 1
```

### 在 MuJoCo 中可视化验证已训练模型

`IBKN_UKF.py` / `DBKN_KESO.py` 会加载 `results/SOARM101/<suffix>/<model>/best_model.pt`，在 MuJoCo 中跑闭环 MPC 轨迹跟踪并弹出可视化窗口。**必须在有图形桌面、可交互输入的终端里运行**（脚本会在弹窗后等待按 Enter 才开始仿真）：

```bash
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real

# DBKN + KESO：Helix 轨迹，默认关闭噪声/负载/KF/ESO（看基础跟踪效果）
python DBKN_KESO.py --model DBKN --suffix 12_22

# IBKN + UKF：FigStar 轨迹，默认开启噪声 + UKF 状态估计（抗干扰效果）
python IBKN_UKF.py --model IBKN --suffix 12_22
```

- `--model` / `--suffix` 决定加载哪个 checkpoint（对应 `results/SOARM101/<suffix>/<model>/best_model.pt`）
- 轨迹形状（Helix/FigStar/…）、是否加噪声/负载/UKF/ESO 是在脚本 `if __name__ == "__main__":` 里的 `config_dict` 中硬编码的，**不是**通过 `--traj_name`/`--noise`/`--UKF` 这些命令行参数控制（这几个参数在 `args.py` 里存在但这两个脚本没有读取它们）
- 窗口弹出后按提示按 **Enter** 开始仿真；轨迹播放完毕会自动回到 Home 姿态并把结果存到 `control/ControlResults/<suffix>/<traj_name>/<方法标记>.npz`
- 关闭 MuJoCo 窗口即结束仿真

## 🧪 主要实验

本项目重点验证了两种抗干扰控制架构：

### 1. IBKN + UKF (基于可逆 Koopman 的 UKF 状态估计)

结合 **可逆双线性 Koopman 网络 (IBKN)** 与 **无迹卡尔曼滤波 (UKF)**，解决模型不确定性与观测噪声问题。

- **运行命令**:

```bash
  # 需手动进行对照实验 
  python IBKN_UKF.py
```

- **实验结果**: ![控制结果对比](control/FigResults/12_11/noise_robustness.png)

### 2. DBKN + KESO (基于深度双线性 Koopman 的扩张状态观测器)

利用 **深度双线性 Koopman (DBKN)** 结合 **Koopman 扩张状态观测器 (KESO)**，对未建模扰动进行实时估计与补偿。

- **运行命令**:

```bash
  # 需手动进行对照实验
  python DBKN_KESO.py
```

- **抗噪声实验结果**: ![控制结果对比](control/FigResults/12_22/noise_robustness_3rows.png)
- **抗负载实验结果**: ![控制结果对比](control/FigResults/12_22/payload_robustness_3rows.png)
- **抗噪+抗负载实验结果**: ![控制结果对比](control/FigResults/12_22/both_robustness_3rows.png)

## 🧪 Sim2Real 部署

![实验一](control/media/Video1.gif)
![实验二](control/media/Video2.gif)
