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
# 基本格式：--model 可选 DKUC / DBKN / IKN / IBKN / all
python train.py --model <DKUC|DBKN|IKN|IBKN|all> --mode train --suffix <自定义名字>

# 训练 DBKN 模型
python train.py --model DBKN --mode train

# 训练可逆 Koopman 网络
python train.py --model IBKN --mode train

# 一次训练全部 4 种架构
python train.py --model all --mode train --suffix <自定义名字>

# 小规模冒烟测试（验证流程能否跑通，几分钟内出结果）
python train.py --model DBKN --mode train --suffix smoke_test \
  --train_samples 2000 --test_samples 200 --num_epochs 5 --eval_interval 1
```

常用超参（见 `args.py`）：`--train_samples`（默认 50000）、`--num_epochs`（默认 500）、`--lr`（默认 1e-3）、`--batch_size`（默认 256）。训练数据按 `train_data_<train_samples>_<train_steps>.npy` 缓存在 `SOARM101/data/`，同样的样本数/步数会自动复用、不重新采集；如果改了 `SOARM101_Env.py` 里的 `perturbation_limits`（关节扰动范围）想让新范围生效，要么换一组新的 `--train_samples`/`--train_steps`，要么先删掉 `SOARM101/data/` 里的旧缓存。

评估已训练模型（离线算预测误差和曲线图，不开 MuJoCo 窗口）：

```bash
python train.py --model IBKN --mode test --suffix <训练时用的suffix>
# 结果存到 results/SOARM101/<suffix>/IBKN/test_<random|sin|chirp>/
```

### 📈 监控训练进度

这是监督学习，不是强化学习——看的是**预测损失（loss）**，训练目标是让它**下降**，不是"奖励上升"。两种方式看实时进度：

```bash
# 方式一：TensorBoard（推荐，能看曲线），训练过程中随时可以起
tensorboard --logdir results/SOARM101/<suffix>
# 浏览器打开 http://localhost:6006

# 方式二：直接看 JSON（每次验证误差刷新最优值时才会写，间隔 --eval_interval 个 epoch）
cat results/SOARM101/<suffix>/<model>/best_scores.json
```

`train_losses.json`（完整训练曲线历史）只有在训练**全部结束**时才会写入，训练过程中看不到，要看实时曲线用 TensorBoard。

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

想改轨迹形状/平面朝向/干扰开关，需要直接编辑脚本里的这几处：

| 想改什么 | 改哪个文件/哪行 | 可选值 |
|---|---|---|
| 轨迹形状 | `IBKN_UKF.py` / `DBKN_KESO.py` 里 `config_dict` 的 `"traj_name"` | `Fig8` `FigStar` `Heart` `Rectangle` `Lissajous`（平面图形）、`Helix`（唯一的 3D 螺旋轨迹） |
| 平面朝向 | `CartesianTrajectoryGenerator(...)` 调用处的 `idx=` 参数 | `0` = X-Y 平面（水平贴桌面）、`1` = Y-Z 平面（竖直） |
| 是否加噪声 | `config_dict` 的 `"use_nosie"` | `True`/`False` |
| 是否开 UKF（仅 `IBKN_UKF.py`） | `config_dict` 的 `"use_KEM"` | `True`/`False` |
| 是否开 KESO（仅 `DBKN_KESO.py`） | `config_dict` 的 `"use_eso"` | `True`/`False` |
| 是否加负载（仅 `DBKN_KESO.py`） | `config_dict` 的 `"use_payload"` | `0`/`1`/`2` |

### 让 X-Y 桌面平面轨迹跟踪准确（复现 Sim2Real 部署视频里的效果）

`results/SOARM101/12_11`、`12_22` 这两批 checkpoint 是用 `SOARM101_Env.py` 里默认的 `perturbation_limits = ±0.3 rad`（零位附近的小范围随机扰动）采集数据训练出来的。如果直接把 `idx` 改成 `0` 去画 `Fig8`/`Heart` 这类贴桌面的图案，**跟踪效果会明显变差**（末端画出来的轨迹和参考轨迹对不上），原因不是 bug，是这些桌面轨迹需要的关节角范围远超训练数据覆盖的范围，例如：

| 关节 | 训练覆盖范围（默认） | Fig8(X-Y) 实际需要 | Heart(X-Y) 实际需要 |
|---|---|---|---|
| shoulder_lift | ±0.3 rad | -1.26 ~ -0.11 rad | -0.36 ~ 0.57 rad |
| elbow_flex | ±0.3 rad | 0.22 ~ 0.75 rad | -0.03 ~ 0.70 rad |

想让模型在桌面平面也能画准，需要扩大扰动范围、重新采集数据、重新训练：

```python
# SOARM101/SOARM101_Env.py 里把 perturbation_limits 改宽，覆盖全部 5 种平面轨迹所需范围
self.perturbation_limits = np.array([
    [-0.45, -1.35, -1.25, -0.75, -0.3],   # 下限
    [ 0.45,  1.45,  0.85,  0.85,  0.3],   # 上限
])
```

```bash
# 换一个新 suffix 触发重新采集数据 + 训练（数据会被两次训练共用，第二次更快）
python train.py --model IBKN --mode train --suffix wide_range
python train.py --model DBKN --mode train --suffix wide_range

# 训练完之后验证：把脚本里的 idx 改成 0、traj_name 改成 Fig8/Heart，再用新 suffix 跑
python IBKN_UKF.py --model IBKN --suffix wide_range
python DBKN_KESO.py --model DBKN --suffix wide_range
```

改动前记得检查物理关节限位（`SOARM101/SO101/so101_new_calib_v.xml` 里每个 `<joint>` 的 `range` 属性），新的扰动范围要留在限位以内。

## ✍️ 写汉字（多笔画：抬笔 -> 平移 -> 落笔）

### 功能说明

`control/TrajectoryGenerator.py` 的 `CartesianTrajectoryGenerator.generate_multi_stroke()` 支持"多笔画"轨迹——不同于 `generate()` 那种单笔连续画的 Fig8/Heart，这个方法把每一笔画单独贴桌面书写，笔画之间自动插入"抬笔（升高）→ 平移（在抬笔高度移动）→ 落笔（降低）"，输出的还是一条完整的 `(N,3)` 笛卡尔点序列，交给现有的 IK + MPC 流程，不用改控制那一层。

`control/hanzi_loader.py` 把 [hanzi-writer-data](https://github.com/chanind/hanzi-writer-data)（makemeahanzi 笔画数据，本机路径 `/home/cym/ROS/Arms/isaac_so_arm101/hanzi-writer-data/data/`，9574 个汉字）的笔画中线（`medians`）转换成 `generate_multi_stroke()` 需要的格式：

```python
from control.hanzi_loader import hanzi_to_strokes, text_to_strokes

strokes = hanzi_to_strokes("永", size_m=0.12)       # 单字
strokes = text_to_strokes("你好", size_m=0.10)       # 一整串字横向排开 (范围会成倍增大，建议先单字验证)
```

`write_demo.py` 是现成的演示脚本，复用 `DBKN_KESO.py` 里的 `Test` 类（MuJoCo 仿真 + MPC 播放器）：

```bash
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real
python write_demo.py --model DBKN --suffix wide_range
```

改 `write_demo.py` 顶部的 `CHAR = "二"` 换字。里面还有个 `check_reachability()`，会在真正开始仿真前粗查这个字需要的关节角度是否超出 `wide_range` 训练覆盖的范围（超出只警告不中止）。

`write_height`（落笔书写高度）/`lift_height`（抬笔转移高度）在 `generate_multi_stroke()` 调用处设置，单位米，参照的是世界坐标系——这个模型里桌面大致在 `z≈0`，所以 `write_height=0.055` 就是离桌面 5.5cm；如果末端夹了实体笔，这个高度要减去笔尖比夹爪中心低出的距离。两处调用（预览一遍 + 正式生成一遍，见下文 `fixed_wrist_roll_rad`）的值要保持一致。

### 固定夹爪朝向 (wrist_roll)

默认 `target_orientation=None`（IK 不约束姿态），`wrist_roll` 会在每个点自由变化。如果想让夹爪扭转角在整条轨迹里锁死在某个固定值（比如相对轨迹起点的自然朝向再转 90°），用 `fixed_wrist_roll_rad` 参数——这不是简单把结果数组的那一列数值覆盖掉（会导致末端 XYZ 位置和记录的不一致），而是求解 IK 时就把 `wrist_roll` 从可解关节里剔除、只用另外 4 个关节去够位置，运动学上是一致的：

```python
# 先自由求解一遍，拿到轨迹起点的自然 wrist_roll
_, preview_joint_traj, _ = traj_generator.generate_multi_stroke(strokes, center=(0.35, 0.0))
fixed_wrist_roll_rad = preview_joint_traj[0, 4] - np.radians(90)   # 起点基础上转 -90°

# 再生成一遍，这次锁死 wrist_roll
cartesian_points, joint_angle_traj, pen_down_mask = traj_generator.generate_multi_stroke(
    strokes, center=(0.35, 0.0), fixed_wrist_roll_rad=fixed_wrist_roll_rad,
)
```

> 注意：锁定的角度如果离训练时这个关节覆盖的范围（`wide_range` 训练时 `wrist_roll` 只覆盖了 ±0.3rad≈±17°）太远，`check_reachability()` 会警告——末端画字的形状受影响不大（XYZ 主要靠另外 4 个关节），但夹爪朝向本身的跟踪精度没保证，机械臂可能会在这个关节上晃/抖。回归 Home 阶段不受这个参数影响，仍然按 Home 关键帧的 `wrist_roll=0` 走（这段是从轨迹终点线性插值回 Home，跟 IK 无关）。

## 🤖 部署到真实机械臂

这一步依赖同一工作区下的另一个项目 [`so-arm101-ros2-bridge`](../so-arm101-ros2-bridge/)，它已经把 `Lerobot-mujoco-sim2real/lerobot_sim2real/so101_real.py` 改造成了双向 ZMQ 桥接（详见该项目的 `README_CN.md`）：

```
koopman-sim2real (python3.10)          lerobot-env (python3.12, 装了 lerobot==0.6.2 + feetech-servo-sdk)
──────────────────────────             ──────────────────────────────────────────────────
write_demo.py / DBKN_KESO.py 等   ──ZMQ PUB :5555 (目标关节角度,度)──►  so101_real.py ──► 真实机械臂
(MuJoCo 仿真 + Koopman + MPC)     ◄──ZMQ PUB :5556 (当前关节角度,度)──  (SUB :5555, PUB :5556)
```

两个环境分开是因为 ROS2 Humble 自带的 `rclpy` 编译扩展要求 Python 3.10，而支持 `so101_follower` 的 LeRobot 版本需要 Python 3.12，没法在同一进程共存——这也是为什么真机驱动 (`so101_real.py`) 必须在单独的 `lerobot-env` 里跑，而不是我们训练/仿真用的 `koopman-sim2real`。

### 运行

```bash
# 终端1 —— 真机驱动 (独占串口)
conda activate lerobot-env
cd Lerobot-mujoco-sim2real/lerobot_sim2real
python so101_real.py

# 终端2 —— 仿真+MPC+写字, 会自动通过 ZMQ 把指令发给真机
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real
python write_demo.py --model DBKN --suffix wide_range
```

`write_demo.py`/`DBKN_KESO.py`/`IBKN_UKF.py` 正式开始书写/画图前，会先调用 `sync_real_to_sim_start()`：订阅 `so101_real.py` 在 `:5556` 广播的真机当前姿态，插值 5 秒钟平滑移动到轨迹起点，而不是让真机在指令流开始后自己"盲追"——这样可以在真正开始前用肉眼确认真机和仿真已经对齐。**这一步需要 `so101_real.py` 已经在运行**，否则会在 2 秒超时后报错退出（不会瞎跑）。

### 标定 joint_offsets（必须先做，否则角度会算错）

真机指令角度 = 仿真角度 − `joint_offsets`（`DBKN_KESO.py`/`IBKN_UKF.py` 顶部的 `sim_to_real()`），这个偏移量是仿真模型的关节零点和真机标定零点之间的差异，**每次重新 `lerobot-calibrate` 之后都要重新标定这个值**，否则真机会往错误的角度走。

1. 先用 `lerobot-calibrate` 校准真机（这是 `so-arm101-ros2-bridge` 项目里的标准步骤，不是本项目自带的），得到 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`。
   - **重新标定**：同一条命令再跑一次，出现提示时输入 `c` 回车（直接回车只会复用旧文件，不会真正重新标定）。
   - **只想读当前姿态、不想重新标定**：用 `read_current_pose.py`（见下），不要跑 `lerobot-calibrate`。
2. 开一个 MuJoCo 实时窗口，把仿真定在 `home` 关键帧姿态作为参照：
   ```bash
   conda activate koopman-sim2real
   cd Lerobot-mujoco-sim2real
   python show_home_pose.py
   ```
   窗口里能自己转视角/缩放，终端会打印目标角度。
3. 对照窗口，把真机手动摆成同样的姿态（可能需要先松开/断电舵机扭矩才能自由掰动），摆好后读真机当前角度（只读，不会动，也不会改标定文件）：
   ```bash
   conda activate lerobot-env
   cd Lerobot-mujoco-sim2real/lerobot_sim2real
   python read_current_pose.py
   ```
4. 对每个关节算 `offset = 仿真目标角度 - 真机读数`，填进 `DBKN_KESO.py`/`IBKN_UKF.py` 顶部的 `joint_offsets`（6 个值，顺序 `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`，`gripper` 这一项 MPC 不控制，无所谓）。

**踩坑记录：**

| 现象 | 原因 / 应对 |
|---|---|
| 算出来的偏移量每次都不太一样 | 人工对照姿态本身有误差，尤其 `wrist_roll` 这种扭转关节肉眼很难判断对不对；多摆几次、比较是否收敛，`shoulder_pan`/`shoulder_lift`/`elbow_flex` 这种大臂关节容易摆准，手腕两个关节需要更仔细 |
| 不确定标定文件准不准，想重新标定一个"更干净的零点" | 可以，重新走一遍 `lerobot-calibrate` 流程；但**之前算的 `joint_offsets` 会失效**，需要按上面步骤重新走一遍 Home 姿态对齐 |
| 反复对齐后残差都很小 (几度以内) | 可以考虑直接把 `joint_offsets` 全设成 0，信任 `lerobot-calibrate` 本身给出的零点足够干净——`real_robot.py` 里 `max_relative_target=10.0`（每条指令最多让真机动 10°）的安全网能兜住几度量级的残余误差 |
| `python read_current_pose.py` 报 `could not open port /dev/ttyACM0` | USB 连接断开或机械臂断电；重新插好/上电，`ls /dev/ttyACM0` 确认设备回来后再试 |
| 想用 `lerobot-calibrate` 单纯"获取位置信息" | 不对，`lerobot-calibrate` 是重新标定（会要求转动关节到运动范围两端），不是读取。只读用 `read_current_pose.py` |

### 安全须知

- `real_robot.py`（属于 `so-arm101-ros2-bridge` 那条链路，路径在 `lerobot_sim2real/real_robot.py`）已经设置了 `max_relative_target=10.0`（度）——每条指令实际执行时最多让真机相对**当前位置**移动 10°，这是硬件层面的最后一道安全网，但不代表可以不用看着它动。
- 这套控制是**单向开环**的：MPC/Koopman 完全基于仿真自己的状态计算指令，不读真机的实际反馈。真机只是被动"追"仿真发过来的目标角度。
- 第一次上真机测试任何新轨迹（新字/新高度/新的 `fixed_wrist_roll_rad` 设置）之前：确认机械臂周围、桌面上没有障碍物，手放在旁边，全程盯着看，尤其是关节角度范围检查（`check_reachability`）报过警告的那些关节，有异常随时 Ctrl+C。

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
