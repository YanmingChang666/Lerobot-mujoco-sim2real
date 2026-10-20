# Piper 汉字书写：MuJoCo → Koopman DBKN → MPC

[全部机械臂使用教程与命令速查](../docs/机械臂书写使用教程.md)


这是 **标准 Piper 六关节机械臂** 的独立仿真流程。资产来自本机
`/home/cym/ROS/Arms/agx_arm_sim/agx_arm_description/agx_arm_urdf/piper`。
未使用 Piper-L/H/X；没有连接真机、CAN 或 SO-101 的 ZMQ 端口。

## 当前实现与假设

- 复制 11 个 STL：`base_link.stl`、`link1.stl` 到 `link6.stl`，以及
  `flange.stl`、`gripper_base.stl`、`gripper_link1.stl`、`gripper_link2.stl`。
  源目录保持原样。模型、STL、来源清单及许可证在 `PIPER/assets/`。
- 从标准 Piper URDF 导入关节原点、轴、限位和惯性，生成相对 mesh 路径的 MJCF。
  DAE 外观改用对应 STL，因此不保留原 DAE 的材质颜色。
- 六个臂关节为 `joint1`…`joint6`，不把其中任何一个当夹爪。
- 演示采用**带平行夹爪的 Piper**，两片手指固定在 8mm 握笔开度。
  笔尖 site 为 `pen_tip`，距 `link6` 原点 0.18m（沿局部 +Z）。
  夹爪作为固定工具参与惯性计算，不增加学习模型的控制维度；没有模拟真实抓握接触。
- 状态 `x=[pen_x,pen_y,pen_z,q1,...,q6]`，9 维；动作是 6 维 rad/s 目标速度。
  物理步长 0.002s，控制周期 0.02s，动作限制 ±0.5rad/s。
- 使用理想重力补偿。速度执行器增益和力矩上限是仿真参数，未经真机辨识。
- 当前禁用机器人与纸面的碰撞/接触；纸与墨迹是可视化。
  因而这验证的是**书写轨迹跟踪**，不是接触压力、真实墨迹或避碰能力。
- IK 同时约束笔尖位置与完整朝向，笔局部 +Z 朝世界 -Z。
  各笔画之间生成抬笔、平移、落笔，段内使用五次时间缩放，拐点停稳。
- 不沿用 SO-101 的 wrist_roll=-90°、joint_offsets、checkpoint 或真机驱动。

## 使用已有环境

所有命令在项目根目录执行：

```bash
conda activate koopman-sim2real
cd /home/cym/ROS/Arms/isaac_so_arm101/Lerobot-mujoco-sim2real
```

本机已验证该环境包含 MuJoCo 3.2.3、NumPy、SciPy、PyTorch、CasADi、Matplotlib、
Gymnasium、TensorBoard 等既有训练依赖，无需重新安装 ROS 或 LeRobot。

新机器克隆本仓库后，另外下载汉字笔画数据到同级目录：

```bash
git clone https://github.com/chanind/hanzi-writer-data.git ../hanzi-writer-data
```

Piper MJCF、11 个 STL 以及 `results/PIPER/piper_gripper_demo/DBKN/` 的示例权重
和兼容性元数据随仓库提供，可以直接运行。`PIPER/data/`、`PIPER/outputs/`、
其他新训练结果和 TensorBoard 日志被 `.gitignore` 排除；文中截图/实验数据路径
指本地生成物，克隆后运行相应命令即可重新生成。原始 `agx_arm_sim` 仅在重新
执行 `prepare_assets` 时需要，不是运行已提供 MJCF 的前提。

## 1. 复制资源并生成 MJCF

已经完成；需要重建时：

```bash
python -m PIPER.prepare_assets \
  --source /home/cym/ROS/Arms/agx_arm_sim
```

默认目标是 `PIPER/assets/`，路径结构：

```text
PIPER/
  assets/
    meshes/{base_link,link1,link2,link3,link4,link5,link6}.stl
    meshes/{flange,gripper_base,gripper_link1,gripper_link2}.stl
    piper.xml                  # 仿真、IK、数据采集共用
    piper_source.urdf           # 相对 mesh 路径的源模型副本
    source_manifest.json       # 来源路径、STL SHA256
    SOURCE_LICENSE
  env.py                       # 六关节动态环境
  trajectory.py                # 汉字轨迹、笔尖姿态 IK
  data_collection.py           # 任务附近数据采样与缓存
  write.py                     # plan / baseline / mpc 入口
  test_workflow.py
  data/reference.npz           # 当前任务参考，plan 生成
  data/<fingerprint>/          # 动力学训练数据
  outputs/                    # 轨迹、误差图、JSON 指标
```

资源脚本支持 `--destination`，但运行入口默认读取 `PIPER/assets/piper.xml`；
如改变资源目标，还需同步 `PIPER/env.py` 的 MODEL 和 `args.py` 的 PIPER 资产映射。
建议先保持默认目录。生成脚本会覆盖目标目录内同名的生成资源，不改变源模型。

## 2. 不训练，先看 Piper 写字

```bash
python -m PIPER.write --mode baseline --text 二
python -m PIPER.write --mode baseline --text 永
```

在有桌面的终端运行会打开 MuJoCo，机械臂沿笔画移动，落笔段显示实际笔尖轨迹。
运行结束后默认窗口关闭，添加 `--hold` 可停留查看，结果保留在 `PIPER/outputs/`。

按原 SO-101 脚本配色：**红色散点 = 完整参考轨迹，绿色线 = 实际笔尖历史轨迹**。
两者同时显示；落笔段不透明，抬笔/转移段半透明。红点按空间距离抽样，
防止高精度跟踪时红色覆盖绿色；两条轨迹不人为错位。
保存的二维对照图也使用红/绿配色。

```bash
python -m PIPER.write --mode baseline --text 永 --hold
# 无桌面导出带夹爪和双色轨迹的三维截图：
MUJOCO_GL=egl python -m PIPER.write --mode baseline --text 永 --headless \
  --snapshot PIPER/outputs/piper_gripper_trails.png
```
没有桌面时加 `--headless`，仍生成 PNG、NPZ 和 JSON。

这里 baseline 是参考关节速度前馈 + 关节位置反馈，**不使用学习模型**。
每步通过速度执行器和 `mj_step` 移动；只有开头设置一次 Home 姿态，
不再直接设置为第一笔位置。进场结束后沿用实际状态继续落笔、书写。
它用于确认资产、运动学和笔画生成正确，不能当成 DBKN/MPC 的结果。

默认设置：字框 8cm，中心世界坐标 `(0.30,0)`m，纸面 `z=0`，抬笔 3cm。
纸面高度相对于机器人基座坐标系，不是相对于笔的法兰。

```bash
python -m PIPER.write --mode baseline --text 永 \
  --size 0.08 --center 0.30 0 --height 0 --lift 0.03 --speed 0.025
```

`--speed` 是每段最大笛卡尔速度，单位 m/s；拐点会减速到零。
`--text` 支持多个字，但总宽度必须可达。不可达、姿态误差过大或参考关节速度
超过 0.4rad/s 时程序中止，不静默使用失败的 IK 解。

### 初始位置与 5 秒进场

默认 Home 是 `joint1..joint6 = [0,0,0,0,0,0]°`，对应原 Piper MoveIt 的
`initial_positions.yaml`，定义在 `PIPER/startup.py:HOME_DEG`。
带夹爪和笔的当前模型在此姿态下，笔尖世界坐标约为
`[0.235450, 0, 0.228867]m`（基座原点与世界原点重合）。
这不是实机反馈读数，真机当前姿态可能不同。

现在 baseline 和 mpc 都默认执行：

1. 在 Home 启动，图形窗口展示 1 秒（`--start-hold`）。
2. 在 5 秒内平滑进场（`--approach-seconds`），到第一笔起点上方。
3. 检查关节位置误差与实际速度，然后沿原轨迹落笔、开始书写。

目标是第一笔**上方** `--lift` 高度，不是在 5 秒末直接压到纸面。
例如默认“永”字进场目标约为 `[0.2934375,0.0340625,0.03]m`；之后再落到 `z=0`。
六关节同步插值，使用余弦加减速的速度曲线，位置、速度和加速度连续，
参考起终速度和加速度为零。若指定起点和时长需要超出速度余量，则报错提示延长时间。
执行使用关节速度前馈和位置反馈；MPC 仅在后续书写区域启用，不在零位附近外推。
到达误差超过 0.5°或关节速度超过 0.02rad/s 时停止，不继续书写。

只验证进场，在终点停留查看：

```bash
python -m PIPER.write --text 永 --approach-only --hold
```

正常完成进场并用已有 MPC 模型书写（模型无需重新训练）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m PIPER.write --mode mpc --text 永 --hold \
  --checkpoint results/PIPER/piper_gripper_demo/DBKN/best_model.pt
```

替换仿真起始姿态、延长初始展示或进场时长：

```bash
python -m PIPER.write --text 永 --approach-only --hold \
  --home-deg 0 10 -10 0 0 0 --start-hold 3 --approach-seconds 5
```

`--home-deg` 按 joint1 到 joint6 顺序传入角度，内部转换为弧度。
终端打印初始姿态、笔尖坐标、目标姿态、进场时长和到达误差。
红色参考轨迹、绿色实际轨迹会同时显示进场和写字，进场段使用半透明颜色。

结果单独保存为 `<字>_approach.npz` 与 `<字>_approach.json`：

- `time_s`：从 0 到 5 秒的 251 个采样点，250 次动力学控制步。
- `q_reference_rad` / `q_reference_deg`：六关节位置参考。
- `qvel_reference_rad_s`、`xyz_reference_m`：速度和笔尖位置参考。
- `actual_states`：实际 `[pen_xyz,q1..q6]`，位置单位 m、关节单位 rad。
- `actions_rad_s`：实际下发给仿真的 250 条速度指令。
- JSON 包含仿真时长、墙钟时长、到达误差和剩余速度。

GUI 模式按实际时间播放 5 秒；`--headless` 加速执行，但物理仿真仍推进 5 秒。
初始展示的 1 秒不计入进场时长。正常写字指标仍只计算书写阶段。

后续接入实机时，应将读取并转换到模型坐标系的实际关节角作为插值起点；
不能假定实机已经在零位。当前是无碰撞检测的关节空间进场，不是避障轨迹规划，
也尚未向真机发送指令。

## 3. 准备任务参考

```bash
python -m PIPER.write --mode plan --text 二
```

生成 `PIPER/data/reference.npz`。包含笔尖 XYZ、六关节参考、落笔标记、控制周期、
模型 SHA256。采集器围绕这些姿态采样，而非照搬 SO-101 的零位扰动区间。
重新执行 plan 会替换当前参考；之前的数据缓存和模型结果仍保留。

默认汉字数据路径从项目位置推导到同工作区 `hanzi-writer-data/data`。
其他位置通过 `--hanzi-data /path/to/data` 指定。

## 4. 重新训练 Piper DBKN

SO-101 的模型不能复用。Piper 接入了现有 `train.py`，`--env PIPER`
自动设置 `x_dim=9,u_dim=6`，不会修改其他机器人的默认维度。
当前写字 MPC 仅支持默认 DBKN；不要添加 `--u_z`、`--use_stable` 或切换为其他架构。

以下是本次端到端验证使用的小规模任务训练：

```bash
python train.py --env PIPER --model DBKN --mode train \
  --suffix piper_gripper_demo \
  --train_samples 1500 --train_steps 30 \
  --test_samples 32 --test_steps 60 \
  --num_epochs 100 --eval_interval 10 \
  --batch_size 128 --eval_batch_size 128 --device cpu
```

输出：`results/PIPER/piper_gripper_demo/DBKN/best_model.pt`。
同目录 `piper_data.json` 记录模型哈希、周期、数据来源和训练关节范围。

Piper 专用初始化将 A 设为单位矩阵，用任务中心的局部运动学导数初始化 B，
H 从零开始，然后由 MuJoCo 数据训练。这符合 50Hz 慢速状态接近恒等映射的特点，
不代表训练前已经辨识了真机动力学。

缓存指纹包含采样代码、环境代码、MJCF、STL、参考轨迹和随机种子。
修改模型或任务会使用不同缓存，避免仅换 suffix 却继续使用旧数据的问题。
XML 改动后，必须先重新执行 plan。

## 5. 用 DBKN + MPC 写字

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m PIPER.write --mode mpc --text 二 \
  --checkpoint results/PIPER/piper_gripper_demo/DBKN/best_model.pt
```

加 `--headless` 可离屏运行。线程环境变量用于减少小矩阵运算的调度开销。
带夹爪模型改变了末端惯量和笔尖偏移，旧 `piper_write_demo` checkpoint 与新模型不兼容。
本机已重新训练 `piper_gripper_demo`，请使用上述新路径。

MPC 通过学习的升维模型预测未来 10 步，采用当前状态冻结双线性 B 的近似，
并在优化过程中限制速度到 ±0.5rad/s；每步只执行第一条控制量。
目前优化没有机器人碰撞约束，不应直接用于硬件。

入口检查 checkpoint 的维度、模型 SHA256、周期和任务关节范围。
更换字、中心、笔长或纸面高度后，先运行 baseline，再 plan 和重新训练。
关节包围盒检查只是必要条件，不证明所有组合都在训练分布内。

结果文件：

- `<字>_mpc.png`：实际落笔轨迹对照，以及全程笔尖误差。
- `<字>_mpc.npz`：参考与实际位置、关节参考、落笔标记、动作、周期。
- `<字>_mpc.json`：RMSE、最大误差、笔倾角、完整执行标志、2mm 跟踪检查。

`completed` 只表示轨迹执行完毕，不能代替精度指标。
上述小规模训练适合验证整条管线，不代表跨工作空间泛化或真机精度。

### 带夹爪模型验证结果（2026-10-16）

`piper_gripper_demo/DBKN/best_model.pt` 完整执行“永”字 2354 个控制点：
笔尖 RMSE 0.167mm，最大误差 0.275mm，最大倾角 0.107°。
3 项集成测试通过，三维离屏渲染已检查夹爪与红/绿轨迹。
截图：`outputs/piper_gripper_trails.png`，数值结果：`outputs/gripper/永_mpc.json`。

### 旧版无夹爪模型验证结果（2026-10-16，仅作历史记录）

`piper_write_demo/DBKN/best_model.pt`，默认字框/中心/高度参数：

| 字 | 控制方法 | 完整控制点数 | 笔尖 RMSE | 最大位置误差 |
|---|---|---:|---:|---:|
| 二 | 速度前馈 + 位置反馈基线 | 1036 | 0.027 mm | 0.060 mm |
| 永 | 速度前馈 + 位置反馈基线 | 2354 | 0.031 mm | 0.087 mm |
| 二 | DBKN + MPC | 1036 | 0.127 mm | 0.180 mm |
| 永 | 同一个 DBKN + MPC | 2354 | 0.127 mm | 0.229 mm |

MPC 没有在这个理想速度控制任务上超过基线。这里验证的是训练、预测、控制与换字
流程可运行，不是证明学习控制比简单反馈更优。所有误差都包含抬笔/转移阶段。
“永”字使用围绕“二”字采样训练的同一个模型，只代表此次工作区内的一次换字测试。
图和原始结果在 `outputs/二_mpc.*` 与 `outputs/extra_character/永_mpc.*`。

## 6. 测试

```bash
python -m unittest PIPER.test_workflow -v
```

检查资源目录迁移后 MJCF/URDF 仍能加载、六关节维度、完整笔尖姿态 IK、
不可达目标拒绝、两笔之间抬笔以及基线动态跟踪误差。
模型训练与 MPC 闭环通过上面的完整命令另行验证。

## 与原 SO-101 路线的对应

| 原模块 | Piper 对应 |
|---|---|
| SOARM101_Env.py | PIPER/env.py，9 状态 / 6 动作 |
| SOARM101_DataCollection.py | PIPER/data_collection.py，复用 loader 接口 |
| train.py / models | 原训练循环和 DBKN，增加 PIPER 分支 |
| write_demo.py / TrajectoryGenerator | PIPER/write.py / trajectory.py |
| gripperframe + 固定 wrist_roll | pen_tip + 六维姿态 IK |
| 无显式速度约束的 MPC | Piper MPC 在优化内施加速度边界 |
| so101_real.py / ZMQ | 未连接，需要单独实现 Piper 硬件适配 |

## 若下一步要实机写字

现有 `agx_arm_sim` 主要提供描述和 MoveIt 假控制器，不能把它直接当 Piper 驱动。
应先明确实际 Piper 型号、固件和 CAN/ROS 驱动，再提供六关节状态与指令适配。
Piper 真机的位置指令单位必须按实际接口核实，不能使用 SO-101 的角度偏移表。

还需要标定基座到纸面变换、实际夹爪与笔的 TCP（不能直接采用演示的 0.18m 偏移），
并确定位置跟踪或力/柔顺控制方案。真机位置接口与本模型速度接口之间的动态差异
需要建模或重新采集辨识，不能仅换一个发送函数就宣称完成 Sim2Real。
本实现没有 UKF/KESO 抗扰实验、实体纸面压力控制或真实笔迹验证。
